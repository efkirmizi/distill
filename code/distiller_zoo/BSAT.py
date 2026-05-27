import torch
import torch.nn as nn
import torch.nn.functional as F


class CoupledTensorLoss(nn.Module):
    """
    BSAT distillation loss (v2) with two parts:

    Part 1 — Spatial Attention Matching (AT loss):
        Aligns per-sample spatial attention maps between student and teacher
        at each hint point. Sign of activations squared → channel-mean →
        L2-normalised → MSE. Unchanged from v1.

    Part 2 — Adaptive Spectral Subspace Alignment (replaces hard-rank BSA):
        Batch-unfolds each feature map into a B×D matrix, forms the Gram
        matrix G = M Mᵀ (B×B), eigendecomposes it, and builds a SOFT
        projector P = U diag(w) Uᵀ where the weights w are a smooth
        function of the eigenvalue spectrum.

        Two changes vs. v1 fix the CIFAR small-batch collapse:

        (a) Batch-adaptive effective rank.  Instead of a fixed rank R, the
            number of "active" subspace directions is set by an energy
            threshold: the smallest q whose top-q eigenvalues capture
            `energy` fraction of trace(G).  At B=256 this naturally selects
            more directions; at B=64 fewer — the constraint stiffness no
            longer depends on the accident of batch size.  `rank` now acts
            only as a hard CAP, not a fixed target.

        (b) Soft spectral weighting.  Rather than a hard 0/1 cutoff at
            eigenvector index B-q (which makes near-equal eigenvalues swap
            across the boundary and spike gradients), each eigenvector is
            weighted by a smooth temperature-controlled softmax over its
            eigenvalue.  The projector is still exactly sign-invariant:
            (-u)(-u)ᵀ = u uᵀ.  But it is now continuous in the features,
            so eigenvalue crossings no longer produce gradient discontinuities.

        Setting `energy=1.0` and `soft_temp→0` recovers the original
        hard-rank-R behaviour, so v1 is a strict special case.

    Coupling term (dual-student mode):
        Unchanged.  When coupling_proj is supplied (peer student's soft
        projectors, detached), the student is additionally pulled toward the
        peer's batch subspace: coupling_weight · ‖P_self − P_peer‖_F².
    """

    def __init__(self, rank=8, coupling_weight=1.0,
                 energy=0.9, soft_temp=0.25, bsa_warmup_steps=0, **kwargs):
        """
        Args:
            rank             : hard CAP on the number of subspace directions
                               (q is never larger than this). The energy
                               threshold usually selects fewer.
            coupling_weight  : λ for the peer-coupling term.
            energy           : fraction of Gram-matrix trace the retained
                               directions must capture (0 < energy ≤ 1).
                               0.9 is a good default; 1.0 disables adaptivity.
            soft_temp        : temperature for the softmax spectral weights.
                               Smaller → sharper (closer to a hard top-q
                               projector); larger → smoother. ~0.25 works.
            bsa_warmup_steps : if > 0, the BSA term is linearly ramped from
                               0 to full weight over this many forward calls.
                               Lets the student build structured features
                               before the subspace constraint engages.
        """
        super(CoupledTensorLoss, self).__init__()
        self.rank = rank
        self.coupling_weight = coupling_weight
        self.energy = energy
        self.soft_temp = soft_temp
        self.bsa_warmup_steps = bsa_warmup_steps
        # non-trainable step counter; persists in state_dict, moves with .to()
        self.register_buffer('_step', torch.zeros(1, dtype=torch.long))

    def _bsa_scale(self):
        """Linear warmup multiplier for the BSA + coupling terms."""
        if self.bsa_warmup_steps <= 0:
            return 1.0
        s = float(self._step.item())
        return min(1.0, s / float(self.bsa_warmup_steps))

    def _projection(self, feat):
        """
        Adaptive soft batch-mode projector from a feature map.

        feat : B × C × H × W  (any shape with batch first)
        returns : B × B  matrix  P = U diag(w) Uᵀ.

        U holds the eigenvectors of the Gram matrix G = M Mᵀ (= left singular
        vectors of the batch-unfolded M). w is a smooth weight per direction:
        directions outside the energy budget get weight ~0, directions inside
        get a softmax-weighted contribution. The result is sign-invariant and
        continuous in `feat`.
        """
        B = feat.shape[0]
        M = feat.reshape(B, -1).float()          # B × D, force fp32
        D = M.shape[1]

        # Frobenius-normalise: scaling M by a constant leaves U unchanged
        # but keeps the Gram matrix well-scaled.
        norm = M.norm()
        if norm > 0:
            M = M / norm

        # Disable autocast: eigh requires fp32.
        with torch.amp.autocast('cuda', enabled=False):
            G = M @ M.t()                            # B × B, symmetric PSD
            # Diagonal jitter: keeps eigh stable on ill-conditioned G.
            G = G + 1e-6 * torch.eye(B, device=G.device, dtype=G.dtype)
            evals, evecs = torch.linalg.eigh(G)      # ascending eigenvalues

        # ── Batch-adaptive effective rank via energy threshold ───────────
        # Clamp tiny/negative eigenvalues (numerical) to zero before budgeting.
        ev = evals.clamp(min=0.0)
        total = ev.sum()
        if total <= 0:
            # degenerate (near-constant features): return identity-free zero
            return torch.zeros(B, B, device=feat.device, dtype=torch.float32)

        # Cumulative energy from the LARGEST eigenvalue downward.
        ev_desc = torch.flip(ev, dims=[0])                       # descending
        cum = torch.cumsum(ev_desc, dim=0) / total
        # smallest q with cum[q-1] >= energy
        q_energy = int(torch.searchsorted(cum, torch.tensor(
            self.energy, device=cum.device)).item()) + 1
        q = max(1, min(q_energy, self.rank, B, D))

        # Take the top-q directions (eigh sorts ascending → last q).
        top_evals = ev[-q:]                           # q,  ascending
        top_evecs = evecs[:, -q:]                     # B × q

        # ── Soft spectral weights ────────────────────────────────────────
        # Softmax over (eigenvalue / temperature), normalised so weights of
        # the retained directions sum to q (matches the trace of a hard
        # rank-q projector, keeping the loss scale comparable to v1).
        logits = top_evals / (top_evals.max() * self.soft_temp + 1e-12)
        w = F.softmax(logits, dim=0) * q              # q,  sums to q

        # P = U diag(w) Uᵀ  — sign-invariant, continuous in feat.
        P = (top_evecs * w.unsqueeze(0)) @ top_evecs.t()
        return P

    def forward(self, f_s, f_t, coupling_proj=None):
        """
        Args:
            f_s           : list of student feature tensors, one per hint point
            f_t           : list of teacher feature tensors, one per hint point
            coupling_proj : list of B×B projection matrices from the peer
                            student (detached). When provided, adds coupling.

        Returns:
            loss : scalar distillation loss
            proj : list of B×B soft batch-mode projectors for f_s, one per
                   hint point. Pass these (detached) as coupling_proj to the
                   peer student.
        """
        device = f_s[0].device if f_s else torch.device('cpu')
        loss = torch.zeros(1, device=device).squeeze()
        proj = []

        bsa_scale = self._bsa_scale()

        for s, t in zip(f_s, f_t):
            B = s.shape[0]

            # ── Part 1: Spatial Attention Matching ───────────────────────
            s_in = s
            if s_in.shape[2:] != t.shape[2:]:
                s_in = F.adaptive_avg_pool2d(s_in, t.shape[2:])
            s_att = F.normalize(s_in.pow(2).mean(1).view(B, -1), dim=1)
            t_att = F.normalize(t.pow(2).mean(1).view(B, -1), dim=1)
            loss += F.mse_loss(s_att, t_att)

            # ── Part 2: Adaptive Spectral Subspace Alignment ─────────────
            P_t = self._projection(t.detach())    # no grad through teacher
            P_s = self._projection(s)             # grads flow to student
            loss += bsa_scale * F.mse_loss(P_s, P_t)
            proj.append(P_s)

        # ── Coupling term (peer ← peer, target side detached) ────────────
        if coupling_proj is not None:
            for P_s, P_cp in zip(proj, coupling_proj):
                loss += bsa_scale * self.coupling_weight * F.mse_loss(P_s, P_cp)

        if self.training:
            self._step += 1

        return loss, proj