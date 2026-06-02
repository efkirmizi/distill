import torch
import torch.nn as nn
import torch.nn.functional as F


class CoupledTensorLoss(nn.Module):
    """
    BSAT distillation loss (merged): adaptive soft-spectral batch-subspace
    alignment, with a Similarity-Preserving 'gram' fallback.

    Part 1 — Spatial Attention Matching (AT):
        Per-sample squared-activation attention maps → channel-mean →
        L2-normalise → MSE. Stable and batch-size independent.

    Part 2 — Batch-Subspace Alignment (BSA), selected by ``align_mode``:
      * 'projector' (default): adaptive SOFT spectral projector. Batch-unfold
        each feature map to M (B×D), take the left singular subspace of M
        (eigenvectors of G = M Mᵀ), and build P = U diag(w) Uᵀ where
          - the effective rank q is set by an ENERGY threshold (smallest q whose
            top eigenvalues capture ``energy`` of trace(G)), capped by ``rank``.
            This decouples the constraint stiffness from batch size — the cause
            of the CIFAR small-batch collapse — so ``rank`` is only a hard cap.
          - w is a smooth softmax over the retained eigenvalues (temperature
            ``soft_temp``); there is no hard 0/1 cutoff, so eigenvalue crossings
            no longer spike the gradient.
        P is exactly sign-invariant ((-u)(-u)ᵀ = u uᵀ) and continuous in feat.
        With energy=1.0 and soft_temp→∞ (large value, e.g., 100.0) this recovers
        the original hard-rank-R projector (uniform softmax weights → P = UUᵀ),
        so the v1 behaviour is a strict special case.
      * 'gram': eigh-free cosine similarity matrix Ĝ = M̂ M̂ᵀ ∈ [-1,1]^{B×B}
        (Similarity-Preserving). Kept for ablation; it over-constrains tiny
        students on the hardest configs, so it is not the default.

    ``decomp`` ('eigh' | 'svd'): how the left singular subspace is obtained in
        projector mode. 'eigh' forms G = M Mᵀ (B×B) and eigendecomposes it
        (cheap, default). 'svd' decomposes M directly, avoiding the squared
        condition number that forming G incurs — better-conditioned directions
        near the energy boundary — at higher compute/memory (works on B×D).
        The soft weighting removes the *gradient* discontinuity either way, so
        SVD mainly sharpens the forward subspace estimate.

    ``proj_stable``: run the decomposition in float64 with a relative diagonal
        jitter and a NaN-safe fallback (zero-gradient projector), so a single
        ill-conditioned batch cannot poison the optimiser step.

    Coupling (dual-student): when ``coupling_proj`` is supplied, this student's
        B×B matrix is pulled toward the peer's (detached) by coupling_weight·MSE.
        Dimension-agnostic (always B×B). NOTE: the training loops apply coupling
        and BSA warmup themselves (so they can route AT/BSA through the dynamic
        weighter); this in-forward path is kept for backward compatibility.

    Returns ``(loss, proj, {'at', 'bsa'})`` so the loop can weight AT and BSA
    independently and warm up / scale the subspace term.
    """

    def __init__(self, rank=8, coupling_weight=1.0, align_mode='projector',
                 decomp='eigh', energy=0.9, soft_temp=0.25, proj_stable=False,
                 **kwargs):
        super(CoupledTensorLoss, self).__init__()
        self.rank = rank
        self.coupling_weight = coupling_weight
        self.align_mode = align_mode
        self.decomp = decomp
        self.energy = energy
        self.soft_temp = soft_temp
        self.proj_stable = proj_stable

    def _gram(self, feat):
        """Cosine Gram (similarity) matrix Ĝ = M̂ M̂ᵀ ∈ [-1,1]^{B×B}; no eigh/svd."""
        B = feat.shape[0]
        M = feat.reshape(B, -1).float()
        M = F.normalize(M, dim=1, eps=1e-6)      # unit-norm rows; eps guards dead features
        return M @ M.t()                         # B × B, smooth gradient everywhere

    def _spectrum(self, M):
        """
        Ascending eigenvalues of G = M Mᵀ and the matching left singular vectors,
        via eigh(M Mᵀ) or svd(M). M is expected pre-scaled. Returns (evals, evecs).
        """
        B = M.shape[0]
        if self.decomp == 'svd' and not self.proj_stable:
            # Left singular vectors of M == eigenvectors of G; SVD avoids forming
            # G (which squares the condition number). torch.linalg.svd sorts
            # descending, so flip to match eigh's ascending convention.
            # When proj_stable=True we fall through to the jittered eigh path
            # regardless of decomp — SVD offers no jitter protection for backward.
            U, S, _ = torch.linalg.svd(M, full_matrices=False)
            evecs = torch.flip(U, dims=[1])
            evals = torch.flip(S, dims=[0]).pow(2)        # eigenvalues of G = σ²
            return evals, evecs
        G = M @ M.t()
        if self.proj_stable:
            # Relative jitter scales with the matrix; robust on degenerate spectra.
            jitter = torch.clamp(1e-4 * G.diagonal().mean(), min=1e-6)
        else:
            jitter = G.new_tensor(1e-6)
        G = G + jitter * torch.eye(B, device=G.device, dtype=G.dtype)
        evals, evecs = torch.linalg.eigh(G)               # ascending
        return evals, evecs

    def _projection(self, feat):
        """Adaptive soft batch-mode projector P = U diag(w) Uᵀ (B×B)."""
        B = feat.shape[0]
        M = feat.reshape(B, -1).float()          # B × D, force fp32
        D = M.shape[1]

        # Frobenius-normalise: scaling M by a constant leaves U unchanged but
        # keeps the Gram matrix well-scaled (trace(G) = 1).
        norm = M.norm()
        if norm > 0:
            M = M / norm

        dtype = torch.float64 if self.proj_stable else torch.float32
        with torch.amp.autocast('cuda', enabled=False):
            Mc = M.to(dtype)
            try:
                evals, evecs = self._spectrum(Mc)
                ok = bool(torch.isfinite(evals).all()) and bool(torch.isfinite(evecs).all())
            except Exception:
                ok = False
            if not ok:
                # NaN-safe: this hint contributes no signal this batch, instead
                # of propagating NaNs that would skip the whole optimiser step.
                return torch.zeros(B, B, device=feat.device, dtype=feat.dtype)

            ev = evals.clamp(min=0.0)
            total = ev.sum()
            if total <= 0:
                # degenerate (near-constant features): zero-gradient projector.
                return torch.zeros(B, B, device=feat.device, dtype=feat.dtype)

            # ── Batch-adaptive effective rank via energy threshold ──
            cum = torch.cumsum(torch.flip(ev, dims=[0]), dim=0) / total   # descending energy
            q_energy = int(torch.searchsorted(cum, cum.new_tensor(self.energy)).item()) + 1
            q = max(1, min(q_energy, self.rank, B, D))

            top_evals = ev[-q:]                  # ascending
            top_evecs = evecs[:, -q:]            # B × q

            # ── Soft spectral weights: smooth, sum to q (matches a rank-q trace) ──
            logits = top_evals / (top_evals.max() * self.soft_temp + 1e-12)
            w = F.softmax(logits, dim=0) * q
            P = (top_evecs * w.unsqueeze(0)) @ top_evecs.t()
        return P.to(feat.dtype)

    def forward(self, f_s, f_t, coupling_proj=None):
        """
        Args:
            f_s           : list of student feature tensors, one per hint point
            f_t           : list of teacher feature tensors, one per hint point
            coupling_proj : optional list of B×B matrices from the peer student
                            (detached); when provided, adds the coupling term.

        Returns:
            loss  : scalar distillation loss (AT + BSA [+ coupling])
            proj  : list of B×B matrices for f_s, one per hint point (soft
                    projector in 'projector' mode, cosine Gram in 'gram' mode).
                    Pass these (detached) as coupling_proj to the peer student.
            parts : dict {'at': L_AT, 'bsa': L_BSA} for separate dynamic weighting.
        """
        device = f_s[0].device if f_s else torch.device('cpu')
        loss_at = torch.zeros((), device=device)
        loss_bsa = torch.zeros((), device=device)
        proj = []

        for s, t in zip(f_s, f_t):
            B = s.shape[0]

            # ── Part 1: Spatial Attention Matching ───────────────────────────
            s_in = s
            if s_in.shape[2:] != t.shape[2:]:
                s_in = F.adaptive_avg_pool2d(s_in, t.shape[2:])
            s_att = F.normalize(s_in.pow(2).mean(1).view(B, -1), dim=1)
            t_att = F.normalize(t.pow(2).mean(1).view(B, -1), dim=1)
            loss_at = loss_at + F.mse_loss(s_att, t_att)

            # ── Part 2: Batch-Subspace Alignment ─────────────────────────────
            if self.align_mode == 'gram':
                M_t = self._gram(t.detach())     # no grad through teacher
                M_s = self._gram(s)              # grads flow to student
                loss_bsa = loss_bsa + F.mse_loss(M_s, M_t)
                proj.append(M_s)
            else:
                P_t = self._projection(t.detach())
                P_s = self._projection(s)
                loss_bsa = loss_bsa + F.mse_loss(P_s, P_t)
                proj.append(P_s)

        loss = loss_at + loss_bsa

        # ── In-forward coupling (kept for compatibility; loops apply their own) ──
        if coupling_proj is not None:
            for P_s, P_peer in zip(proj, coupling_proj):
                loss = loss + self.coupling_weight * F.mse_loss(P_s, P_peer)

        return loss, proj, {'at': loss_at, 'bsa': loss_bsa}
