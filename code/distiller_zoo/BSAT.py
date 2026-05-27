import torch
import torch.nn as nn
import torch.nn.functional as F


class CoupledTensorLoss(nn.Module):
    """
    BSAT distillation loss with two parts.

    Part 1 — Spatial Attention Matching (AT loss):
        Per-sample spatial attention maps: square activations → channel-mean →
        L2-normalise → MSE. Stable and batch-size-independent.

    Part 2 — Batch-Subspace Alignment (BSA), selected by ``align_mode``:
        * 'projector' (default): batch-unfold each feature map into B×(C·H·W),
          form the Gram matrix G = M Mᵀ (B×B), and align the rank-R orthogonal
          projector P = U Uᵀ obtained from torch.linalg.eigh(G). Sign-invariant.
        * 'gram': skip the eigendecomposition entirely and align the cosine Gram
          (similarity) matrix Ĝ = M̂ M̂ᵀ ∈ [−1,1]^{B×B} of the row-normalised
          batch matrix. Full-rank, eigengap-free, sign-invariant (this is the
          Similarity-Preserving formulation). Avoids the 1/(λ_q − λ_{q+1})
          gradient blow-up that destabilises the projector path on the small,
          noisy batches produced by CIFAR-scale extreme compression.

    ``proj_stable`` (projector mode only): run the Gram + eigh in float64, use a
        relative diagonal jitter, normalise the BSA MSE by the subspace rank q,
        and fall back to a zero-gradient projector if eigh returns non-finite
        values — so one ill-conditioned batch cannot poison the optimiser step.

    Coupling term (dual-student mode):
        When ``coupling_proj`` is supplied, this student's B×B matrix is pulled
        toward the peer's (detached) by coupling_weight·MSE. Works on whichever
        B×B matrix the active mode produces (projector or Gram). NOTE: the
        current training loops apply coupling themselves (so they can warm it up
        and route it through the dynamic weighter); this in-forward path is kept
        for backward compatibility.
    """

    def __init__(self, rank=8, coupling_weight=1.0, align_mode='projector',
                 proj_stable=False, **kwargs):
        super(CoupledTensorLoss, self).__init__()
        self.rank = rank
        self.coupling_weight = coupling_weight
        self.align_mode = align_mode
        self.proj_stable = proj_stable

    def _gram(self, feat):
        """Cosine Gram (similarity) matrix Ĝ = M̂ M̂ᵀ ∈ [-1,1]^{B×B}; no eigh."""
        B = feat.shape[0]
        M = feat.reshape(B, -1).float()
        M = F.normalize(M, dim=1, eps=1e-6)      # unit-norm rows; eps guards dead features
        return M @ M.t()                         # B × B, smooth gradient everywhere

    def _projection(self, feat):
        """
        Rank-R batch-mode orthogonal projector P = U Uᵀ (B×B) from a feature map.

        The eigenvectors of the Gram matrix G = M Mᵀ are the left singular vectors
        of the batch-unfolded matrix M (B×D); we eigendecompose G with the
        symmetric solver, which is far more stable than an SVD path on
        near-constant early-training features.
        """
        B = feat.shape[0]
        M = feat.reshape(B, -1).float()          # B × D, force fp32
        q = min(self.rank, B, M.shape[1])

        # Frobenius-normalise: scaling M by a constant leaves U unchanged but
        # keeps the Gram matrix well-scaled.
        norm = M.norm()
        if norm > 0:
            M = M / norm

        if self.proj_stable:
            # float64 eigh + relative jitter + NaN-safe fallback.
            with torch.amp.autocast('cuda', enabled=False):
                Md = M.double()
                G = Md @ Md.t()
                jitter = torch.clamp(1e-4 * G.diagonal().mean(), min=1e-6)
                G = G + jitter * torch.eye(B, device=G.device, dtype=G.dtype)
                try:
                    U = torch.linalg.eigh(G).eigenvectors[:, -q:]
                    ok = bool(torch.isfinite(U).all())
                except Exception:
                    ok = False
                if not ok:
                    # zero-gradient projector: this hint contributes no signal
                    # this batch, instead of NaNs that skip the whole update.
                    return torch.zeros(B, B, device=feat.device, dtype=feat.dtype)
            return (U @ U.t()).to(feat.dtype)

        # Default path — unchanged numerics (preserves the ImageNet baseline).
        with torch.amp.autocast('cuda', enabled=False):
            G = M @ M.t()                            # B × B, symmetric PSD
            # Diagonal jitter: prevents eigh from failing on ill-conditioned G
            # (near-duplicate eigenvalues cause LAPACK error 257 late in training).
            G = G + 1e-6 * torch.eye(B, device=G.device, dtype=G.dtype)
            # eigh returns eigenvalues in ascending order; take the last q columns.
            U = torch.linalg.eigh(G).eigenvectors[:, -q:]   # B × q
        return U @ U.t()                             # B × B

    def forward(self, f_s, f_t, coupling_proj=None):
        """
        Args:
            f_s           : list of student feature tensors, one per hint point
            f_t           : list of teacher feature tensors, one per hint point
            coupling_proj : optional list of B×B matrices from the peer student
                            (detached); when provided, adds the coupling term.

        Returns:
            loss  : scalar distillation loss (AT + BSA [+ coupling])
            proj  : list of B×B matrices for f_s, one per hint point (rank-R
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
                bsa_k = F.mse_loss(P_s, P_t)
                if self.proj_stable:
                    q = min(self.rank, B, s.reshape(B, -1).shape[1])
                    bsa_k = bsa_k / max(q, 1)    # scale-comparable across batch sizes
                loss_bsa = loss_bsa + bsa_k
                proj.append(P_s)

        loss = loss_at + loss_bsa

        # ── In-forward coupling (kept for compatibility; loops apply their own) ──
        if coupling_proj is not None:
            for P_s, P_peer in zip(proj, coupling_proj):
                loss = loss + self.coupling_weight * F.mse_loss(P_s, P_peer)

        return loss, proj, {'at': loss_at, 'bsa': loss_bsa}
