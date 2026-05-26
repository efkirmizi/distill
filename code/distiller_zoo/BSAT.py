import torch
import torch.nn as nn
import torch.nn.functional as F


class CoupledTensorLoss(nn.Module):
    """
    BSAT distillation loss with two parts:

    Part 1 — Spatial Attention Matching (AT loss):
        Aligns per-sample spatial attention maps between student and teacher
        at each hint point. Sign of activations squared → channel-mean →
        L2-normalised → MSE. Weighted by at_weight.

    Part 2 — Batch-Subspace CKA:
        Batch-unfolds each feature map into a B×(C·H·W) matrix M, computes
        orthonormal left singular vectors U via torch.linalg.svd (direct SVD
        on M instead of eigendecomposing G=MMᵀ, halving the condition number),
        then minimises the subspace CKA loss: 1 − ‖U_s^T U_t‖_F² / rank.
        Bounded in [0,1]; scale-invariant; no LAPACK convergence issues.
        Weighted by bsa_weight.

    Coupling term (dual-student mode):
        When coupling_basis is supplied (CP student's SVD bases, detached),
        the Tucker student is additionally pulled toward the CP student's
        batch subspace via the same CKA loss. Weighted by coupling_weight.
    """

    def __init__(self, rank=8, at_weight=1.0, bsa_weight=0.5,
                 coupling_weight=1.0, **kwargs):
        super(CoupledTensorLoss, self).__init__()
        self.rank = rank
        self.at_weight = at_weight
        self.bsa_weight = bsa_weight
        self.coupling_weight = coupling_weight

    def _svd_basis(self, feat):
        """
        Rank-R orthonormal basis from a feature map via direct SVD.

        feat : B × C × H × W  (any shape with batch first)
        returns : B × q  matrix U, the top-q left singular vectors of the
                  batch-unfolded matrix B×D.

        Uses torch.linalg.svd directly on M rather than eigendecomposing
        G = M Mᵀ. This halves the effective condition number (κ(M) vs
        κ(M)²) and avoids eigenvector instability when two singular values
        are nearly equal late in training.
        """
        B = feat.shape[0]
        M = feat.reshape(B, -1).float()           # B × D, force fp32
        q = min(self.rank, B, M.shape[1])

        norm = M.norm()
        if norm > 0:
            M = M / norm

        # Disable autocast: svd requires fp32.
        with torch.amp.autocast('cuda', enabled=False):
            U, _, _ = torch.linalg.svd(M, full_matrices=False)
            U = U[:, :q]                          # B × q

        return U

    @staticmethod
    def _cka_loss(U_s, U_t):
        """
        Subspace CKA loss: 1 − ‖U_s^T U_t‖_F² / q.

        U_s, U_t : B × q orthonormal matrices (left singular vectors).
        Returns a scalar in [0, 1]; 0 when both subspaces are identical.
        ‖U_s^T U_t‖_F² equals the sum of squared cosines of the principal
        angles between the two rank-q subspaces.
        """
        q = U_s.shape[1]
        overlap = torch.mm(U_s.t(), U_t)          # q × q
        return 1.0 - (overlap ** 2).sum() / q

    def forward(self, f_s, f_t, coupling_basis=None):
        """
        Args:
            f_s            : list of student feature tensors (channel-adapted
                             via ConvReg), one per hint point
            f_t            : list of teacher feature tensors, one per hint point
            coupling_basis : list of B×q SVD-basis matrices from the CP student
                             (detached). When provided, adds CKA coupling term.

        Returns:
            loss  : scalar distillation loss
            basis : list of B×q SVD bases for f_s, one per hint point.
                    Pass these (detached) as coupling_basis to Tucker student.
        """
        device = f_s[0].device if f_s else torch.device('cpu')
        loss = torch.zeros(1, device=device).squeeze()
        basis = []

        for s, t in zip(f_s, f_t):
            B = s.shape[0]

            # ── Part 1: Spatial Attention Matching ───────────────────────────
            s_in = s
            if s_in.shape[2:] != t.shape[2:]:
                s_in = F.adaptive_avg_pool2d(s_in, t.shape[2:])
            s_att = F.normalize(s_in.pow(2).mean(1).view(B, -1), dim=1)
            t_att = F.normalize(t.pow(2).mean(1).view(B, -1), dim=1)
            loss = loss + self.at_weight * F.mse_loss(s_att, t_att)

            # ── Part 2: Batch-Subspace CKA ────────────────────────────────────
            U_t = self._svd_basis(t.detach())     # no grad through teacher
            U_s = self._svd_basis(s)              # grads flow to student
            loss = loss + self.bsa_weight * self._cka_loss(U_s, U_t)
            basis.append(U_s)

        # ── Coupling term (Tucker ← CP, CP side detached) ────────────────────
        if coupling_basis is not None:
            for U_s, U_cp in zip(basis, coupling_basis):
                loss = loss + self.coupling_weight * self._cka_loss(U_s, U_cp.detach())

        return loss, basis
