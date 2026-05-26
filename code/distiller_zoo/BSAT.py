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

    Part 2 — Batch-Subspace CKA (on attention maps):
        Computes spatial attention maps A = (x²).mean(C) → B×(H·W), then
        takes orthonormal left singular vectors U of A via torch.linalg.svd,
        and minimises: 1 − ‖U_s^T U_t‖_F² / rank.
        Operating in attention space (B×H·W) instead of raw feature space
        (B×C·H·W) makes BSA channel-count-agnostic: the matrix is always
        well-conditioned even under extreme compression where C is tiny.
        Bounded in [0,1]; scale-invariant. Weighted by bsa_weight.

    Coupling term (dual-student mode):
        Bidirectional CKA coupling between CP and Tucker student subspaces is
        handled by the training loop (loops.py / train_stu_imagenet100.py) after
        both forward passes complete. Weighted by coupling_weight.
    """

    def __init__(self, rank=8, at_weight=1.0, bsa_weight=0.5,
                 coupling_weight=1.0, **kwargs):
        super(CoupledTensorLoss, self).__init__()
        self.rank = rank
        self.at_weight = at_weight
        self.bsa_weight = bsa_weight
        self.coupling_weight = coupling_weight

    def _svd_basis(self, att):
        """
        Rank-R orthonormal basis from a B×D attention map matrix via direct SVD.

        att : B × D  (pre-computed, L2-normalised spatial attention map)
        returns : B × q  matrix U, the top-q left singular vectors.
        """
        M = att.float()                           # B × D, force fp32
        q = min(self.rank, M.shape[0], M.shape[1])

        norm = M.norm()
        if norm > 0:
            M = M / norm

        # Disable autocast: svd requires fp32.
        with torch.amp.autocast('cuda', enabled=False):
            try:
                U, _, _ = torch.linalg.svd(M, full_matrices=False)
            except torch._C._LinAlgError:
                # Ill-conditioned matrix: add jitter and retry once
                M = M + 1e-4 * torch.randn_like(M)
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

    def forward(self, f_s, f_t):
        """
        Args:
            f_s : list of student feature tensors (channel-adapted via ConvReg),
                  one per hint point
            f_t : list of teacher feature tensors, one per hint point

        Returns:
            loss  : scalar distillation loss
            basis : list of B×q left-singular-vector matrices, one per hint point.
                    Pass these (detached) as coupling targets to the Tucker student.
        """
        device = f_s[0].device if f_s else torch.device('cpu')
        loss = torch.zeros(1, device=device).squeeze()
        basis = []

        for s, t in zip(f_s, f_t):
            B = s.shape[0]

            # Shared L2-normalised attention maps: B × (H·W), channel-agnostic
            s_att = F.normalize(s.pow(2).mean(1).view(B, -1), dim=1)
            t_att = F.normalize(t.pow(2).mean(1).view(B, -1), dim=1)

            # ── Part 1: Spatial Attention Matching ───────────────────────────
            loss = loss + self.at_weight * F.mse_loss(s_att, t_att)

            # ── Part 2: Batch-Subspace CKA (in normalised attention space) ───
            U_t = self._svd_basis(t_att.detach())  # no grad through teacher
            U_s = self._svd_basis(s_att)            # grads flow to student
            loss = loss + self.bsa_weight * self._cka_loss(U_s, U_t)
            basis.append(U_s)

        return loss, basis
