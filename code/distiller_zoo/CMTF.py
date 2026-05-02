import torch
import torch.nn as nn
import torch.nn.functional as F


class CoupledTensorLoss(nn.Module):
    """
    CMTF-guided co-distillation loss with two parts:

    Part 1 — Spatial Attention Matching (AT loss):
        Aligns per-sample spatial attention maps between student and teacher
        at each hint point. Sign of activations squared → channel-mean →
        L2-normalised → MSE. Established, well-grounded signal.

    Part 2 — Batch-Subspace Alignment:
        Batch-unfolds each feature map into a B×(C·H·W) matrix and computes
        the rank-R orthogonal projector P = U Uᵀ via truncated SVD.
        Minimises ‖P_student − P_teacher‖_F².  Sign-invariant (projector is
        unique); no ALS/parafac; gradients flow cleanly through svd_lowrank.

    Coupling term (dual-student mode):
        When coupling_proj is supplied (CP student's projectors, detached),
        the Tucker student is additionally pulled toward the CP student's
        batch subspace: coupling_weight · ‖P_tucker − P_cp‖_F².
        This enforces a shared semantic latent space between architecturally
        heterogeneous students without requiring matching channel/spatial dims.
    """

    def __init__(self, rank=8, coupling_weight=1.0, **kwargs):
        super(CoupledTensorLoss, self).__init__()
        self.rank = rank
        self.coupling_weight = coupling_weight

    def _projection(self, feat):
        """
        Rank-R batch-mode orthogonal projector from a feature map.

        feat : B × C × H × W  (any shape with batch first)
        returns : B × B  matrix  P = U Uᵀ,  where U holds the top-R left
                  singular vectors of the batch-unfolded matrix B×(C·H·W).
        """
        B = feat.shape[0]
        M = feat.reshape(B, -1).float()          # B × D, force fp32
        q = min(self.rank, B, M.shape[1])
        # svd_lowrank internally calls linalg.qr which is not implemented for fp16;
        # disable autocast locally so internal matmuls stay in fp32.
        with torch.amp.autocast('cuda', enabled=False):
            U = torch.svd_lowrank(M, q=q)[0]     # B × q
        return U @ U.t()                          # B × B

    def forward(self, f_s, f_t, coupling_proj=None):
        """
        Args:
            f_s           : list of student feature tensors, one per hint point
            f_t           : list of teacher feature tensors, one per hint point
            coupling_proj : list of B×B projection matrices from the CP student
                            (detached).  When provided, adds coupling term to loss.

        Returns:
            loss : scalar distillation loss
            proj : list of B×B batch-mode projectors for f_s, one per hint point.
                   Pass these (detached) as coupling_proj to the Tucker student.
        """
        device = f_s[0].device if f_s else torch.device('cpu')
        loss = torch.zeros(1, device=device).squeeze()
        proj = []

        for s, t in zip(f_s, f_t):
            B = s.shape[0]

            # ── Part 1: Spatial Attention Matching ───────────────────────────
            s_in = s
            if s_in.shape[2:] != t.shape[2:]:
                s_in = F.adaptive_avg_pool2d(s_in, t.shape[2:])
            s_att = F.normalize(s_in.pow(2).mean(1).view(B, -1), dim=1)
            t_att = F.normalize(t.pow(2).mean(1).view(B, -1), dim=1)
            loss += F.mse_loss(s_att, t_att)

            # ── Part 2: Batch-Subspace Alignment ─────────────────────────────
            P_t = self._projection(t.detach())    # no grad through teacher
            P_s = self._projection(s)             # grads flow to student
            loss += F.mse_loss(P_s, P_t)
            proj.append(P_s)

        # ── Coupling term (Tucker ← CP, CP side detached) ────────────────────
        if coupling_proj is not None:
            for P_s, P_cp in zip(proj, coupling_proj):
                loss += self.coupling_weight * F.mse_loss(P_s, P_cp)

        return loss, proj
