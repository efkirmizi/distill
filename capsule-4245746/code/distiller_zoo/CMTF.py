import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorly as tl
from tensorly.decomposition import parafac

tl.set_backend('pytorch')


class CoupledTensorLoss(nn.Module):
    """
    Coupled Matrix-Tensor Factorization (CMTF) Loss.
    
    Part 1 - Activation Matching: Aligns spatial attention maps at each hint point.
    Part 2 - Structural Coupling: Decomposes teacher activations via CP, extracts
             spatial latent signatures (R×R), and aligns them with the student's
             structural weight factors obtained via get_factors().
    """
    def __init__(self, model=None, rank=16, iter_max=5, **kwargs):
        super(CoupledTensorLoss, self).__init__()
        self.rank = rank
        self.iter_max = iter_max
        self.model = model

        # Collect all structurally decomposed layers from the student model
        self.decomposed_layers = []
        if self.model is not None:
            self._find_decomposed_layers(self.model)

    def _find_decomposed_layers(self, module):
        """Recursively find all CPConv2d / TuckerConv2d layers."""
        # Unwrap DataParallel / torch.compile wrappers
        if hasattr(module, 'module'):
            module = module.module
        if hasattr(module, '_orig_mod'):
            module = module._orig_mod
        for child in module.children():
            if hasattr(child, 'get_factors') and callable(child.get_factors):
                self.decomposed_layers.append(child)
            else:
                self._find_decomposed_layers(child)

    def _frob_normalize(self, x):
        """Normalize a matrix by its Frobenius norm."""
        return x / (torch.norm(x, p='fro') + 1e-8)

    def _extract_spatial_factors(self, layer):
        """
        Extract height and width spatial factor matrices (2D) from a decomposed layer.

        CPConv2d.get_factors() returns:
          [0] pointwise_in  (rank, C_in, 1, 1)
          [1] depthwise_h   (rank, 1, kH, 1)   ← height spatial factor
          [2] depthwise_w   (rank, 1, 1, kW)    ← width spatial factor
          [3] pointwise_out (C_out, rank, 1, 1)

        TuckerConv2d.get_factors() returns:
          [0] pointwise_in  (rank_in, C_in, 1, 1)
          [1] core_conv     (rank_out, rank_in, kH, kW)  ← spatial info baked in core
          [2] pointwise_out (C_out, rank_out, 1, 1)
        """
        from decomposition import CPConv2d
        factors = layer.get_factors()

        if isinstance(layer, CPConv2d):
            # depthwise_h: (rank, 1, kH, 1) → squeeze → (rank, kH)
            f_h = factors[1].squeeze()
            # depthwise_w: (rank, 1, 1, kW) → squeeze → (rank, kW)
            f_w = factors[2].squeeze()
            # Handle edge case where squeeze collapses to 1D (rank=1 or kH=1)
            if f_h.dim() == 1:
                f_h = f_h.unsqueeze(0)
            if f_w.dim() == 1:
                f_w = f_w.unsqueeze(0)
            return f_h, f_w  # both (rank, spatial_dim)
        else:
            # TuckerConv2d: extract spatial from core (rank_out, rank_in, kH, kW)
            core = factors[1]
            # Mode-2 unfolding for height: (kH, rank_out*rank_in*kW)
            f_h = core.permute(2, 0, 1, 3).reshape(core.shape[2], -1).t()
            # Mode-3 unfolding for width: (kW, rank_out*rank_in*kH)
            f_w = core.permute(3, 0, 1, 2).reshape(core.shape[3], -1).t()
            return f_h, f_w  # (features, spatial_dim)

    def _align_and_compare(self, cov_a, cov_b):
        """
        Compare two covariance matrices that may have different sizes.
        Pools the larger one down to match the smaller, then computes MSE.
        """
        if cov_a.shape == cov_b.shape:
            return F.mse_loss(cov_a, cov_b)

        # Pool larger to smaller via adaptive_avg_pool2d
        target_size = min(cov_a.shape[0], cov_b.shape[0])
        if target_size < 2:
            return torch.tensor(0.0, device=cov_a.device)

        cov_a = F.adaptive_avg_pool2d(
            cov_a.unsqueeze(0).unsqueeze(0), (target_size, target_size)
        ).squeeze()
        cov_b = F.adaptive_avg_pool2d(
            cov_b.unsqueeze(0).unsqueeze(0), (target_size, target_size)
        ).squeeze()

        return F.mse_loss(self._frob_normalize(cov_a), self._frob_normalize(cov_b))

    def forward(self, f_s, f_t):
        loss = 0.0

        # ═══ Part 1: Spatial Attention Matching at EACH hint point ═══
        for s, t in zip(f_s, f_t):
            batch_size = s.shape[0]
            if s.shape[2:] != t.shape[2:]:
                s = F.adaptive_avg_pool2d(s, t.shape[2:])

            s_attention = F.normalize(s.pow(2).mean(1).view(batch_size, -1), dim=1)
            t_attention = F.normalize(t.pow(2).mean(1).view(batch_size, -1), dim=1)
            loss += F.mse_loss(s_attention, t_attention)

        # ═══ Part 2: Structural Coupling (Weight Factors ↔ Activation Factors) ═══
        if self.decomposed_layers and len(f_t) > 0:
            # Decompose teacher activations at EACH hint point (not just the deepest)
            for t_feat in f_t:
                try:
                    _, factors_t = parafac(
                        t_feat, rank=self.rank, init='random',
                        n_iter_max=self.iter_max, tol=1e-4
                    )
                    # Spatial factors: Height (H, R) and Width (W, R)
                    t_h, t_w = factors_t[2], factors_t[3]

                    # Teacher latent signatures (R × R)
                    cov_t_h = self._frob_normalize(torch.mm(t_h.t(), t_h))
                    cov_t_w = self._frob_normalize(torch.mm(t_w.t(), t_w))

                    # Compare with each decomposed layer's weight spatial factors
                    for layer in self.decomposed_layers:
                        f_h, f_w = self._extract_spatial_factors(layer)

                        # Student weight latent signatures (R_s × R_s)
                        cov_w_h = self._frob_normalize(torch.mm(f_h, f_h.t()))
                        cov_w_w = self._frob_normalize(torch.mm(f_w, f_w.t()))

                        # Rank-agnostic comparison (handles R ≠ R_s)
                        loss += self._align_and_compare(cov_w_h, cov_t_h)
                        loss += self._align_and_compare(cov_w_w, cov_t_w)

                except Exception:
                    # Fallback: Part 1 attention matching still provides the gradient
                    pass

        return loss
