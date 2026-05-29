import torch
import torch.nn as nn


class DynamicLossWeighter(nn.Module):
    """
    Kendall et al. (CVPR 2018) homoscedastic uncertainty weighting for multi-task losses.

    Learns log-variance s_i = log(sigma_i^2) per task. Combined loss:
        L_total = sum_i [ exp(-s_i) * L_i + s_i ]

    Initialized at zero: exp(0)=1, s=0 → starts as the plain unweighted sum.
    log_vars should be optimized WITHOUT weight decay (separate param group).

    Stability: s_i is clamped to [-LOG_VAR_CLAMP, LOG_VAR_CLAMP] inside the
    forward. Without this bound a task whose loss collapses toward zero drives
    its s_i → -inf, which both (a) sends the +s_i regularizer to -inf (the
    total loss goes negative and keeps falling) and (b) overflows exp(-s_i) to
    +inf → NaN. Clamping saturates the gradient at the bound (clamp has zero
    grad outside the range), so s_i settles instead of running away.
    """
    LOG_VAR_CLAMP = 10.0  # exp(10) ~ 2.2e4, exp(-10) ~ 4.5e-5: permissive but finite

    def __init__(self, num_losses: int = 3):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_losses))

    def _clamped_log_vars(self):
        return self.log_vars.clamp(-self.LOG_VAR_CLAMP, self.LOG_VAR_CLAMP)

    def forward(self, *losses):
        lv = self._clamped_log_vars()
        return sum(
            torch.exp(-lv[i]) * losses[i] + lv[i]
            for i in range(len(losses))
        )

    def effective_weights(self):
        """exp(-log_var) per task as a CPU Python list (for logging)."""
        with torch.no_grad():
            return torch.exp(-self._clamped_log_vars()).cpu().tolist()
