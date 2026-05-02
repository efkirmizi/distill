import torch
import torch.nn as nn


class DynamicLossWeighter(nn.Module):
    """
    Kendall et al. (CVPR 2018) homoscedastic uncertainty weighting for multi-task losses.

    Learns log-variance s_i = log(sigma_i^2) per task. Combined loss:
        L_total = sum_i [ exp(-s_i) * L_i + s_i ]

    Initialized at zero: exp(0)=1, s=0 → starts as the plain unweighted sum.
    log_vars should be optimized WITHOUT weight decay (separate param group).
    """
    def __init__(self, num_losses: int = 3):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_losses))

    def forward(self, *losses):
        return sum(
            torch.exp(-self.log_vars[i]) * losses[i] + self.log_vars[i]
            for i in range(len(losses))
        )

    def effective_weights(self):
        """exp(-log_var) per task as a CPU Python list (for logging)."""
        with torch.no_grad():
            return torch.exp(-self.log_vars).cpu().tolist()
