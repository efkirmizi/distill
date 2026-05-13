import gc
import torch
import torch.nn as nn
import tensorly as tl
from tensorly.decomposition import parafac, tucker

tl.set_backend('pytorch')


# ---------------------------------------------------------------------------
# Empirical VBMF rank estimation (Nakajima et al., JMLR 2013)
# ---------------------------------------------------------------------------

def _evbmf_rank(W_2d):
    """
    Estimate the rank of a 2-D weight matrix via Empirical Variational Bayes
    Matrix Factorization (Nakajima et al., JMLR 2013).

    Uses the Marchenko-Pastur distribution to threshold singular values.
    Iterates to jointly estimate noise variance σ² and rank R.

    Returns an int ≥ 1.
    """
    W = W_2d.detach().float().cpu()
    m, n = W.shape
    if m < n:
        W = W.t()
        m, n = n, m  # ensure m >= n

    s2 = torch.linalg.svd(W, full_matrices=False).S.pow(2).numpy()
    total = float(s2.sum())

    # Initial σ²: assume all entries are noise
    sigma2 = total / (m * n)

    alpha = n / m  # aspect ratio (≤ 1)

    for _ in range(100):
        # Marchenko-Pastur upper edge for squared singular values
        tau = m * sigma2 * (1.0 + alpha ** 0.5) ** 2
        rank = int((s2 > tau).sum())
        if rank == 0:
            break
        # Re-estimate σ² from residual (noise) degrees of freedom
        noise_dof = m * n - rank * (m + n - rank)
        if noise_dof <= 0:
            break
        sigma2_new = max((total - float(s2[:rank].sum())) / noise_dof, 1e-12)
        if abs(sigma2_new - sigma2) / (sigma2 + 1e-12) < 1e-7:
            sigma2 = sigma2_new
            break
        sigma2 = sigma2_new

    tau = m * sigma2 * (1.0 + alpha ** 0.5) ** 2
    return max(int((s2 > tau).sum()), 1)


def _tucker_vbmf_ranks(weight):
    """
    Estimate Tucker (rank_out, rank_in) via per-mode EVBMF on the weight tensor itself.
    Used as fallback when no teacher model is provided.
    """
    out_ch, in_ch = weight.shape[0], weight.shape[1]
    rank_out = _evbmf_rank(weight.reshape(out_ch, -1))
    rank_out = max(rank_out, min(4, out_ch))
    rank_out = min(rank_out, out_ch)

    W1 = weight.permute(1, 0, 2, 3).reshape(in_ch, -1)
    rank_in = _evbmf_rank(W1)
    rank_in = max(rank_in, min(4, in_ch))
    rank_in = min(rank_in, in_ch)
    return rank_out, rank_in


def _cp_vbmf_rank(weight):
    """
    Estimate CP rank via EVBMF on the mode-0 unfolding of the weight tensor itself.
    Used as fallback when no teacher model is provided.
    """
    out_ch, in_ch, k_h, k_w = weight.shape
    rank = _evbmf_rank(weight.reshape(out_ch, -1))
    dense_params = out_ch * in_ch * k_h * k_w
    max_rank = max(dense_params // max(in_ch + k_h + k_w + out_ch, 1), 1)
    cp_floor = min(4, min(out_ch, in_ch))
    return max(min(rank, max_rank), cp_floor)


def _collect_decomposable_convs(model):
    """Flat list of Conv2d layers eligible for CP/Tucker decomposition (no 1×1, no groups)."""
    result = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            k = module.kernel_size
            if k not in [(1, 1), 1] and module.groups == 1:
                result.append(module)
    return result


def _vbmf_rank_map_from_teacher(student_model, teacher_model, method):
    """
    Build {id(student_conv): rank} using EVBMF run on the *teacher's* trained weights.

    Student and teacher typically differ in depth/width.  Student layer i is matched
    to teacher layer j via positional interpolation:
        j = round(i * (T-1) / (S-1))
    The EVBMF rank of the teacher layer is expressed as a *fraction* of the teacher's
    channel count, and that fraction is scaled to the student's channel dimensions.
    This transfers "how much of its capacity does this teacher layer actually use" to
    the student's differently-sized layer.
    """
    s_convs = _collect_decomposable_convs(student_model)
    t_convs = _collect_decomposable_convs(teacher_model)

    if not s_convs or not t_convs:
        return {}

    n_s, n_t = len(s_convs), len(t_convs)
    rank_map = {}

    for i, s_conv in enumerate(s_convs):
        j = round(i * (n_t - 1) / (n_s - 1)) if n_s > 1 else (n_t - 1) // 2
        t_conv = t_convs[j]
        out_s, in_s, k_h, k_w = s_conv.weight.shape
        out_t, in_t = t_conv.weight.shape[0], t_conv.weight.shape[1]

        # Always work on a CPU float copy — teacher may be on CUDA
        t_w = t_conv.weight.detach().float().cpu()

        if method == 'cp':
            t_rank = _evbmf_rank(t_w.reshape(out_t, -1))
            frac = t_rank / max(out_t, 1)
            s_rank = round(frac * max(out_s, in_s))
            dense = out_s * in_s * k_h * k_w
            max_rank = max(dense // max(in_s + k_h + k_w + out_s, 1), 1)
            cp_floor = min(4, min(out_s, in_s))
            rank_map[id(s_conv)] = max(min(s_rank, max_rank), cp_floor)

        elif method == 'tucker':
            # Output mode
            t_ro = _evbmf_rank(t_w.reshape(out_t, -1))
            rank_out = max(round(t_ro / max(out_t, 1) * out_s), min(4, out_s))
            rank_out = min(rank_out, out_s)
            # Input mode (mode-1 unfolding: permute input dim to front)
            t_ri = _evbmf_rank(t_w.permute(1, 0, 2, 3).reshape(in_t, -1))
            rank_in = max(round(t_ri / max(in_t, 1) * in_s), min(4, in_s))
            rank_in = min(rank_in, in_s)
            rank_map[id(s_conv)] = (rank_out, rank_in)

    return rank_map


class CPConv2d(nn.Module):
    """
    Custom Module to replace Conv2d.
    Exposes the individual factor weights for the BSAT loss.
    """
    def __init__(self, layer, rank):
        super().__init__()
        self.device = layer.weight.device
        weight = layer.weight.data
        out_channels, in_channels, kernel_height, kernel_width = weight.shape

        # Random init avoids SVD on mode unfoldings (SVD of [3, out*in*k] with
        # full_matrices=True allocates V[98304,98304] ≈ 38 GB for large layers).
        # tol=0 skips per-iteration kruskal_to_tensor reconstruction.
        # Quality doesn't matter: student weights are random Kaiming init and
        # will be fine-tuned from scratch.
        with torch.no_grad():
            cp_tensor = parafac(weight, rank=rank, init='random', n_iter_max=10, tol=0)
        cp_weights, (f_out, f_in, f_h, f_w) = cp_tensor

        # Absorb the CP scaling weights (lambda) into the output factor
        f_out = f_out * cp_weights.unsqueeze(0)

        dtype = weight.dtype

        # Layer 1: Pointwise convolution (in_channels -> rank)
        self.pointwise_in = nn.Conv2d(in_channels, rank, kernel_size=1, bias=False).to(self.device)
        self.pointwise_in.weight.data = f_in.t().unsqueeze(2).unsqueeze(3).to(dtype=dtype).contiguous()

        # Layer 2: Depthwise convolution over height
        self.depthwise_h = nn.Conv2d(rank, rank, kernel_size=(kernel_height, 1),
                                     stride=(layer.stride[0], 1), padding=(layer.padding[0], 0),
                                     groups=rank, bias=False).to(self.device)
        self.depthwise_h.weight.data = f_h.t().unsqueeze(1).unsqueeze(3).to(dtype=dtype).contiguous()

        # Layer 3: Depthwise convolution over width
        self.depthwise_w = nn.Conv2d(rank, rank, kernel_size=(1, kernel_width),
                                     stride=(1, layer.stride[1]), padding=(0, layer.padding[1]),
                                     groups=rank, bias=False).to(self.device)
        self.depthwise_w.weight.data = f_w.t().unsqueeze(1).unsqueeze(2).to(dtype=dtype).contiguous()

        # Layer 4: Pointwise convolution (rank -> out_channels)
        self.pointwise_out = nn.Conv2d(rank, out_channels, kernel_size=1,
                                       bias=True if layer.bias is not None else False).to(self.device)
        self.pointwise_out.weight.data = f_out.unsqueeze(2).unsqueeze(3).to(dtype=dtype).contiguous()
        
        if layer.bias is not None:
            self.pointwise_out.bias.data = layer.bias.data

        del cp_tensor, cp_weights, f_out, f_in, f_h, f_w
        gc.collect()

    def forward(self, x):
        x = self.pointwise_in(x)
        x = self.depthwise_h(x)
        x = self.depthwise_w(x)
        x = self.pointwise_out(x)
        return x
    
    def get_factors(self):
        """Returns the factor weights for the BSAT loss function"""
        return [self.pointwise_in.weight, self.depthwise_h.weight, 
                self.depthwise_w.weight, self.pointwise_out.weight]


class TuckerConv2d(nn.Module):
    """
    Custom Module for Tucker Decomposition.
    Exposes the core and factor weights for the Coupled Tensor Loss.
    """
    def __init__(self, layer, ranks):
        super().__init__()
        self.device = layer.weight.device
        weight = layer.weight.data
        out_channels, in_channels, kernel_height, kernel_width = weight.shape
        rank_out, rank_in = ranks

        dtype = weight.dtype

        with torch.no_grad():
            core, factors = tucker(weight, rank=[rank_out, rank_in, kernel_height, kernel_width],
                                   init='random', n_iter_max=10, tol=0)
        f_out, f_in, f_h, f_w = factors

        # Layer 1: Pointwise convolution (in_channels -> rank_in)
        self.pointwise_in = nn.Conv2d(in_channels, rank_in, kernel_size=1, bias=False).to(self.device)
        self.pointwise_in.weight.data = f_in.t().unsqueeze(-1).unsqueeze(-1).to(dtype=dtype).contiguous()

        # Layer 2: Core convolution (rank_in -> rank_out)
        self.core_conv = nn.Conv2d(rank_in, rank_out, kernel_size=(kernel_height, kernel_width),
                                   stride=layer.stride, padding=layer.padding, bias=False).to(self.device)

        # Reconstruct spatial core
        core_spatial = tl.tenalg.mode_dot(core, f_h, mode=2)
        core_spatial = tl.tenalg.mode_dot(core_spatial, f_w, mode=3)
        self.core_conv.weight.data = core_spatial.to(dtype=dtype).contiguous()

        # Layer 3: Pointwise convolution (rank_out -> out_channels)
        self.pointwise_out = nn.Conv2d(rank_out, out_channels, kernel_size=1,
                                       bias=True if layer.bias is not None else False).to(self.device)
        self.pointwise_out.weight.data = f_out.unsqueeze(-1).unsqueeze(-1).to(dtype=dtype).contiguous()
        
        if layer.bias is not None:
            self.pointwise_out.bias.data = layer.bias.data

        del core, factors, f_out, f_in, f_h, f_w, core_spatial
        gc.collect()

    def forward(self, x):
        x = self.pointwise_in(x)
        x = self.core_conv(x)
        x = self.pointwise_out(x)
        return x
    
    def get_factors(self):
        """Returns the core and factor weights for the BSAT loss function"""
        return [self.pointwise_in.weight, self.core_conv.weight, self.pointwise_out.weight]


def decompose_model(model, method='cp', cp_rank_ratio=0.5, tucker_rank_ratio=0.5,
                    use_vbmf=False, teacher_model=None, _rank_map=None, device=None):
    """
    Recursively replaces all nn.Conv2d layers in the model with decomposed equivalents.

    Rank selection modes (controlled by use_vbmf and teacher_model):
      - use_vbmf=False          : global ratio (cp_rank_ratio / tucker_rank_ratio)
      - use_vbmf=True, teacher  : EVBMF on teacher weights → rank fraction → student dims
      - use_vbmf=True, no teacher : EVBMF on student's own weights (fallback, less meaningful
                                    for randomly-initialized students)

    _rank_map is internal — built once at the top-level call, then passed through recursion.
    """
    # Build rank map once at the top-level call
    if use_vbmf and _rank_map is None:
        if teacher_model is not None:
            _rank_map = _vbmf_rank_map_from_teacher(model, teacher_model, method)
            print(f"  [VBMF] ranks derived from teacher weights for {len(_rank_map)} layers")
        else:
            _rank_map = {}  # per-layer self-VBMF as fallback
            print("  [VBMF] no teacher provided — using self-VBMF per layer (less meaningful for random weights)")

    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            decompose_model(module, method, cp_rank_ratio, tucker_rank_ratio,
                            use_vbmf, teacher_model=None, _rank_map=_rank_map, device=device)
        elif isinstance(module, nn.Conv2d):
            # Skip 1×1 and grouped/depthwise (decomposition undefined or counterproductive)
            if module.kernel_size == (1, 1) or module.kernel_size == 1 or module.groups > 1:
                continue

            out_channels, in_channels, k_h, k_w = module.weight.shape
            dense_params = out_channels * in_channels * k_h * k_w

            if method == 'cp':
                if use_vbmf:
                    if _rank_map and id(module) in _rank_map:
                        rank = _rank_map[id(module)]
                        print(f"  {name}: CP rank={rank} (teacher-VBMF)")
                    else:
                        rank = _cp_vbmf_rank(module.weight)
                        print(f"  {name}: CP rank={rank} (self-VBMF fallback)")
                else:
                    max_rank = dense_params // max(in_channels + k_h + k_w + out_channels, 1)
                    rank = max(int(max(out_channels, in_channels) * cp_rank_ratio), 1)
                    rank = min(rank, max(max_rank, 1))
                new_module = CPConv2d(module, rank)
                gc.collect()

            elif method == 'tucker':
                if use_vbmf:
                    if _rank_map and id(module) in _rank_map:
                        rank_out, rank_in = _rank_map[id(module)]
                        print(f"  {name}: Tucker ranks=({rank_out}, {rank_in}) (teacher-VBMF)")
                    else:
                        rank_out, rank_in = _tucker_vbmf_ranks(module.weight)
                        print(f"  {name}: Tucker ranks=({rank_out}, {rank_in}) (self-VBMF fallback)")
                else:
                    # Floor at min(4, channels) so cuDNN never sees degenerate tensors.
                    rank_out = max(int(out_channels * tucker_rank_ratio), min(4, out_channels))
                    rank_in = max(int(in_channels * tucker_rank_ratio), min(4, in_channels))
                new_module = TuckerConv2d(module, [rank_out, rank_in])
                gc.collect()

            else:
                raise ValueError(f"Unknown decomposition method: {method}")

            setattr(model, name, new_module)
            if device is not None:
                new_module.to(device)
                gc.collect()

    return model