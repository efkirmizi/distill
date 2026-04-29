import torch
import torch.nn as nn
import tensorly as tl
from tensorly.decomposition import parafac, tucker

tl.set_backend('pytorch')

class CPConv2d(nn.Module):
    """
    Custom Module to replace Conv2d.
    Exposes the individual factor weights for the Coupled Tensor Loss (CMTF).
    """
    def __init__(self, layer, rank):
        super().__init__()
        self.device = layer.weight.device
        weight = layer.weight.data
        out_channels, in_channels, kernel_height, kernel_width = weight.shape

        # Perform CP decomposition
        cp_tensor = parafac(weight, rank=rank, init='random', tol=10e-5)
        cp_weights, (f_out, f_in, f_h, f_w) = cp_tensor

        # Absorb the CP scaling weights (lambda) into the output factor
        f_out = f_out * cp_weights.unsqueeze(0)

        # Layer 1: Pointwise convolution (in_channels -> rank)
        self.pointwise_in = nn.Conv2d(in_channels, rank, kernel_size=1, bias=False).to(self.device)
        self.pointwise_in.weight.data = f_in.t().unsqueeze(2).unsqueeze(3)

        # Layer 2: Depthwise convolution over height
        self.depthwise_h = nn.Conv2d(rank, rank, kernel_size=(kernel_height, 1), 
                                     stride=(layer.stride[0], 1), padding=(layer.padding[0], 0), 
                                     groups=rank, bias=False).to(self.device)
        self.depthwise_h.weight.data = f_h.t().unsqueeze(1).unsqueeze(3)

        # Layer 3: Depthwise convolution over width
        self.depthwise_w = nn.Conv2d(rank, rank, kernel_size=(1, kernel_width), 
                                     stride=(1, layer.stride[1]), padding=(0, layer.padding[1]), 
                                     groups=rank, bias=False).to(self.device)
        self.depthwise_w.weight.data = f_w.t().unsqueeze(1).unsqueeze(2)

        # Layer 4: Pointwise convolution (rank -> out_channels)
        self.pointwise_out = nn.Conv2d(rank, out_channels, kernel_size=1, 
                                       bias=True if layer.bias is not None else False).to(self.device)
        self.pointwise_out.weight.data = f_out.unsqueeze(2).unsqueeze(3)
        
        if layer.bias is not None:
            self.pointwise_out.bias.data = layer.bias.data

    def forward(self, x):
        x = self.pointwise_in(x)
        x = self.depthwise_h(x)
        x = self.depthwise_w(x)
        x = self.pointwise_out(x)
        return x
    
    def get_factors(self):
        """Returns the factor weights for the CMTF loss function"""
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

        # Perform Tucker decomposition
        core, factors = tucker(weight, rank=[rank_out, rank_in, kernel_height, kernel_width], init='random', tol=10e-5)
        f_out, f_in, f_h, f_w = factors

        # Layer 1: Pointwise convolution (in_channels -> rank_in)
        self.pointwise_in = nn.Conv2d(in_channels, rank_in, kernel_size=1, bias=False).to(self.device)
        self.pointwise_in.weight.data = f_in.t().unsqueeze(-1).unsqueeze(-1)

        # Layer 2: Core convolution (rank_in -> rank_out)
        self.core_conv = nn.Conv2d(rank_in, rank_out, kernel_size=(kernel_height, kernel_width),
                                   stride=layer.stride, padding=layer.padding, bias=False).to(self.device)
        
        # Reconstruct spatial core
        core_spatial = tl.tenalg.mode_dot(core, f_h, mode=2) # Use mode_dot sequentially for safety across tensorly versions
        core_spatial = tl.tenalg.mode_dot(core_spatial, f_w, mode=3)
        self.core_conv.weight.data = core_spatial

        # Layer 3: Pointwise convolution (rank_out -> out_channels)
        self.pointwise_out = nn.Conv2d(rank_out, out_channels, kernel_size=1, 
                                       bias=True if layer.bias is not None else False).to(self.device)
        self.pointwise_out.weight.data = f_out.unsqueeze(-1).unsqueeze(-1)
        
        if layer.bias is not None:
            self.pointwise_out.bias.data = layer.bias.data

    def forward(self, x):
        x = self.pointwise_in(x)
        x = self.core_conv(x)
        x = self.pointwise_out(x)
        return x
    
    def get_factors(self):
        """Returns the core and factor weights for the CMTF loss function"""
        return [self.pointwise_in.weight, self.core_conv.weight, self.pointwise_out.weight]


def decompose_model(model, method='cp', cp_rank_ratio=0.5, tucker_rank_ratio=0.5):
    """
    Recursively replaces all nn.Conv2d layers in the model with custom Decomposed classes.
    """
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            decompose_model(module, method, cp_rank_ratio, tucker_rank_ratio)
        elif isinstance(module, nn.Conv2d):
            # Skip 1x1 convolutions (decomposing increases param count) and
            # grouped/depthwise convolutions (CP/Tucker factorization is undefined for groups>1)
            if module.kernel_size == (1, 1) or module.kernel_size == 1 or module.groups > 1:
                continue
                
            out_channels, in_channels, k_h, k_w = module.weight.shape
            
            # Mathematical Ceiling: Ensure compression doesn't bloat the model
            dense_params = out_channels * in_channels * k_h * k_w
            
            if method == 'cp':
                # Rank calculation with a safety ceiling
                max_rank = dense_params // (in_channels + k_h + k_w + out_channels)
                rank = max(int(max(out_channels, in_channels) * cp_rank_ratio), 1)
                rank = min(rank, max(max_rank, 1)) # Enforce ceiling
                
                new_module = CPConv2d(module, rank)
                
            elif method == 'tucker':
                rank_out = max(int(out_channels * tucker_rank_ratio), 1)
                rank_in = max(int(in_channels * tucker_rank_ratio), 1)
                new_module = TuckerConv2d(module, [rank_out, rank_in])
                
            else:
                raise ValueError(f"Unknown decomposition method: {method}")
            
            setattr(model, name, new_module)
            
    return model