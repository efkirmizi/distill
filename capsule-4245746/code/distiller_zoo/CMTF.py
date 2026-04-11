import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorly as tl
from tensorly.decomposition import parafac

tl.set_backend('pytorch')

class CoupledTensorLoss(nn.Module):
    """
    Coupled Matrix-Tensor Factorization (CMTF) / Structure-Revealing Data Fusion Loss.
    Treats the feature maps of the Teacher and Student networks as coupled tensors and
    aligns their shared factors (Batch, Height, Width) using permutation-invariant Gram matrices.
    """
    def __init__(self, rank=16, iter_max=5, **kwargs):
        super(CoupledTensorLoss, self).__init__()
        self.rank = rank
        self.iter_max = iter_max

    def forward(self, f_s, f_t):
        loss = 0.0
        for s, t in zip(f_s, f_t):
            batch_size = s.shape[0]
            
            # Align spatial dimensions if they don't match
            if s.shape[2:] != t.shape[2:]:
                s = F.adaptive_avg_pool2d(s, t.shape[2:])
            
            try:
                # factors are: [Batch, Channels, Height, Width]
                # CRITICAL FIX 1: Use init='random' to prevent PyTorch SVD backward graph crashes
                _, factors_s = parafac(s, rank=self.rank, init='random', n_iter_max=self.iter_max, tol=1e-4)
                _, factors_t = parafac(t, rank=self.rank, init='random', n_iter_max=self.iter_max, tol=1e-4)
                
                # Extract factors (ignoring Channel dimension at index 1)
                s_b, s_h, s_w = factors_s[0], factors_s[2], factors_s[3]
                t_b, t_h, t_w = factors_t[0], factors_t[2], factors_t[3]
                
                # CRITICAL FIX 2: Compute Gram Matrices (Factor @ Factor.T) 
                # This makes the loss invariant to the arbitrary column permutation of CP decomposition.
                gram_s_b = torch.mm(s_b, s_b.t())
                gram_t_b = torch.mm(t_b, t_b.t())
                
                gram_s_h = torch.mm(s_h, s_h.t())
                gram_t_h = torch.mm(t_h, t_h.t())
                
                gram_s_w = torch.mm(s_w, s_w.t())
                gram_t_w = torch.mm(t_w, t_w.t())
                
                # Normalize the Gram matrices using Frobenius norm to prevent gradient explosions
                loss += F.mse_loss(gram_s_b / torch.norm(gram_s_b, p='fro'), 
                                   gram_t_b / torch.norm(gram_t_b, p='fro'))
                
                loss += F.mse_loss(gram_s_h / torch.norm(gram_s_h, p='fro'), 
                                   gram_t_h / torch.norm(gram_t_h, p='fro'))
                
                loss += F.mse_loss(gram_s_w / torch.norm(gram_s_w, p='fro'), 
                                   gram_t_w / torch.norm(gram_t_w, p='fro'))
                
            except Exception as e:
                # Fallback to standard spatial attention transfer loss if Parafac fails to converge
                s_attention = F.normalize(s.pow(2).mean(1).view(batch_size, -1), dim=1)
                t_attention = F.normalize(t.pow(2).mean(1).view(batch_size, -1), dim=1)
                loss += F.mse_loss(s_attention, t_attention)
                
        return loss