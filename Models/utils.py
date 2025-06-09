# -*- coding: utf-8 -*-
import torch.nn as nn
import torch

class MultiSequential(nn.Sequential):
    """Multi-input multi-output torch.nn.Sequential."""
    def forward(self, *args):
        """Repeat."""
        for m in self:
            args = m(*args)
        return args


def repeat(N, fn):
    """Repeat module N times.
    :param int N: repeat time
    :param function fn: function to generate module
    :return: repeated modules
    :rtype: MultiSequential
    """
    return MultiSequential(*[fn() for _ in range(N)])

#def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
#    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
#    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
#    freqs = torch.outer(t, freqs)
#    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
#    return freqs_cis
#
#
#def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
#    ndim = x.ndim
#    assert 0 <= 1 < ndim
#    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
#    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
#    return freqs_cis.view(*shape)
#
#
#def apply_rotary_emb(
#    xq: torch.Tensor,
#    xk: torch.Tensor,
#    freqs_cis: torch.Tensor,
#) -> Tuple[torch.Tensor, torch.Tensor]:
#    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
#    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
#    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
#    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
#    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
#    return xq_out.type_as(xq), xk_out.type_as(xk)
