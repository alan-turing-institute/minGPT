import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from mingpt.utils import CfgNode as CN


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat the key, value tensors n_rep times along the heads attention.
    Used to repeat the key, value tensors to get the same number
    of heads as the query tensor.
    """
    B, n_kv_heads, T, hs = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(B, n_kv_heads, n_rep, T, hs)
        .reshape(B, n_kv_heads * n_rep, T, hs)
    )


class CausalSelfAttention(nn.Module):
    """
    A grouped query masked self-attention layer with a projection at the end.
    Also implemented with key, value cache option to speed up autoregressive
    generation at inference time.
    """

    def __init__(self, config: CN):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_kv_heads = (
            config.n_head if config.n_kv_heads is None else config.n_kv_heads
        )
        assert config.n_head % self.n_kv_heads == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_size = config.n_embd // config.n_head
        self.n_rep = (
            self.n_head // self.n_kv_heads
        )  # number of times to repeat key, value tensors

        # key, query, value projections for all heads, but in a batch
        # in MHA case where n_head = n_kv_heads, the output is just 3 * n_embd
        self.c_attn = nn.Linear(
            config.n_embd, config.n_embd + 2 * self.n_kv_heads * self.head_size
        )
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )
        self.max_batch_size = config.max_batch_size
        self.block_size = config.block_size

        # key, value cache for fast autoregressive generation
        # initialised to None now to avoid allocating memory to cache
        # when it's not used or during training
        # it will be initialsied when requested during inference in forward pass
        self.cache_k = None
        self.cache_v = None

    def forward(self, x: torch.Tensor, use_kv_cache: bool = False, start_pos: int = 0):
        """
        If use_kv_cache is True, then the key, value computed in a forward pass
        will be cached and can be used in the next forward pass to speed up computation.

        If use_kv_cache is True, then x will often be a tensor with shape (B, 1, C),
        since we only need to input the last token in the sequence to generate the next one,
        and the key, value cache will contain the keys and values for the entire sequence.
        In this case, you will need to provide the start_pos argument to indicate the position
        of the last token in the sequence - start_pos essentially indicates from where should we
        store the computed keys and values in the cache.
        If the tensor x has shape (B, T, C), then start_pos is often 0 as this would typically
        be for "prefilling" the cache up to the first T tokens in the sequence.

        If use_kv_cache is False, then x will often be a tensor with shape (B, T, C),
        and we compute the key, value for the entire sequence in a forward pass.
        No key, values are cached or retrieved in this case.
        """
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and
        # move head forward to be the batch dim
        if self.n_head == self.n_kv_heads:
            q, xk, xv = self.c_attn(x).split(self.n_embd, dim=2)
        else:
            q, xk, xv = self.c_attn(x).split(
                [
                    self.n_embd,
                    self.n_kv_heads * self.head_size,
                    self.n_kv_heads * self.head_size,
                ],
                dim=2,
            )

        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, nh, T, hs)
        xk = xk.view(B, T, self.n_kv_heads, self.head_size).transpose(
            1, 2
        )  # (B, n_kv_h, T, hs)
        xv = xv.view(B, T, self.n_kv_heads, self.head_size).transpose(
            1, 2
        )  # (B, n_kv_h, T, hs)

        # if enabled, use key, value cache to speed up computation during inference
        if use_kv_cache:
            # check if cache is initialised; if not, initialise it to maximum batch and sequence lengths
            if self.cache_k is None:
                self.cache_k = torch.zeros(
                    self.max_batch_size,
                    self.n_kv_heads,
                    self.block_size,
                    self.head_size,
                )  # (max(B), nh, max(T), hs)
            if self.cache_v is None:
                self.cache_v = torch.zeros(
                    self.max_batch_size,
                    self.n_kv_heads,
                    self.block_size,
                    self.head_size,
                )  # (max(B), nh, max(T), hs)

            # make sure cache is on correct device
            self.cache_k = self.cache_k.to(x)
            self.cache_v = self.cache_v.to(x)

            # store the computed keys and values in cache
            self.cache_k[:B, :, start_pos : start_pos + T] = xk
            self.cache_v[:B, :, start_pos : start_pos + T] = xv

            # retrieve the cached keys and values
            k = self.cache_k[:B, :, : start_pos + T]
            v = self.cache_v[:B, :, : start_pos + T]
        else:
            k, v = xk, xv

        # repeat key and value heads if n_kv_heads < n_heads
        k = repeat_kv(k, self.n_rep)  # (B, nh, T, hs)
        v = repeat_kv(v, self.n_rep)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
