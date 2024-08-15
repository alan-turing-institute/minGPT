import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from mingpt.utils import CfgNode as CN


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config: CN):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
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
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.max_batch_size = config.max_batch_size
        self.block_size = config.block_size

        # key, value cache for fast autoregressive generation
        # initialised to None now to avoid allocating memory to cache
        # when it's not used or during training
        # it will be initialsied when requested during inference in forward pass
        self.cache_k = None
        self.cache_v = None

    def forward(self, x: torch.Tensor, use_kv_cache: bool = False, start_pos: int = 0):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and
        # move head forward to be the batch dim
        q, xk, xv = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        xk = xk.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        xv = xv.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # if enabled, use key, value cache to speed up computation during inference
        if use_kv_cache:
            # check if cache is initialised; if not, initialise it to maximum batch and sequence lengths
            if self.cache_k is None:
                self.cache_k = torch.zeros(
                    self.max_batch_size,
                    self.n_head,
                    self.block_size,
                    self.n_embd // self.n_head,
                )  # (max(B), nh, max(T), hs)
            if self.cache_v is None:
                self.cache_v = torch.zeros(
                    self.max_batch_size,
                    self.n_head,
                    self.block_size,
                    self.n_embd // self.n_head,
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
