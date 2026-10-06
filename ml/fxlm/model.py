"""fxlm.model: GPTConfig + GPT — pre-norm transformer (fp16 dostu).

Blok: RMSNorm -> qkv Linear(D->3D, bias yok) -> flash_attention_qkv
( (B,T,3,H,Dh) gorunumu, permute yok ) -> proj Linear -> residual;
RMSNorm -> fc(D->4D) -> GELU(tanh) -> fc2 -> residual.
Final RMSNorm; LM head = tied embedding (ya da ayri Linear).
attn_impl: "flash"|"sdpa" ; norm_impl: "fused"|"torch" ; ce_impl:
"fused"|"torch" — ablation/parity icin duz PyTorch yollari.
"""
import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as Fn

from .attention import flash_attention_qkv
from .norm import rms_norm
from .cross_entropy import linear_cross_entropy


@dataclass
class GPTConfig:
    vocab_size: int
    seq_len: int
    n_layer: int
    n_head: int
    d_model: int
    mlp_ratio: int = 4
    tie_embeddings: bool = True
    attn_impl: str = "flash"      # "flash" | "sdpa"
    norm_impl: str = "torch"      # "fused" | "torch"
    ce_impl: str = "fused"        # "fused" | "torch"
    eps: float = 1e-6


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        D, H = cfg.d_model, cfg.n_head
        self.n_head = H
        self.head_dim = D // H
        self.attn_impl = cfg.attn_impl
        self.norm_impl = cfg.norm_impl
        self.eps = cfg.eps
        self.ln1_w = nn.Parameter(torch.ones(D))
        self.ln2_w = nn.Parameter(torch.ones(D))
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.proj = nn.Linear(D, D, bias=False)
        self.fc = nn.Linear(D, cfg.mlp_ratio * D, bias=False)
        self.fc2 = nn.Linear(cfg.mlp_ratio * D, D, bias=False)

    def _norm(self, x, w):
        if self.norm_impl == "fused":
            return rms_norm(x, w, self.eps)
        return Fn.rms_norm(x, (x.shape[-1],), w, self.eps)

    def forward(self, x):
        B, T, D = x.shape
        h = self._norm(x, self.ln1_w)
        qkv = self.qkv(h).view(B, T, 3, self.n_head, self.head_dim)
        if self.attn_impl == "flash":
            y = flash_attention_qkv(qkv, causal=True)
        else:
            q, k, v = (qkv[:, :, i].transpose(1, 2)
                       for i in range(3))             # (B,H,T,Dh)
            y = Fn.scaled_dot_product_attention(q, k, v, is_causal=True)
            y = y.transpose(1, 2).reshape(B, T, D)
        x = x + self.proj(y.reshape(B, T, D))
        h = self._norm(x, self.ln2_w)
        x = x + self.fc2(Fn.gelu(self.fc(h), approximate="tanh"))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        if cfg.d_model % cfg.n_head:
            raise ValueError("d_model, n_head'in kati olmali")
        self.cfg = cfg
        D = cfg.d_model
        self.tok_emb = nn.Embedding(cfg.vocab_size, D)
        self.pos_emb = nn.Parameter(torch.zeros(cfg.seq_len, D))
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.lnf_w = nn.Parameter(torch.ones(D))
        if not cfg.tie_embeddings:
            self.head = nn.Linear(D, cfg.vocab_size, bias=False)
        self.apply(self._init)
        # proj/fc2 kucuk init (residual derinligi olcegi)
        s = 0.02 / math.sqrt(2 * cfg.n_layer)
        for b in self.blocks:
            nn.init.normal_(b.proj.weight, std=s)
            nn.init.normal_(b.fc2.weight, std=s)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def _norm_f(self, x):
        if self.cfg.norm_impl == "fused":
            return rms_norm(x, self.lnf_w, self.cfg.eps)
        return Fn.rms_norm(x, (x.shape[-1],), self.lnf_w, self.cfg.eps)

    def _head_w(self):
        return self.tok_emb.weight if self.cfg.tie_embeddings \
            else self.head.weight

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, grad_scale=1.0):
        B, T = idx.shape
        if T > self.cfg.seq_len:
            raise ValueError(f"T={T} > seq_len={self.cfg.seq_len}")
        x = self.tok_emb(idx) + self.pos_emb[:T]
        for blk in self.blocks:
            x = blk(x)
        x = self._norm_f(x)
        if targets is None:
            return Fn.linear(x, self._head_w())          # (B,T,V) fp16
        if self.cfg.ce_impl == "fused":
            return linear_cross_entropy(
                x.reshape(-1, x.shape[-1]), self._head_w(),
                targets.reshape(-1), grad_scale=grad_scale)
        logits = Fn.linear(x, self._head_w())
        return Fn.cross_entropy(
            logits.float().view(-1, logits.shape[-1]), targets.reshape(-1))
