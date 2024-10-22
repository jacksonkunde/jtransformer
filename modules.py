# modules.py
from config import TransformerConfig

import math

import torch.nn as nn
import torch as th
from jaxtyping import Float, Int
import einops as ein


class LayerNorm(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg: TransformerConfig = cfg
        self.w = nn.Parameter(th.ones(cfg.d_model))
        self.b = nn.Parameter(th.zeros(cfg.d_model))

    def forward(
        self, res: Float[th.Tensor, "batch position d_model"]
    ) -> Float[th.Tensor, "batch position d_model"]:
        mean = res.mean(dim=-1, keepdim=True)
        var = res.var(dim=-1, keepdim=True).sqrt()
        res = res - mean / th.sqrt(var + self.cfg.layer_norm_eps)

        return self.w * res + self.b


class Embed(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        # t.empty reserves the memory without initalizing like t.ones or t.zeros
        self.embed = nn.Parameter(th.empty((cfg.d_vocab, cfg.d_model)))
        # init with values from normal dist
        nn.init.normal_(self.embed, std=self.cfg.init_range)

    def forward(
        self, tokens: Int[th.Tensor, "batch position"]
    ) -> Float[th.Tensor, "batch position d_model"]:
        # Index the embeddings
        return self.embed[tokens]


class GPT2PositionalEmbed(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        # Similar lookup table to token embeddings
        self.pos_embed = nn.Parameter(th.empty((cfg.n_ctx, cfg.d_model)))
        nn.init.normal_(self.pos_embed, std=self.cfg.init_range)

    def forward(
        self, tokens: Int[th.Tensor, "batch position"]
    ) -> Float[th.Tensor, "batch position d_model"]:
        batch, seq_len = tokens.shape
        return ein.repeat(
            self.pos_embed[:seq_len],
            "position d_model -> batch position d_model",
            batch=batch,
        )


# I'll work on this another time.
class RotaryPositionalEmbed(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.pos_embed = nn.Parameter(th.empty())
        raise NotImplementedError

    def forward(self):
        # TODO
        raise NotImplementedError


class Attention(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        # Masking
        self.MASK: th.tensor
        self.register_buffer("MASK", th.tensor(float("-inf"), dtype=th.float32))

        # Attn weights
        # key, query, value projections for all heads, but in a batch
        self.W_attn = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=True)
        nn.init.normal_(self.W_attn.weight, std=cfg.init_range)

        self.W_O = nn.Linear(cfg.d_model, cfg.d_model, bias=True)

    def causal_mask(
        self, attn_scores: Float[th.Tensor, "batch n_heads query_pos key_pos"]
    ) -> Float[th.Tensor, "batch n_heads query_pos key_pos"]:
        batch, n_heads, query_pos, key_pos = attn_scores.shape
        o = th.ones((query_pos, key_pos))  # Create matrix in attn matrix shape
        mask = th.triu(
            input=o, diagonal=1
        ).bool()  # Upper triangular matrix 1 off diagonal
        attn_scores.masked_fill(mask=mask, value=self.MASK)
        return attn_scores

    def forward(self, res: Float[th.Tensor, "batch position d_model"]):
        batch, seq_len, d_model = res.shape

        qkv = self.W_attn(res)
        q, k, v = qkv.split(self.cfg.d_model, dim=-1)

        # Seperate attn heads
        q = ein.rearrange(
            q,
            "batch position (n_heads d_head) -> batch n_heads position d_head",
            n_heads=self.cfg.n_heads,
        )
        k = ein.rearrange(
            k,
            "batch position (n_heads d_head) -> batch n_heads position d_head",
            n_heads=self.cfg.n_heads,
        )
        v = ein.rearrange(
            v,
            "batch position (n_heads d_head) -> batch n_heads position d_head",
            n_heads=self.cfg.n_heads,
        )
        # A = QK^T / sqrt(d_head) (transpose last two indicies of K and matrix multiply)
        attn_scores = (
            th.einsum(
                "b h q d, b h k d -> b h q k",
                q,  # shape: (batch, n_heads, pos_q, d_head)
                k,  # shape: (batch, n_heads, pos_k, d_head)
            )
            / (d_model // self.cfg.n_heads) ** 0.5
        )
        attn_scores = self.causal_mask(attn_scores)
        attn_scores = attn_scores.softmax(-1)

        # Multiply with values
        print(attn_scores.shape, v.shape)
        inter = th.einsum(
            "b h q k, b h k d -> b h q d",  # Use single-character subscripts
            attn_scores,  # Shape: (batch, n_heads, pos_q, pos_k)
            v,  # Shape: (batch, n_heads, pos_k, d_head)
        )

        # Rearrange back to original shape: [batch, n_heads, pos_q, d_head] -> [batch, pos_q, d_model]
        inter = ein.rearrange(
            inter, "batch n_heads pos_q d_head -> batch pos_q (n_heads d_head)"
        )

        return self.W_O(inter)


class MLP(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.linear_in = nn.Linear(cfg.d_model, 4 * cfg.d_model)
        self.linear_out = nn.Linear(cfg.d_model * 4, cfg.d_model)
        self.gelu = nn.GELU()

    def forward(
        self, res: Float[th.Tensor, "batch position d_model"]
    ) -> Float[th.Tensor, "batch position d_model"]:
        res = self.linear_in(res)
        res = self.gelu(res)
        res = self.linear_out(res)
        res = self.gelu(res)
        return res


class Unembed(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Linear(cfg.d_model, cfg.d_vocab, bias=True)

    def forward(self, res: Float[th.Tensor, "batch position d_model"]):
        return self.W_U(res)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.ln1 = LayerNorm(cfg)
        self.ln2 = LayerNorm(cfg)
        self.attn = Attention(cfg)
        self.mlp = MLP(cfg)

    def forward(self, res):
        inter = self.attn(self.ln1(res)) + res
        out = self.mlp(self.ln2(inter)) + inter
        return out


class Transformer(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = GPT2PositionalEmbed(cfg)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.ln_last = LayerNorm(cfg)
        self.unembed = Unembed(cfg)

    def forward(self, tokens):
        res = self.embed(tokens) + self.pos_embed(tokens)
        for block in self.transformer_blocks:
            res = block(res)
        logits = self.unembed(self.ln_last(res))
        return logits
