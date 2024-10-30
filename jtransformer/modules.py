# modules.py
from jtransformer.config import TransformerConfig

import os
import json
from typing import List

import torch as th
import torch.nn as nn
from torch.nn.functional import softmax
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
        var = res.var(dim=-1, keepdim=True, unbiased=False).sqrt()
        res = (res - mean) / (var + self.cfg.layer_norm_eps)

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
        self, input_ids: Int[th.Tensor, "batch position"]
    ) -> Float[th.Tensor, "batch position d_model"]:
        # Index the embeddings
        return self.embed[input_ids]


class GPT2PositionalEmbed(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        # Similar lookup table to token embeddings
        self.pos_embed = nn.Parameter(th.empty((cfg.n_ctx, cfg.d_model)))
        nn.init.normal_(self.pos_embed, std=self.cfg.init_range)

    def forward(
        self, input_ids: Int[th.Tensor, "batch position"]
    ) -> Float[th.Tensor, "batch position d_model"]:
        batch, seq_len = input_ids.shape
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
        self.register_buffer("MASK", th.tensor(float("-inf"), dtype=th.float32))

        # Attn weights
        # key, query, value projections for all heads, but in a batch
        self.W_attn = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=True)
        nn.init.normal_(self.W_attn.weight, std=cfg.init_range)

        self.W_O = nn.Linear(cfg.d_model, cfg.d_model, bias=True)
        nn.init.normal_(self.W_O.weight, std=cfg.init_range)

        self.d_head = self.cfg.d_model // self.cfg.n_heads

    def causal_mask(
        self, attn_scores: Float[th.Tensor, "batch n_heads query_pos key_pos"]
    ) -> Float[th.Tensor, "batch n_heads query_pos key_pos"]:
        batch, n_heads, query_pos, key_pos = attn_scores.shape
        o = th.ones((query_pos, key_pos))  # Create matrix in attn matrix shape
        mask = (
            th.triu(input=o, diagonal=1).bool().to(attn_scores.device)
        )  # Upper triangular matrix 1 off diagonal
        attn_scores = attn_scores.masked_fill(
            mask, value=self.MASK.to(attn_scores.device)
        )
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
            / (self.d_head) ** 0.5
        )
        attn_scores = self.causal_mask(attn_scores)
        attn_scores = attn_scores.softmax(dim=-1)

        # Multiply with values
        inter = th.einsum(
            "b h q k, b h k d -> b h q d",
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
        self.linear_in = nn.Linear(cfg.d_model, cfg.d_mlp)
        self.linear_out = nn.Linear(cfg.d_mlp, cfg.d_model)
        self.gelu = nn.GELU()

    def forward(
        self, res: Float[th.Tensor, "batch position d_model"]
    ) -> Float[th.Tensor, "batch position d_model"]:
        res = self.linear_in(res)
        res = self.gelu(res)
        res = self.linear_out(res)
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


class Jtransformer(nn.Module):
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

    def forward(self, input_ids) -> th.Tensor:
        res = self.embed(input_ids) + self.pos_embed(input_ids)
        for block in self.transformer_blocks:
            res = block(res)
        logits = self.unembed(self.ln_last(res))
        return logits

    def generate(
        self,
        input_ids: th.Tensor | List[int],
        eos_token_id=int,
        max_length: int = 100,
        max_new_tokens: int | None = None,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_p: float | None = None,
        top_k: int | None = None,
    ):
        if isinstance(input_ids, List):
            input_ids = th.tensor(input_ids)

        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)

        n_input_tokens = input_ids.size(-1)
        if max_new_tokens is None:
            max_new_tokens = max_length - n_input_tokens

        batch_size = input_ids.size(0)
        is_finished = th.zeros(batch_size, dtype=th.bool, device=input_ids.device)

        for i in range(max_new_tokens):
            logits = self.forward(input_ids=input_ids)

            # Apply temp
            logits = logits[:, -1, :] / temperature

            # Mask logits for finished sequences
            logits = logits.masked_fill(is_finished, float("-inf"))

            # apply top-k filtering if set
            if top_k is not None:
                top_k_values, _ = th.topk(logits, top_k, dim=-1)
                print(f"topk val{top_k_values.shape}")
                min_top_k = top_k_values[:, -1]
                print(f" min top k{min_top_k.shape}")
                logits = th.where(
                    logits < min_top_k, th.full_like(logits, float("-inf")), logits
                )
                print(logits.shape)

            # Sample or use argmax based on the `sample` flag
            if do_sample:
                print(logits.shape)
                probabilities = softmax(logits, dim=-1)
                print(probabilities.shape)
                new_tokens = th.multinomial(probabilities, num_samples=1)
            else:
                new_tokens = logits.argmax(dim=-1, keepdim=True)

            print(logits)
            print(new_tokens)
            print(input_ids)

            input_ids = th.cat((input_ids, new_tokens), dim=-1)

            # Update the input_ids only for unfinished sequences
            new_tokens = new_tokens.masked_fill(is_finished, eos_token_id)

            # Check if any new EOS tokens were generated and update the mask using bitwise OR
            is_finished = is_finished | (new_tokens == eos_token_id)

            # Break if all sequences are finished
            if is_finished.all():
                break

        return input_ids

    def save(self, save_dir: str) -> None:
        """
        Save the model's state_dict and configuration to the given directory.
        """
        os.makedirs(save_dir, exist_ok=True)

        # Save the model's state dict
        model_path = os.path.join(save_dir, "model.pth")
        th.save(self.state_dict(), model_path)

        # Save the configuration as JSON
        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.cfg.__dict__, f, indent=4)

        print(f"Model and config saved to {save_dir}")

    @classmethod
    def load(cls, load_dir: str) -> "Jtransformer":
        """
        Load the model's state_dict and configuration from the given directory.
        """
        # Load the configuration from JSON
        config_path = os.path.join(load_dir, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)

        # Recreate the config object
        cfg = TransformerConfig(**cfg_dict)

        # Initialize the model with the loaded config
        model = cls(cfg)

        # Load the state dict
        model_path = os.path.join(load_dir, "model.pth")
        model.load_state_dict(th.load(model_path, map_location="cpu"))

        print(f"Model loaded from {load_dir}")
        return model
