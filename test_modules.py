import torch as th
import pytest
from modules import (
    LayerNorm,
    Embed,
    GPT2PositionalEmbed,
    RotaryPositionalEmbed,
    Attention,
    MLP,
    Unembed,
    TransformerBlock,
    Transformer,
)
from utils import load_gpt2_weights_into_transformer
from config import TransformerConfig


@pytest.fixture
def config():
    """Fixture to provide a default TransformerConfig."""
    return TransformerConfig()


def test_layer_norm(config):
    ln = LayerNorm(config)
    x = th.randn(
        2, 10, config.d_model
    )  # Random tensor with batch size 2, sequence length 10
    out = ln(x)
    assert out.shape == (2, 10, config.d_model)


def test_embed(config):
    embed = Embed(config)
    tokens = th.randint(
        0, config.d_vocab, (2, 10)
    )  # Random token ids with batch size 2, sequence length 10
    out = embed(tokens)
    assert out.shape == (2, 10, config.d_model)


def test_gpt2_positional_embed(config):
    pos_embed = GPT2PositionalEmbed(config)
    tokens = th.randint(0, config.d_vocab, (2, 10))  # Same shape as input tokens
    out = pos_embed(tokens)
    assert out.shape == (2, 10, config.d_model)


def test_rotary_positional_embed_not_implemented(config):
    """Ensure RotaryPositionalEmbed raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        rotary_embed = RotaryPositionalEmbed(config)
        rotary_embed(th.zeros(2, 10, config.d_model))  # Example input


def test_attention(config):
    attn = Attention(config)
    x = th.randn(2, 10, config.d_model)  # Batch size 2, sequence length 10
    out = attn(x)
    assert out.shape == (2, 10, config.d_model)


def test_mlp(config):
    mlp = MLP(config)
    x = th.randn(2, 10, config.d_model)
    out = mlp(x)
    assert out.shape == (2, 10, config.d_model)


def test_unembed(config):
    unembed = Unembed(config)
    x = th.randn(2, 10, config.d_model)  # Batch size 2, sequence length 10
    out = unembed(x)
    assert out.shape == (2, 10, config.d_vocab)


def test_transformer_block(config):
    block = TransformerBlock(config)
    x = th.randn(2, 10, config.d_model)
    out = block(x)
    assert out.shape == (2, 10, config.d_model)


def test_transformer(config):
    transformer = Transformer(config)
    # load_gpt2_weights_into_transformer(transformer)  # Load weights

    tokens = th.randint(0, config.d_vocab, (2, 10))  # Batch size 2, sequence length 10
    logits = transformer(tokens)
    assert logits.shape == (2, 10, config.d_vocab)
