# config.py
from dataclasses import dataclass

import torch as th


@dataclass
class TransformerConfig:
    d_model: int = 768
    debug: bool = True
    layer_norm_eps: float = 1e-5
    d_vocab: int = 50257
    init_range: float = 0.02
    n_ctx: int = 1024
    d_mlp: int = 3072
    n_heads: int = 12
    n_layers: int = 12


@dataclass
class TrainingConfig:
    batch_size: int = 16
    n_epochs: int = 20
    save_freq: int = 10
    save_path: str = "models"
    max_steps_per_epoch: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-2
    device = "cuda" if th.cuda.is_available() else "cpu"
    debug: bool = False
    wandb_project: str = "transformer_trainer"
    wandb_display_name: str | None = None
    train_data_path: str = ""
    val_data_path: str = ""
    n_workers: int = 1
