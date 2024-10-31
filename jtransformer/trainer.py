from jtransformer.config import TrainingConfig
from jtransformer.char_tokenizer import CharTokenizer

import os
from typing import Dict, Optional, Union, cast
from abc import ABC, abstractmethod  # For extensible trainers

from transformers import PreTrainedTokenizer
import torch as th
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.optim.lr_scheduler import _LRScheduler
import wandb
from tqdm import tqdm

from datasets import Dataset, load_dataset, load_from_disk


class Jtrainer(ABC):
    def __init__(
        self,
        cfg: TrainingConfig,
        model: nn.Module,
        tokenizer: Union[PreTrainedTokenizer, CharTokenizer],
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.device = cfg.device
        self.optimizer: Optional[optim.Optimizer] = self._setup_optimizer()
        self.scheduler: Optional[optim.lr_scheduler._LRScheduler] = (
            self._setup_scheduler()
        )
        self.criterion: Optional[nn.Module] = self._setup_loss()
        self.train_dataloader: Optional[DataLoader] = None
        self.val_dataloader: Optional[DataLoader] = None
        self.n_steps = 0
        self.best_val_loss = float("inf")
        self.early_stopping_counter = 0

    @abstractmethod
    def _setup_loss(self) -> nn.Module:
        """Subclasses implement their specific loss function setup."""
        pass

    @abstractmethod
    def _setup_optimizer(self) -> optim.Optimizer:
        """Subclasses implement their optimizer setup."""
        pass

    @abstractmethod
    def val_metrics(self, logits: th.Tensor, targets: th.Tensor) -> dict[str, float]:
        """Subclasses implement their own validation metric."""
        pass

    def _setup_scheduler(self) -> Optional[_LRScheduler]:
        """Set up the learning rate scheduler based on the config."""
        assert (
            self.optimizer is not None
        ), "Optimizer must be initialized before scheduler."
        scheduler_kwargs = self.cfg.scheduler_kwargs or {}

        if self.cfg.scheduler_type == "steplr":
            return cast(
                _LRScheduler,
                optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=scheduler_kwargs.get("step_size", 10),
                    gamma=scheduler_kwargs.get("gamma", 0.1),
                ),
            )
        elif self.cfg.scheduler_type == "cosine":
            return cast(
                _LRScheduler,
                optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=scheduler_kwargs.get("T_max", self.cfg.n_epochs),
                ),
            )
        elif self.cfg.scheduler_type == "reduce_on_plateau":
            return cast(
                _LRScheduler,
                optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode=scheduler_kwargs.get("mode", "min"),
                    patience=scheduler_kwargs.get("patience", 3),
                    factor=scheduler_kwargs.get("factor", 0.5),
                ),
            )
        return None

    def train_step(self, batch: Dict[str, th.Tensor]) -> float:
        """Shared logic for a single training step."""
        self.model.train()
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["label"].float().to(self.device)  # Cast to float

        predictions = self.model(input_ids).squeeze(-1)
        loss = self.criterion(predictions, labels)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.n_steps += 1
        wandb.log({"train_loss": loss.item()}, step=self.n_steps)
        return loss.item()

    def val_step(self, batch: Dict[str, th.Tensor]) -> dict:
        """Shared logic for a single validation step."""
        assert (
            self.criterion is not None
        ), "Criterion must be initalized before validation steps."
        self.model.eval()
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["label"].float().to(self.device)

        with th.no_grad():
            logits = self.model(input_ids)
            return dict(
                **self.val_metrics(logits, labels),
                n_datapoints=labels.size(0),
            )

    def aggregate_metrics(
        self, val_metrics_list: list[Dict[str, float]]
    ) -> Dict[str, float]:
        """Aggregate metrics across the entire validation set using weighted averages."""
        aggregated = {}
        total_datapoints = sum(metrics["n_datapoints"] for metrics in val_metrics_list)

        # Sum up each metric weighted by the number of datapoints in each batch
        for key in val_metrics_list[0].keys():
            if key == "n_datapoints":
                continue  # Skip this field in the aggregation

            weighted_sum = sum(
                metrics[key] * metrics["n_datapoints"] for metrics in val_metrics_list
            )
            aggregated[key] = (
                weighted_sum / total_datapoints
            )  # Compute weighted average

        return aggregated

    def train(self) -> None:
        """Main training loop, shared across trainers."""
        self.model.to(self.device)
        # Now load the dataloaders for training and validation
        self._load_data()
        assert (
            self.train_dataloader is not None
        ), "There was a problem loading the data into the training dataloader."
        assert (
            self.val_dataloader is not None
        ), "There was a problem loading the data into the validation dataloader."

        if not self.cfg.debug:
            wandb.init(
                project=self.cfg.wandb_project,
                name=self.cfg.wandb_display_name,
                config=self.cfg.to_dict(),
            )

        progress_bar = tqdm(total=self.cfg.n_epochs * self.cfg.max_steps_per_epoch)

        for epoch in range(self.cfg.n_epochs):
            steps = 0
            for batch in self.train_dataloader:
                loss = self.train_step(batch)
                progress_bar.update(1)
                progress_bar.set_description(f"Epoch {epoch+1}, Loss: {loss:.4f}")

                steps += 1

                if steps > self.cfg.max_steps_per_epoch:
                    break

            if epoch % self.cfg.save_freq == 0:
                save_path = os.path.join(self.cfg.save_path, f"epoch_{epoch+1}")
                self.model.save(save_path)

            val_metrics_list = [self.val_step(b) for b in self.val_dataloader]

            aggregated_metrics = self.aggregate_metrics(val_metrics_list)

            if not self.cfg.debug:
                wandb.log(aggregated_metrics, step=self.n_steps)

            progress_bar.set_postfix(aggregated_metrics)

            val_loss = aggregated_metrics.get("val_loss")
            if val_loss is not None and self._early_stopping(val_loss):
                print("Early stopping triggered.")
                break

            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)  # Pass val_loss to ReduceLROnPlateau
            elif self.scheduler is not None:
                self.scheduler.step()  # Step other schedulers normally
            if not self.cfg.debug:
                wandb.log({"lr": self.scheduler.get_lr()})

        save_path = os.path.join(self.cfg.save_path, "final")
        self.model.save(save_path)

    def _early_stopping(self, val_loss: float) -> bool:
        """Checks if early stopping should be triggered."""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1

        return self.early_stopping_counter >= self.cfg.early_stopping_patience

    def save_model(self, save_name: str) -> None:
        """Save the model with error handling."""
        try:
            save_path = os.path.join(self.cfg.save_path, save_name)
            os.makedirs(save_path, exist_ok=True)
            th.save(self.model.state_dict(), os.path.join(save_path, "model.pth"))
            print(f"Model saved to {save_path}")
        except Exception as e:
            print(f"Error saving model: {e}")

    def _load_data(self) -> None:
        """Load train and validation datasets."""
        if self.train_dataloader is None:
            train_dataset = load_from_disk(self.cfg.train_data_path)
            train_dataset.set_format(type="torch")
            self.train_dataloader = DataLoader(
                train_dataset,
                batch_size=self.cfg.batch_size,
                num_workers=self.cfg.n_workers,
                shuffle=True,
            )
        if self.val_dataloader is None:
            val_dataset = load_from_disk(self.cfg.val_data_path)
            val_dataset.set_format(type="torch")
            self.val_dataloader = DataLoader(
                val_dataset,
                batch_size=self.cfg.batch_size,
                num_workers=self.cfg.n_workers,
                shuffle=False,
            )

    @classmethod
    def create_dataset(
        cls,
        tokenizer: PreTrainedTokenizer,
        file_path: Optional[str] = None,
        hf_dataset_name: Optional[str] = None,
        tokenizer_kwargs: dict = {},
        chunk_size: Optional[int] = None,
        overlap_size: int = 0,
    ) -> Dataset:
        """
        Converts a plain text file into a tokenized dataset for next-token prediction.
        """

        def read_txt_to_dict_chunks(
            file_path: str,
            tokenizer: PreTrainedTokenizer,
            chunk_size: int,
            overlap_size: int,
            tokenizer_kwargs: dict = {},
        ) -> list[dict]:
            """
            Streams a large text file in chunks with overlap to avoid loss of context.
            Each chunk contains `chunk_size` tokens, with `overlap_size` tokens overlapping between chunks.
            """
            chunks = []
            current_chunk = []

            # Open the file and read it line by line
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    # Ignore empty lines
                    line = line.strip()
                    if not line:
                        continue

                    # Tokenize the line and append to the current chunk
                    tokens = tokenizer(line + "\n", padding=False, truncation=False)[
                        "input_ids"
                    ]
                    current_chunk.extend(tokens)

                    # If the current chunk exceeds the desired chunk size, save it
                    while len(current_chunk) >= chunk_size:
                        chunks.append({"input_ids": current_chunk[:chunk_size]})

                        # Create the next chunk starting with the overlapping part
                        current_chunk = current_chunk[chunk_size - overlap_size :]

            # Add the last chunk if any tokens remain
            if current_chunk:
                if len(current_chunk) < chunk_size:
                    current_chunk.extend(
                        [tokenizer.pad_token_id] * (chunk_size - len(current_chunk))
                    )
                chunks.append({"input_ids": current_chunk})

            return chunks

        def read_txt_to_dict_lines(file_path: str):
            """Reads a plain text file and returns a list of dictionaries containing plain text."""
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            return [{"text": line.strip() + "\n"} for line in lines if line.strip()]

        def tokenize_function(examples):
            """Tokenizes the input text using the provided tokenizer."""
            return tokenizer(examples["text"], **tokenizer_kwargs)

        if hf_dataset_name:
            tokenized_dataset = load_dataset(hf_dataset_name).map(
                tokenize_function, batched=True
            )
        elif file_path:
            # Read text and convert to dataset
            if chunk_size:
                data_dict = read_txt_to_dict_chunks(
                    file_path=file_path,
                    tokenizer=tokenizer,
                    chunk_size=chunk_size,
                    overlap_size=overlap_size,
                    tokenizer_kwargs=tokenizer_kwargs,
                )
                tokenized_dataset = Dataset.from_list(data_dict)
            else:
                data_dict = read_txt_to_dict_lines(file_path)
                tokenized_dataset = Dataset.from_list(data_dict).map(
                    tokenize_function, batched=True
                )
        else:
            raise Exception

        return tokenized_dataset


class NextTokenPredictionTrainer(Jtrainer):
    def _setup_loss(self) -> nn.Module:
        pad_token_id = getattr(self.tokenizer, "pad_token_id", -100)
        return nn.CrossEntropyLoss(ignore_index=pad_token_id)

    def _setup_optimizer(self) -> optim.Optimizer:
        return optim.AdamW(self.model.parameters(), lr=self.cfg.lr)

    def val_metrics(self, logits: th.Tensor, labels: th.Tensor) -> Dict[str, float]:
        """Computes multiple metrics for a given batch."""
        assert self.criterion is not None
        loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1)).item()
        predictions = logits.argmax(dim=-1)
        accuracy = (predictions == labels).float().mean().item()

        return {"val_loss": loss, "val_accuracy": accuracy}

    def train_step(self, batch: Dict[str, th.Tensor]) -> float:
        """Override to handle input-label shifting."""
        assert self.criterion is not None
        assert self.optimizer is not None
        self.model.train()
        input_ids = batch["input_ids"].to(self.device)

        # Shift inputs to create labels for next-token prediction
        labels = input_ids[:, 1:].clone()
        input_ids = input_ids[:, :-1]

        # Forward pass and loss calculation
        logits = self.model(input_ids)  # [batch_size, seq_len-1, vocab_size]
        loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.n_steps += 1
        wandb.log({"train_loss": loss.item()}, step=self.n_steps)
        return loss.item()

    def val_step(self, batch: Dict[str, th.Tensor]) -> dict:
        """Shared logic for a single validation step."""
        self.model.eval()
        input_ids = batch["input_ids"].to(self.device)

        # Shift inputs to create labels for next-token prediction
        labels = input_ids[:, 1:].clone()
        input_ids = input_ids[:, :-1]

        with th.no_grad():
            logits = self.model(input_ids)  # [batch_size, seq_len-1, vocab_size]

            # Compute metrics for this batch
            metrics = self.val_metrics(logits, labels)

            # Add the number of datapoints in the batch
            return {**metrics, "n_datapoints": labels.size(0)}
