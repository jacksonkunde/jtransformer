from jtransformer.trainer import Jtrainer
from jtransformer.config import TrainingConfig, TransformerConfig
from jtransformer.modules import Jtransformer
from transformers import AutoTokenizer
from jtransformer.char_tokenizer import CharTokenizer
import os


def main():
    contex_window_size = 256
    file_path = "path/to/input.txt"  # add your filepath here

    tokenizer = CharTokenizer()

    tokenizer_kwargs = {
        "max_length": contex_window_size,
        "padding": True,
        "truncation": True,
    }
    dataset = Jtrainer.create_dataset(
        tokenizer=tokenizer,
        file_path=file_path,
        tokenizer_kwargs=tokenizer_kwargs,
        chunk_size=contex_window_size,
        overlap_size=contex_window_size // 4,
    )

    tokenizer.save("char_tokenizer.json")

    train_dataset = dataset[: int(len(dataset) * 0.9)]
    val_dataset = dataset[int(len(dataset) * 0.9) :]

    print(type(val_dataset))

    dataset.from_dict(train_dataset).save_to_disk("./train_dataset")
    dataset.from_dict(val_dataset).save_to_disk("./val_dataset")

    training_cfg = TrainingConfig(
        debug=False,
        batch_size=32,
        n_epochs=8,
        train_data_path=os.path.abspath("./train_dataset"),
        val_data_path=os.path.abspath("./val_dataset"),
    )

    model_cfg = TransformerConfig(
        d_model=384,
        n_ctx=contex_window_size,
        d_mlp=4 * 384,
        n_heads=6,
        n_layers=6,
        d_vocab=len(tokenizer.vocab),
    )

    model = Jtransformer(model_cfg)

    trainer = Jtrainer(cfg=training_cfg, model=model, tokenizer=tokenizer)

    trainer.train()


if __name__ == "__main__":
    main()
