from jtransformer.trainer import Jtrainer
from jtransformer.config import TrainingConfig, TransformerConfig
from jtransformer.modules import Jtransformer
from transformers import AutoTokenizer
from jtransformer.char_tokenizer import CharTokenizer
import os


def main():
    file_path = "/Users/newemployee/Desktop/transformer/cookbook/shakespeare/input.txt"

    tokenizer = CharTokenizer()

    tokenizer_kwargs = {"max_length": 64, "padding": True, "truncation": True}
    dataset = Jtrainer.create_dataset(
        tokenizer=tokenizer,
        file_path=file_path,
        tokenizer_kwargs=tokenizer_kwargs,
        chunk_size=256,
        overlap_size=64,
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
        n_ctx=256,
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
