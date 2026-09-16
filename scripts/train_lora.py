"""Treina o adapter QLoRA da Vortexa sobre o Qwen2.5-1.5B-Instruct."""

# Unsloth deve ser importado antes de transformers/trl para aplicar os patches de desempenho.
from unsloth import FastLanguageModel

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_LENGTH = 512
OUTPUT_DIR = "outputs/vortexa-lora-500"


def format_conversations(examples, tokenizer):
    """Converte um exemplo ou um lote de conversas em uma lista de textos."""
    conversations = examples["messages"]
    if conversations and isinstance(conversations[0], dict):
        conversations = [conversations]

    return [
        tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
        )
        for conversation in conversations
    ]


def main() -> None:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_LENGTH,
        dtype=None,  # Unsloth escolhe o tipo adequado para a GPU.
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    dataset = load_dataset(
        "json",
        data_files={"train": "data/train.jsonl", "eval": "data/eval.jsonl"},
    )

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=2,
        optim="adamw_8bit",
        weight_decay=0.01,
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        max_length=MAX_LENGTH,
        packing=False,
        report_to="none",
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        processing_class=tokenizer,
        formatting_func=lambda examples: format_conversations(examples, tokenizer),
    )

    print("Iniciando treinamento QLoRA...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Adapter salvo em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
