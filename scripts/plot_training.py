"""Gera um grafico comparando os historicos dos dois treinamentos."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "40 exemplos": ROOT / "outputs/vortexa-lora/checkpoint-25/trainer_state.json",
    "500 exemplos": ROOT / "outputs/vortexa-lora-500/checkpoint-250/trainer_state.json",
}
OUTPUT = ROOT / "docs/training-comparison.png"


def load_history(path: Path) -> tuple[list[dict], list[dict]]:
    state = json.loads(path.read_text(encoding="utf-8"))
    history = state["log_history"]
    losses = [item for item in history if "loss" in item]
    eval_losses = [item for item in history if "eval_loss" in item]
    return losses, eval_losses


def main() -> None:
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    for label, path in RUNS.items():
        losses, eval_losses = load_history(path)
        axes[0].plot(
            [item["epoch"] for item in losses],
            [item["loss"] for item in losses],
            label=label,
            linewidth=2,
        )
        axes[1].plot(
            [item["epoch"] for item in eval_losses],
            [item["eval_loss"] for item in eval_losses],
            marker="o",
            label=label,
            linewidth=2,
        )

    axes[0].set_title("Loss de treinamento")
    axes[1].set_title("Loss de avaliacao")
    axes[1].set_xlabel("Epoca")
    for axis in axes:
        axis.set_ylabel("Loss")
        axis.grid(True, alpha=0.25)
        axis.legend()

    figure.suptitle("Comparacao dos treinamentos QLoRA da Vortexa", fontsize=15)
    figure.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=160, bbox_inches="tight")
    print(f"Grafico salvo em: {OUTPUT}")


if __name__ == "__main__":
    main()
