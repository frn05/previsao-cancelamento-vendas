"""
Compara Decision Tree, Random Forest e XGBoost no MESMO conjunto de teste.

Uso:
    python src/compare_models.py
    python src/compare_models.py --dataset vendas

O script treina os três modelos (mesmo split, mesmo pré-processamento), salva os
artefatos individuais em results/<modelo>/ e gera em results/comparison/:
    metrics_comparison.csv / .md   tabela comparativa (métricas calculadas na execução)
    metrics_comparison.png         barras com Accuracy, Precision, Recall, F1 e ROC-AUC
    roc_curves.png                 curvas ROC dos três modelos
    confusion_matrices.png         matrizes de confusão lado a lado
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve

import decision_tree
import random_forest
import xgboost_model
from common import PROJECT_ROOT, load_dataset, parse_args, run_experiment

MODELS = [decision_tree, random_forest, xgboost_model]
METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1-score",
    "roc_auc": "ROC-AUC",
}
COLORS = ["#4C72B0", "#DD8452", "#2E7D32"]


def build_table(results: dict) -> pd.DataFrame:
    rows = []
    for name, res in results.items():
        m = res["report"]["metrics"]
        cm = m["confusion_matrix"]
        rows.append(
            {
                "Model": name,
                **{METRIC_LABELS[k]: m[k] for k in METRIC_COLUMNS},
                "TN": cm["tn"],
                "FP": cm["fp"],
                "FN": cm["fn"],
                "TP": cm["tp"],
            }
        )
    return pd.DataFrame(rows)


def to_markdown(table: pd.DataFrame) -> str:
    """Tabela em Markdown sem depender de 'tabulate'."""
    cols = list(table.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in table.iterrows():
        cells = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_metrics(table: pd.DataFrame, path: Path) -> None:
    labels = [METRIC_LABELS[k] for k in METRIC_COLUMNS]
    x = np.arange(len(labels))
    width = 0.8 / len(table)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, (_, row) in enumerate(table.iterrows()):
        bars = ax.bar(x + i * width, [row[l] for l in labels], width, label=row["Model"], color=COLORS[i])
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    ax.set_xticks(x + width * (len(table) - 1) / 2, labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (conjunto de teste)")
    ax.set_title("Comparação de desempenho dos modelos")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc(results: dict, y_test, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for (name, res), color in zip(results.items(), COLORS):
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        auc = res["report"]["metrics"]["roc_auc"]
        ax.plot(fpr, tpr, color=color, label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Aleatório")
    ax.set_xlabel("Taxa de falsos positivos")
    ax.set_ylabel("Taxa de verdadeiros positivos (Recall)")
    ax.set_title("Curvas ROC - conjunto de teste")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusions(results: dict, y_test, path: Path) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4.6))
    for ax, (name, res) in zip(np.atleast_1d(axes), results.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            res["y_pred"],
            display_labels=["Não cancelado", "Cancelado"],
            cmap="Blues",
            colorbar=False,
            ax=ax,
        )
        ax.set_title(name)
    fig.suptitle("Matrizes de confusão - mesmo conjunto de teste")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    args = parse_args("Comparação entre Decision Tree, Random Forest e XGBoost", argv)
    ds = load_dataset(args.dataset, args.data_dir)  # UM split para os três modelos

    results = {}
    for module in MODELS:
        results[module.MODEL_NAME] = run_experiment(module, ds)

    out = args.output_dir or (
        PROJECT_ROOT / "results" / "comparison" / ("" if args.dataset == "pedidos" else args.dataset)
    )
    out.mkdir(parents=True, exist_ok=True)

    table = build_table(results)
    table.to_csv(out / "metrics_comparison.csv", index=False, float_format="%.4f")
    (out / "metrics_comparison.md").write_text(to_markdown(table) + "\n", encoding="utf-8")
    plot_metrics(table, out / "metrics_comparison.png")
    plot_roc(results, ds.y_test, out / "roc_curves.png")
    plot_confusions(results, ds.y_test, out / "confusion_matrices.png")

    print("=" * 60)
    print(f"COMPARAÇÃO FINAL (teste: {len(ds.y_test)} linhas | base: {ds.name})")
    print("=" * 60)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nArquivos salvos em: {out}")


if __name__ == "__main__":
    main()
