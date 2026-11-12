"""
Previsão de cancelamento de vendas com Decision Tree.

Uso:
    python src/decision_tree.py                      # base "pedidos" (data/raw ou data/sample)
    python src/decision_tree.py --dataset vendas     # base pequena original
    python src/decision_tree.py --data-dir /caminho/com/os/csvs

Saídas (results/decision_tree/): metrics.json, feature_importance.(csv|png),
confusion_matrix.png, arvore_decisao.png e predictions_new.csv.
"""

import sys

import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

from common import RANDOM_STATE, load_dataset, parse_args, run_experiment

MODEL_NAME = "Decision Tree"
RESULTS_SUBDIR = "decision_tree"


def build_model(ds):
    """Hiperparâmetros originais do projeto, agora com class_weight para
    compensar o desbalanceamento (mesma estratégia usada na Random Forest)."""
    return DecisionTreeClassifier(
        max_depth=5,  # limita a profundidade para evitar overfitting
        min_samples_leaf=10,  # mínimo de amostras por folha
        class_weight="balanced",  # dá mais peso à classe minoritária (cancelado)
        random_state=RANDOM_STATE,
    )


def extra_plots(pipeline, feature_names, output_dir):
    """Desenha os primeiros níveis da árvore (a árvore completa fica ilegível)."""
    fig = plt.figure(figsize=(20, 10))
    plot_tree(
        pipeline.named_steps["model"],
        feature_names=list(feature_names),
        class_names=["Não cancelado", "Cancelado"],
        filled=True,
        rounded=True,
        fontsize=8,
        max_depth=3,
    )
    plt.title("Árvore de Decisão - Previsão de Cancelamento de Vendas (3 primeiros níveis)")
    fig.savefig(output_dir / "arvore_decisao.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    args = parse_args("Decision Tree - previsão de cancelamento", argv)
    ds = load_dataset(args.dataset, args.data_dir)
    run_experiment(sys.modules[__name__], ds, args.output_dir)


if __name__ == "__main__":
    main()
