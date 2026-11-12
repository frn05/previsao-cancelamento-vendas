"""
Previsão de cancelamento de vendas com Random Forest.

Uso:
    python src/random_forest.py
    python src/random_forest.py --dataset vendas
    python src/random_forest.py --data-dir /caminho/com/os/csvs

Saídas (results/random_forest/): metrics.json, feature_importance.(csv|png),
confusion_matrix.png, exemplo_arvore_rf.png e predictions_new.csv.
"""

import sys

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree

from common import RANDOM_STATE, load_dataset, parse_args, run_experiment

MODEL_NAME = "Random Forest"
RESULTS_SUBDIR = "random_forest"


def build_model(ds):
    """Hiperparâmetros originais do projeto."""
    return RandomForestClassifier(
        n_estimators=300,  # número de árvores
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",  # compensa o desbalanceamento
        random_state=RANDOM_STATE,
        n_jobs=-1,  # usa todos os núcleos
    )


def extra_plots(pipeline, feature_names, output_dir):
    """Desenha UMA árvore da floresta, só como ilustração: a decisão final é a
    votação das 300 árvores, não desta."""
    floresta = pipeline.named_steps["model"]
    fig = plt.figure(figsize=(20, 10))
    plot_tree(
        floresta.estimators_[0],
        feature_names=list(feature_names),
        class_names=["Não cancelado", "Cancelado"],
        filled=True,
        rounded=True,
        fontsize=8,
        max_depth=3,
    )
    plt.title(f"Exemplo de 1 árvore dentro da Random Forest (de {len(floresta.estimators_)})")
    fig.savefig(output_dir / "exemplo_arvore_rf.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    args = parse_args("Random Forest - previsão de cancelamento", argv)
    ds = load_dataset(args.dataset, args.data_dir)
    run_experiment(sys.modules[__name__], ds, args.output_dir)


if __name__ == "__main__":
    main()
