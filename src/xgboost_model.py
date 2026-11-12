"""
Previsão de cancelamento de pedidos com XGBoost.

(O arquivo se chama xgboost_model.py, e não xgboost.py, para não "esconder" a
biblioteca xgboost no import.)

Uso:
    python src/xgboost_model.py
    python src/xgboost_model.py --dataset vendas
    python src/xgboost_model.py --data-dir /caminho/com/os/csvs

Saídas (results/xgboost/): metrics.json, feature_importance.(csv|png),
confusion_matrix.png e predictions_new.csv.

Compatibilidade: requer xgboost >= 2.1.4 quando usado com scikit-learn >= 1.6
(versões anteriores do XGBoost falham em Pipeline/clone com o scikit-learn novo).
"""

import sys

from xgboost import XGBClassifier

from common import RANDOM_STATE, load_dataset, parse_args, run_experiment

MODEL_NAME = "XGBoost"
RESULTS_SUBDIR = "xgboost"


def build_model(ds):
    """Hiperparâmetros originais do projeto.

    scale_pos_weight = negativos / positivos, calculado SOMENTE no treino.
    As categóricas chegam já codificadas (One-Hot) pelo pipeline, igual aos
    outros modelos, o que mantém a comparação justa.

    Diferença em relação ao script original: o conjunto de TESTE não é mais
    passado como `eval_set`. Sem early stopping ele não influenciava o modelo,
    mas manter o teste dentro do fit é má prática (se alguém ativar early
    stopping depois, o teste passa a vazar para o treinamento).
    """
    negativos = int((ds.y_train == 0).sum())
    positivos = int((ds.y_train == 1).sum())
    return XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,  # 80% das linhas por árvore
        colsample_bytree=0.8,  # 80% das colunas por árvore
        min_child_weight=5,
        scale_pos_weight=negativos / positivos,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def main(argv=None):
    args = parse_args("XGBoost - previsão de cancelamento", argv)
    ds = load_dataset(args.dataset, args.data_dir)
    run_experiment(sys.modules[__name__], ds, args.output_dir)


if __name__ == "__main__":
    main()
