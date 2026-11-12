"""
Funções compartilhadas pelos três modelos (Decision Tree, Random Forest, XGBoost).

Por que existe este módulo?
---------------------------
Para a comparação entre modelos ser justa, os três precisam:
  * ler os mesmos dados;
  * usar EXATAMENTE o mesmo conjunto de treino/teste;
  * usar o mesmo pré-processamento, ajustado SOMENTE com o treino;
  * ser avaliados com as mesmas métricas.

Tudo isso fica centralizado aqui; cada script de modelo só define o estimador.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sem janela: funciona em servidor/CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ---------------------------------------------------------------------------
# Constantes de reprodutibilidade
# ---------------------------------------------------------------------------
RANDOM_STATE = 42  # mesma semente em split, modelos e amostragem
TEST_SIZE = 0.20  # 80% treino / 20% teste (base "pedidos")
TARGET = "cancelado"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Descrição dos datasets suportados
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DatasetSpec:
    """Onde procurar os arquivos de um dataset (na ordem de prioridade)."""

    id_col: str
    # lista de (pasta relativa à raiz do projeto, {papel: nome do arquivo})
    candidates: tuple


DATASETS: dict[str, DatasetSpec] = {
    # Base principal (a mesma usada originalmente no XGBoost): 100 mil pedidos
    # rotulados + 50 mil pedidos novos sem rótulo.
    # 1) data/raw    -> base completa (NÃO versionada; ver data/raw/README.md)
    # 2) data/sample -> amostra versionada, para qualquer pessoa rodar o projeto
    "pedidos": DatasetSpec(
        id_col="id_pedido",
        candidates=(
            (
                "data/raw",
                {"labeled": "pedidos_completo_100k.csv", "new": "pedidos_novos_50k.csv"},
            ),
            (
                "data/sample",
                {"labeled": "pedidos_completo_sample.csv", "new": "pedidos_novos_sample.csv"},
            ),
        ),
    ),
    # Base pequena usada originalmente em Decision Tree / Random Forest.
    # Já vem dividida em treino/teste/novas pelo autor; essa divisão é mantida.
    "vendas": DatasetSpec(
        id_col="id_venda",
        candidates=(
            (
                "data/sample/vendas",
                {
                    "train": "vendas_treino.csv",
                    "test": "vendas_teste.csv",
                    "new": "vendas_novas.csv",
                },
            ),
        ),
    ),
}


@dataclass
class Dataset:
    name: str
    source_dir: Path
    id_col: str
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    X_new: pd.DataFrame
    ids_new: pd.Series


# ---------------------------------------------------------------------------
# CLI comum
# ---------------------------------------------------------------------------
def parse_args(description: str, argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        default="pedidos",
        help="Base de dados a utilizar (padrão: pedidos).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Pasta com os CSVs. Se omitida, usa data/raw e, na falta dela, data/sample.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Pasta de saída. Padrão: results/<modelo> (ou results/<modelo>/vendas).",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Carga e preparação dos dados
# ---------------------------------------------------------------------------
def _locate_files(spec: DatasetSpec, data_dir: Path | None) -> tuple[Path, dict[str, Path]]:
    """Retorna (pasta, {papel: caminho}) da primeira combinação que existir."""
    tried = []
    for rel_dir, files in spec.candidates:
        folder = Path(data_dir) if data_dir else PROJECT_ROOT / rel_dir
        paths = {role: folder / name for role, name in files.items()}
        if all(p.exists() for p in paths.values()):
            return folder, paths
        tried.append(folder)
    if data_dir is None:
        searched = "\n  - ".join(str(p) for p in tried)
        raise FileNotFoundError(
            "Arquivos de dados não encontrados. Pastas verificadas:\n  - "
            f"{searched}\nConsulte data/raw/README.md para saber quais arquivos colocar."
        )
    raise FileNotFoundError(
        f"A pasta '{data_dir}' não contém os arquivos esperados: "
        + ", ".join(sorted({n for _, f in spec.candidates for n in f.values()}))
    )


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features a partir da data/hora do pedido (transformação linha a linha).

    Não aprende nada com os dados (sem médias, sem categorias), portanto pode
    ser aplicada antes do split sem causar vazamento.
    """
    df = df.copy()
    if "data_pedido" in df.columns:
        df["mes_pedido"] = pd.to_datetime(df["data_pedido"]).dt.month
        df = df.drop(columns=["data_pedido"])
    if "dia_semana" in df.columns:  # convenção da base: 0 = segunda ... 6 = domingo
        df["fim_de_semana"] = (df["dia_semana"] >= 5).astype(int)
    if "hora_pedido" in df.columns:
        df["periodo_dia"] = pd.cut(
            df["hora_pedido"],
            bins=[-1, 5, 11, 17, 23],
            labels=["madrugada", "manha", "tarde", "noite"],
        ).astype(str)
    return df


def load_dataset(name: str = "pedidos", data_dir: Path | None = None) -> Dataset:
    """Carrega o dataset e devolve treino/teste/novos já separados.

    O split é determinístico (RANDOM_STATE) e estratificado pelo alvo, então
    qualquer modelo que chame esta função recebe exatamente o mesmo teste.
    """
    spec = DATASETS[name]
    folder, paths = _locate_files(spec, data_dir)
    id_col = spec.id_col

    if "labeled" in paths:  # base única rotulada -> split estratificado
        df = add_date_features(pd.read_csv(paths["labeled"]))
        X = df.drop(columns=[TARGET, id_col])
        y = df[TARGET]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
    else:  # treino/teste já fornecidos
        df_train = add_date_features(pd.read_csv(paths["train"]))
        df_test = add_date_features(pd.read_csv(paths["test"]))
        X_train, y_train = df_train.drop(columns=[TARGET, id_col]), df_train[TARGET]
        X_test, y_test = df_test.drop(columns=[TARGET, id_col]), df_test[TARGET]

    df_new = add_date_features(pd.read_csv(paths["new"]))
    ids_new = df_new[id_col].reset_index(drop=True)
    X_new = df_new.drop(columns=[id_col])

    # Validação: os dados novos precisam ter as mesmas colunas do treino
    missing = set(X_train.columns) - set(X_new.columns)
    if missing:
        raise ValueError(f"Dados novos sem as colunas: {sorted(missing)}")
    X_new = X_new[X_train.columns]  # mesma ordem de colunas

    return Dataset(name, folder, id_col, X_train, X_test, y_train, y_test, X_new, ids_new)


# ---------------------------------------------------------------------------
# Pré-processamento (ajustado apenas com o treino, dentro do Pipeline)
# ---------------------------------------------------------------------------
def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Imputação + One-Hot Encoding.

    * Numéricas: mediana do TREINO para valores ausentes.
    * Categóricas: moda do TREINO para ausentes + One-Hot Encoding.
    * handle_unknown='ignore': categoria inédita em teste/novos vira vetor de
      zeros, em vez de quebrar ou de "vazar" o vocabulário do teste para o treino.
    Os modelos de árvore não precisam de normalização, por isso não há scaler.
    """
    cat_cols = [c for c in X_train.columns if not pd.api.types.is_numeric_dtype(X_train[c])]
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat_cols,
            ),
        ],
        verbose_feature_names_out=False,
    )


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def evaluate(y_true, y_pred, y_proba) -> dict:
    """Calcula as métricas de classificação (classe positiva = cancelado)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _environment_info() -> dict:
    info = {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
    }
    try:
        import xgboost

        info["xgboost"] = xgboost.__version__
    except ImportError:  # xgboost é opcional para DT/RF
        pass
    return info


# ---------------------------------------------------------------------------
# Execução padrão de um experimento
# ---------------------------------------------------------------------------
def default_output_dir(results_subdir: str, dataset_name: str) -> Path:
    base = PROJECT_ROOT / "results" / results_subdir
    return base if dataset_name == "pedidos" else base / dataset_name


def run_experiment(model_module, ds: Dataset, output_dir: Path | None = None) -> dict:
    """Treina, avalia e salva os artefatos de um modelo.

    `model_module` precisa expor:
      MODEL_NAME, RESULTS_SUBDIR, build_model(ds) e (opcional) extra_plots(...).
    """
    output_dir = Path(output_dir) if output_dir else default_output_dir(
        model_module.RESULTS_SUBDIR, ds.name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Aviso explícito quando rodando sobre a amostra
    if "sample" in ds.source_dir.parts and ds.name == "pedidos":
        print(
            "[AVISO] Usando a AMOSTRA em data/sample. As métricas não representam "
            "a base completa (veja data/raw/README.md)."
        )

    print(
        f"[{model_module.MODEL_NAME}] dados: {ds.source_dir} | treino={len(ds.X_train)} "
        f"teste={len(ds.X_test)} novos={len(ds.X_new)}"
    )

    # Pipeline = pré-processamento + modelo. O .fit() só enxerga o TREINO.
    pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor(ds.X_train)),
            ("model", model_module.build_model(ds)),
        ]
    )
    pipeline.fit(ds.X_train, ds.y_train)

    y_pred = pipeline.predict(ds.X_test)
    y_proba = pipeline.predict_proba(ds.X_test)[:, 1]
    metrics = evaluate(ds.y_test, y_pred, y_proba)

    # --- Importância das variáveis -------------------------------------
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    model = pipeline.named_steps["model"]
    importances = (
        pd.DataFrame({"variavel": feature_names, "importancia": model.feature_importances_})
        .sort_values("importancia", ascending=False)
        .reset_index(drop=True)
    )
    importances.to_csv(output_dir / "feature_importance.csv", index=False)
    save_importance_plot(importances, model_module.MODEL_NAME, output_dir)

    # --- Matriz de confusão --------------------------------------------
    save_confusion_matrix_plot(ds.y_test, y_pred, model_module.MODEL_NAME, output_dir)

    # --- Gráficos específicos do modelo (árvore etc.) --------------------
    if hasattr(model_module, "extra_plots"):
        model_module.extra_plots(pipeline, feature_names, output_dir)

    # --- Previsões para os dados novos (sem rótulo) ----------------------
    proba_new = pipeline.predict_proba(ds.X_new)[:, 1]
    predictions = pd.DataFrame(
        {
            ds.id_col: ds.ids_new,
            "previsao_cancelado": pipeline.predict(ds.X_new),
            "probabilidade_cancelamento": proba_new.round(4),
        }
    )
    predictions.to_csv(output_dir / "predictions_new.csv", index=False)

    # --- Métricas + metadados da execução ---------------------------------
    report = {
        "model": model_module.MODEL_NAME,
        "dataset": ds.name,
        "data_dir": _relative(ds.source_dir),
        "n_train": int(len(ds.X_train)),
        "n_test": int(len(ds.X_test)),
        "n_new": int(len(ds.X_new)),
        "positive_rate_train": float(ds.y_train.mean()),
        "positive_rate_test": float(ds.y_test.mean()),
        "random_state": RANDOM_STATE,
        "hyperparameters": {k: _jsonable(v) for k, v in model.get_params().items()},
        "metrics": metrics,
        "environment": _environment_info(),
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print_metrics(model_module.MODEL_NAME, metrics)
    print(f"Top 5 variáveis:\n{importances.head(5).to_string(index=False)}")
    print(f"Taxa de cancelamento prevista (dados novos): {predictions['previsao_cancelado'].mean():.3f}")
    print(f"Artefatos salvos em: {_relative(output_dir)}\n")

    return {"report": report, "y_pred": y_pred, "y_proba": y_proba, "importances": importances}


# ---------------------------------------------------------------------------
# Helpers de saída
# ---------------------------------------------------------------------------
def _relative(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def print_metrics(name: str, m: dict) -> None:
    cm = m["confusion_matrix"]
    print("=" * 60)
    print(f"AVALIAÇÃO NO CONJUNTO DE TESTE - {name}")
    print("=" * 60)
    print(f"Accuracy : {m['accuracy']:.3f}")
    print(f"Precision: {m['precision']:.3f}")
    print(f"Recall   : {m['recall']:.3f}")
    print(f"F1-score : {m['f1']:.3f}")
    print(f"ROC-AUC  : {m['roc_auc']:.3f}")
    print(f"Matriz de confusão [[TN FP] [FN TP]]: [[{cm['tn']} {cm['fp']}] [{cm['fn']} {cm['tp']}]]")


def save_importance_plot(importances: pd.DataFrame, model_name: str, output_dir: Path, top: int = 15) -> None:
    top_df = importances.head(top).sort_values("importancia")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top_df["variavel"], top_df["importancia"], color="#4C72B0")
    ax.set_xlabel("Importância")
    ax.set_title(f"Top {top} variáveis mais importantes - {model_name}")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix_plot(y_true, y_pred, model_name: str, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["Não cancelado", "Cancelado"],
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title(f"Matriz de confusão (teste) - {model_name}")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    sys.exit("common.py é um módulo auxiliar. Execute decision_tree.py, random_forest.py, "
             "xgboost_model.py ou compare_models.py.")
