# Methodology and technical decisions

This document explains *why* the project is built the way it is. It also records what was changed relative to the
first version of the scripts (one script per model, each with its own preprocessing).

## 1. Problem framing

* **Task:** binary classification — will an order be cancelled (`cancelado = 1`) or not (`0`)?
* **Positive class:** cancelled. Precision and recall refer to this class.
* **Unit of prediction:** one order; the model outputs a class and a cancellation probability.

## 2. Data

Two datasets exist in the project:

| Dataset | Used by | Structure |
|---|---|---|
| `pedidos` (default) | all three models | 100,000 labelled orders + 50,000 unlabelled "new" orders, 29 raw feature columns (+3 derived from the date) |
| `vendas` (`--dataset vendas`) | first versions of Decision Tree / Random Forest | 800 train + 200 test + 50 new, 12 features, pre-split by the author |

The first versions of the project evaluated the Decision Tree and Random Forest on `vendas` and the XGBoost on
`pedidos`, so their metrics **could not be compared**. Now all three models run on `pedidos` with exactly the same
split, and `vendas` is kept as a small dataset for quick tests.

Data checks performed (see `notebooks/analise_exploratoria.ipynb`): no missing values, no duplicated IDs or rows, no
ID overlap between the labelled and the new files, cancellation rate of 35.5% in the labelled data.

> The origin of the data is not documented and several signs indicate it is simulated (see `data/sample/README.md`).

## 3. Train / test split

* `train_test_split(test_size=0.20, random_state=42, stratify=y)` — 80,000 train / 20,000 test.
* Stratification keeps the 35.5% cancellation rate in both parts.
* The split lives in `common.load_dataset()`, so **every model receives the identical test set**. This is what makes the
  comparison fair.
* The split is random, not temporal: order dates are spread over the same period in the labelled and in the new data
  (2024-01-01 → 2025-06-28), so there is no "future" to validate against. See *Limitations*.

## 4. Preprocessing (and the data-leakage review)

### What the original scripts did

The original scripts concatenated train, test and new data, ran `pd.get_dummies` on the whole thing and split it
back. Strictly speaking this **did not leak the target**: `get_dummies` is stateless — it uses no statistic of the
data and never looks at `cancelado`. But the approach is fragile:

* the set of encoded columns was defined using the test and the new data (the encoder "knew" categories of data that
  should be unseen);
* in a real deployment, new orders arrive one batch at a time and may contain categories never seen in training;
* the moment any *fitted* transformation is added (median imputation, scaling, target encoding…), the same pattern
  would leak information from test/new data into training.

### What the project does now

A scikit-learn `Pipeline` with a `ColumnTransformer` (`common.build_preprocessor`):

| Column type | Missing values | Encoding |
|---|---|---|
| Numeric | median **of the training set** | none (tree models do not need scaling) |
| Categorical | most frequent value **of the training set** | `OneHotEncoder(handle_unknown="ignore")` |

* The pipeline is fitted **only** on the training set; test and new data are just transformed.
* A category that never appeared in training becomes an all-zero vector instead of raising an error.
* Missing values are handled even though the current data has none (a quick manual test with an unseen payment method
  and missing numeric/categorical values scored without errors).
* Date-derived features (`mes_pedido`, `fim_de_semana`, `periodo_dia`) are computed row by row before the split. They
  learn nothing from the data, so this is safe. `dia_semana` follows Monday = 0 (verified against `data_pedido`), so
  `fim_de_semana = dia_semana >= 5` is correct.

## 5. Class imbalance

The positive class is ~35.5% of the data — imbalanced, but not extremely. Strategy: **class weighting**, no resampling
(resampling would also have to be done inside the training folds to avoid leakage, and the imbalance does not justify
the extra complexity).

| Model | Mechanism |
|---|---|
| Decision Tree | `class_weight="balanced"` |
| Random Forest | `class_weight="balanced"` |
| XGBoost | `scale_pos_weight = negatives / positives`, computed on the training set only (≈ 1.82) |

Consequence: with weighting, predicted probabilities are **not calibrated** (they overestimate the cancellation
probability). They are useful to *rank* orders, not to read as literal probabilities. Likewise the 0.5 decision
threshold was not tuned (see *Limitations*).

## 6. Models and hyperparameters

Hyperparameters are the ones from the original scripts. **They were not tuned**; the goal is a fair, reproducible
comparison, not the best possible score.

| Model | Hyperparameters |
|---|---|
| Decision Tree | `max_depth=5`, `min_samples_leaf=10` |
| Random Forest | `n_estimators=300`, `max_depth=8`, `min_samples_leaf=5` |
| XGBoost | `n_estimators=400`, `max_depth=6`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=5` |

Changes with respect to the originals:

* The Decision Tree script originally had no class weighting (only the baseline tree inside the Random Forest script
  did). It now uses `class_weight="balanced"` like the other tree models.
* XGBoost: the **test set is no longer passed as `eval_set`** to `fit()`. Without early stopping it did not influence
  the model, but evaluating on the test set during training is a bad habit and would become real leakage as soon as
  early stopping were enabled. If early stopping is wanted in the future, carve a validation set out of the training
  data.
* XGBoost: `eval_metric="auc"` was removed together with `eval_set` (it only had an effect with an evaluation set).

### XGBoost: library compatibility and categorical variables

* **Versions.** With scikit-learn 1.6.x, XGBoost 2.1.3 fails at *prediction* time inside a `Pipeline`
  (`AttributeError: 'super' object has no attribute '__sklearn_tags__'`; `fit` works, which makes the problem easy to
  miss). XGBoost 2.1.4 works with the same scikit-learn. This was reproduced while preparing the project, and is why
  `requirements.txt` asks for `xgboost>=2.1.4`. Two stacks were tested end to end: pandas 3.0.6 / numpy 2.5.4 /
  scikit-learn 1.9.1 / xgboost 3.4.1 (used for the published results) and pandas 2.2.3 / numpy 2.0.2 /
  scikit-learn 1.6.1 / xgboost 2.1.4 / matplotlib 3.9.4 (oldest versions declared).
* **Categoricals.** XGBoost has experimental native categorical support (`enable_categorical=True`). It was **not**
  used: the three models receive the same One-Hot-encoded matrix, so that differences in results come from the
  algorithm and not from different encodings. Trying native categoricals is listed under *Next steps*.

## 7. Evaluation metrics

All computed on the common test set (`common.evaluate`); nothing is hard-coded.

| Metric | Reading in this problem |
|---|---|
| Accuracy | share of orders classified correctly |
| Precision | of the orders flagged as "will cancel", how many really were cancelled |
| Recall | of the orders that were really cancelled, how many were caught |
| F1-score | harmonic mean of precision and recall |
| ROC-AUC | how well the probability ranks cancelled above non-cancelled orders (threshold-independent) |
| Confusion matrix | TN, FP, FN, TP counts |

Which metric matters most is a **business decision**: if contacting a customer is cheap and a cancellation is
expensive, favour recall; if each intervention is costly, favour precision.

## 8. Reproducibility

* Fixed `random_state=42` for the split, the three models and the sample generation.
* `metrics.json` stores hyperparameters and library versions of the run that produced it.
* Exact numbers can still vary slightly across library versions and platforms. For example, the sample run of the
  Random Forest gave an accuracy of 0.698 with the newest tested stack and 0.695 with the oldest one.

## 9. Limitations

* **Feature availability (possible target leakage by timing).** `abriu_chamado_suporte`,
  `tempo_aprovacao_pagamento_horas`, `estoque_disponivel_no_ato` and `taxa_cancelamento_historica_cliente` are among
  the most important features. Whether they are known at the moment the prediction would be used cannot be decided from
  the data. If any of them is recorded *after* the cancellation, the real-world performance would be lower than the test
  metrics suggest. This must be confirmed with whoever owns the data.
* **Simulated data.** The data appears to be synthetic, so conclusions about "what drives cancellations" are about the
  data-generating process, not about a real business.
* **Random split.** No out-of-time validation; the performance on future periods is unknown.
* **No hyperparameter search, no threshold tuning, no probability calibration, no cross-validation.** Differences
  between models were not tested for statistical significance; with a 20,000-order test set, small differences (e.g.
  between the F1-scores of Random Forest and XGBoost) should not be over-interpreted.
* **Feature importance is not causal.** One-Hot encoding splits one categorical variable into several columns, and
  impurity-based importances are biased towards high-cardinality / continuous features; strongly correlated features
  (e.g. `prazo_entrega_dias` and `distancia_km`, r ≈ 0.91 in the sample) share importance.

## 10. Possible next steps

* Cross-validation and hyperparameter search (always inside the training data).
* Decision-threshold selection based on a cost matrix; probability calibration.
* Permutation importance / SHAP for more reliable explanations.
* XGBoost with native categorical support; comparison with Logistic Regression as a simple baseline.
* Time-based validation, if real dated data becomes available.
