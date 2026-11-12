# data/sample — versioned samples

| File | Rows | Description |
|---|---|---|
| `pedidos_completo_sample.csv` | 5,000 | Stratified random sample (by `cancelado`) of `pedidos_completo_100k.csv`. Cancellation rate 35.50%, identical to the full file. |
| `pedidos_novos_sample.csv` | 1,000 | Random sample of `pedidos_novos_50k.csv` (no target column). |
| `vendas/vendas_treino.csv` | 800 | Original small dataset (train), used by the first versions of the Decision Tree / Random Forest. |
| `vendas/vendas_teste.csv` | 200 | Original small dataset (test). |
| `vendas/vendas_novas.csv` | 50 | Original small dataset (new sales, no target). |

The `pedidos` samples were produced by `python src/make_samples.py` (fixed seed 42), so they can be regenerated from the full files.
Original order/sale IDs are kept.

## About the origin of the data

The original files do not document where the data comes from. Several characteristics point to **simulated data**
(sequential IDs, rounded values, and almost perfectly uniform counts per product category — between 9,800 and 10,113
orders in each of the 10 categories). None of the files contains personal identifiers (no names, CPF, e-mail, phone
or address). Treat them as educational/demonstration data: **results obtained here say nothing about any real business.**
