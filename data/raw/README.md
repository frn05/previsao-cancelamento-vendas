# data/raw — full datasets (not versioned)

The full CSV files are **not** published in this repository (the labelled file alone is ~13 MB and the
provenance/redistribution rights of the original files are not documented).

To run the models on the full data, place these two files in **this folder**:

| File | Rows | Has `cancelado`? | Purpose |
|---|---|---|---|
| `pedidos_completo_100k.csv` | 100,000 | yes | split 80/20 into train and test |
| `pedidos_novos_50k.csv` | 50,000 | no | new orders to score |

Both must have the same columns as the samples in `data/sample/` (the new file simply has no `cancelado` column).

If this folder has no CSVs, every script automatically falls back to `data/sample/` and prints a warning.
You can also keep the files elsewhere and pass `--data-dir /path/to/folder`.

Everything in this folder except this README is ignored by Git (see `.gitignore`).
