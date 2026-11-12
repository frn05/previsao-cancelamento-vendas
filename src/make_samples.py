"""
Gera as amostras versionadas em data/sample/ a partir da base completa.

Uso (com os CSVs completos em data/raw/):
    python src/make_samples.py

- pedidos_completo_sample.csv: 5.000 linhas, amostra ESTRATIFICADA pelo alvo
  'cancelado' (mantém a proporção de cancelamentos da base completa).
- pedidos_novos_sample.csv: 1.000 linhas aleatórias (base sem rótulo).
Os IDs originais são mantidos. Semente fixa => resultado reproduzível.
"""

import pandas as pd

from common import PROJECT_ROOT, RANDOM_STATE, TARGET

RAW = PROJECT_ROOT / "data" / "raw"
SAMPLE = PROJECT_ROOT / "data" / "sample"
N_LABELED, N_NEW = 5000, 1000


def main():
    full = pd.read_csv(RAW / "pedidos_completo_100k.csv")
    new = pd.read_csv(RAW / "pedidos_novos_50k.csv")

    # amostragem estratificada: mesma fração em cada classe do alvo
    frac = N_LABELED / len(full)
    sample = full.groupby(TARGET, group_keys=False).sample(frac=frac, random_state=RANDOM_STATE)
    sample = sample.sort_values("id_pedido")
    sample.to_csv(SAMPLE / "pedidos_completo_sample.csv", index=False)

    new.sample(n=N_NEW, random_state=RANDOM_STATE).sort_values("id_pedido").to_csv(
        SAMPLE / "pedidos_novos_sample.csv", index=False
    )
    print(f"Amostra rotulada: {len(sample)} linhas | taxa de cancelamento: "
          f"{sample[TARGET].mean():.4f} (base completa: {full[TARGET].mean():.4f})")
    print(f"Amostra nova: {N_NEW} linhas")


if __name__ == "__main__":
    main()
