"""Roda os experimentos da H1/H3 e salva em resultados/. Uso: PYTHONHASHSEED=0 .venv/bin/python scripts/run_h3.py [temporal] [cv]"""

import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from noshow.evaluation import patient_groups, repeated_group_folds
from noshow.experiments import run_config
from noshow.features import CATEGORICAL, CONTINUOUS
from noshow.pipeline import raw_matrix

warnings.filterwarnings("ignore")
if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("Defina PYTHONHASHSEED=0 para resultados reprodutíveis.")

RAIZ = Path(__file__).resolve().parents[1]
SAIDA = RAIZ / "resultados"
SAIDA.mkdir(exist_ok=True)

df = pd.read_parquet(RAIZ / "data" / "processed" / "features.parquet")
raw = pd.read_csv(RAIZ / "data" / "raw" / "medical-appointments-no-show-en.csv")
assert (pd.to_datetime(raw.appointment_date, format="%d/%m/%Y").to_numpy() == df.appointment_date.to_numpy()).all()
y = df.no_show.to_numpy()
X_raw, RAW_CAT, RAW_CONT = raw_matrix(raw)
X_eng = df[CATEGORICAL + CONTINUOUS]

CONFIGS = [  # (nome, base, escalonador, smotenc)
    ("C0 bruto", "raw", "sem", False),
    ("C0 bruto + SMOTENC (robust)", "raw", "robust", True),
    ("C1 engenharia", "eng", "sem", False),
    ("C3 engenharia + SMOTENC (sem escalonador)", "eng", "sem", True),
    ("C3 engenharia + SMOTENC (minmax)", "eng", "minmax", True),
    ("C3 engenharia + SMOTENC (standard)", "eng", "standard", True),
    ("C3 engenharia + SMOTENC (robust)", "eng", "robust", True),
]
BASES = {"raw": (X_raw, RAW_CAT, RAW_CONT), "eng": (X_eng, CATEGORICAL, CONTINUOUS)}


def validacao_cruzada():
    linhas, t0 = [], time.time()
    for r, f, tr, te in repeated_group_folds(y, patient_groups(df.patient_id), n_splits=5, n_repeats=3):
        for nome, base, scaler, smote in CONFIGS:
            X, cat, cont = BASES[base]
            linha = run_config(nome, X.iloc[tr], y[tr], X.iloc[te], y[te], cat, cont, scaler, smote, seed=42 + r)
            linhas.append({"rep": r, "fold": f, **linha})
        print(f"rep {r} fold {f}: {time.time() - t0:.0f}s", flush=True)
        pd.DataFrame(linhas).to_csv(SAIDA / "h3_validacao_cruzada.csv", index=False)


def validacao_temporal():
    """Treino até 2020, teste 2021–2022. Variáveis de ano/período ficam de fora: o futuro nunca tem um
    valor visto no treino, então elas não podem generalizar."""
    data = df.appointment_date
    tr, te = np.where(data.dt.year <= 2020)[0], np.where(data.dt.year >= 2021)[0]
    deriva = {"appointment_year", "periodo"}
    linhas = []
    for criterio in ["aic-d", "bic-d"]:  # o AIC venceu na CV aleatória; o BIC testa se a parcimônia generaliza
        for nome, base, scaler, smote in CONFIGS:
            X, cat, cont = BASES[base]
            cat = [c for c in cat if c not in deriva]
            linha = run_config(nome, X.iloc[tr], y[tr], X.iloc[te], y[te], cat, cont, scaler, smote, scoring=criterio)
            linhas.append({"criterio": criterio.split("-")[0].upper(), **linha})
            print(criterio, nome, f"{linha['roc_auc']:.3f}", flush=True)
    pd.DataFrame(linhas).to_csv(SAIDA / "h3_validacao_temporal.csv", index=False)


if __name__ == "__main__":
    etapas = sys.argv[1:] or ["temporal", "cv"]
    if "temporal" in etapas:
        validacao_temporal()
    if "cv" in etapas:
        validacao_cruzada()
