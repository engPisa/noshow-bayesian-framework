"""Carga e limpeza mínima da base de agendamentos (sem engenharia de variáveis)."""

from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "medical-appointments-no-show-en.csv"

TARGET = "no_show"

# Só é preenchida depois que a falta acontece: usar como preditor é vazamento.
LEAKAGE_COLS = ["no_show_reason"]

# Atributos estáveis do paciente usados para reconstruir um identificador (a base não tem ID).
PATIENT_KEY = ["date_of_birth", "gender", "city", "entry_service_date", "disability"]

DATE_COLS = ["appointment_date", "date_of_birth", "entry_service_date"]

MONTH_ORDER = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sept", "oct", "nov", "dec"]


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Lê o CSV original e aplica apenas correções de tipo e de formatação."""
    df = pd.read_csv(path)

    str_cols = df.select_dtypes(include=["object", "string"]).columns
    df[str_cols] = df[str_cols].apply(lambda s: s.str.strip()).replace("", np.nan)

    for col in DATE_COLS:
        df[col] = pd.to_datetime(df[col], format="%d/%m/%Y", errors="coerce")

    df[TARGET] = (df[TARGET] == "yes").astype(int)
    df["appointment_month"] = pd.Categorical(df["appointment_month"], categories=MONTH_ORDER, ordered=True)
    return df


def add_patient_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """Cria `patient_id` a partir de PATIENT_KEY; linhas sem a chave completa ficam com NaN."""
    df = df.copy()
    complete = df[PATIENT_KEY].notna().all(axis=1)
    df["patient_id"] = np.nan
    df.loc[complete, "patient_id"] = df.loc[complete].groupby(PATIENT_KEY, observed=True).ngroup()
    df["patient_id"] = df["patient_id"].astype("Int64")
    return df
