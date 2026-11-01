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
RELAXED_KEY = ["gender", "city", "entry_service_date", "disability"]  # sem a data de nascimento

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


def add_patient_proxy(df: pd.DataFrame, relaxed: bool = False) -> pd.DataFrame:
    """Cria `patient_id` a partir de PATIENT_KEY; linhas sem a chave completa ficam com NaN.

    Com `relaxed=True`, linhas sem data de nascimento (93% dos casos sem ID que têm cadastro) são ligadas pela
    chave sem ela (RELAXED_KEY): ao paciente conhecido, se a chave aponta para um único paciente, ou a um
    paciente novo por chave. Essa chave colide em ~26% dos grupos com data de nascimento conhecida, então
    pode juntar pacientes diferentes; `id_qualidade` registra o nível de cada ligação.
    """
    df = df.copy()
    complete = df[PATIENT_KEY].notna().all(axis=1)
    df["patient_id"] = np.nan
    df.loc[complete, "patient_id"] = df.loc[complete].groupby(PATIENT_KEY, observed=True).ngroup()
    df["id_qualidade"] = np.where(complete, "exato", "ausente")

    if relaxed:
        conhecidos = df[complete].groupby(RELAXED_KEY, observed=True)["patient_id"].unique()
        unicos = {k: v[0] for k, v in conhecidos.items() if len(v) == 1}
        pendentes = ~complete & df[RELAXED_KEY].notna().all(axis=1)
        chaves = list(df.loc[pendentes, RELAXED_KEY].itertuples(index=False, name=None))
        ligado = pd.Series([unicos.get(k, np.nan) for k in chaves], index=df.index[pendentes])
        df.loc[ligado.index, "patient_id"] = ligado
        novos = ligado.index[ligado.isna()]
        df.loc[novos, "patient_id"] = (df.loc[novos].groupby(RELAXED_KEY, observed=True).ngroup()
                                       + df["patient_id"].max() + 1)
        df.loc[ligado.index, "id_qualidade"] = "aproximado"

    df["patient_id"] = df["patient_id"].astype("Int64")
    return df
