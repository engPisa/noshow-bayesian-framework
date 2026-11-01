"""Engenharia de variáveis para a predição de no-show.

Princípios:
- Só entra informação disponível no momento do agendamento (nada do próprio desfecho ou de datas futuras).
- O histórico do paciente usa apenas dias anteriores: faltas no mesmo dia são simultâneas e 93% dos dias com
  várias consultas têm desfecho homogêneo, então usá-las vazaria o alvo.
- Transformações deterministas ficam aqui; tudo que aprende parâmetros com os dados (cortes de discretização,
  escalonadores, SMOTENC) é ajustado só no fold de treino, via `Discretizer` e o pipeline do notebook 04.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from noshow.data import TARGET, add_patient_proxy, load_raw
from noshow.weather import fix_weather

MISSING = "ausente"

SPECIALTY_MAIN = ["speech therapy", "psychotherapy", "physiotherapy", "occupational therapy", "pedagogo"]
CID_CHAPTERS = {"F": "F_mental", "R": "R_sintomas", "I": "I_circulatorio", "G": "G_nervoso", "Q": "Q_congenito"}
WEEKDAYS = ["seg", "ter", "qua", "qui", "sex"]
SEASONS = {12: "verao", 1: "verao", 2: "verao", 3: "outono", 4: "outono", 5: "outono",
           6: "inverno", 7: "inverno", 8: "inverno", 9: "primavera", 10: "primavera", 11: "primavera"}

# Especificação das variáveis candidatas ao modelo: nome -> (tipo, grupo de fatores da Seção 2.1.4).
FEATURES = {
    # demográficas
    "faixa_etaria": ("categorical", "demográfico"),
    "idade_consulta": ("continuous", "demográfico"),
    "gender": ("categorical", "demográfico"),
    "disability": ("categorical", "demográfico"),
    "city_grp": ("categorical", "demográfico"),
    "cid_capitulo": ("categorical", "demográfico"),
    "cadastro_ausente": ("categorical", "demográfico"),
    # contextuais
    "specialty_grp": ("categorical", "contextual"),
    "patient_needs_companion": ("categorical", "contextual"),
    "consultas_no_dia": ("continuous", "contextual"),
    "heat_intensity": ("categorical", "contextual (clima)"),
    "rain_intensity": ("categorical", "contextual (clima)"),
    "max_temp_day": ("continuous", "contextual (clima)"),
    "max_rain_day": ("continuous", "contextual (clima)"),
    # temporais
    "dia_semana": ("categorical", "temporal"),
    "faixa_horaria": ("categorical", "temporal"),
    "mes": ("categorical", "temporal"),
    "estacao": ("categorical", "temporal"),
    "periodo": ("categorical", "temporal"),
    # comportamentais
    "dias_em_servico": ("continuous", "comportamental"),
    "n_agendamentos_anteriores": ("continuous", "comportamental"),
    "n_faltas_anteriores": ("continuous", "comportamental"),
    "taxa_faltas_anteriores": ("continuous", "comportamental"),
    "dias_desde_ultima_consulta": ("continuous", "comportamental"),
    "faltou_ultima": ("categorical", "comportamental"),
}
CATEGORICAL = [f for f, (t, _) in FEATURES.items() if t == "categorical"]
CONTINUOUS = [f for f, (t, _) in FEATURES.items() if t == "continuous"]


def add_age(df: pd.DataFrame) -> pd.DataFrame:
    """Idade na data da consulta (a coluna `age` original é a idade na extração da base, ~2022)."""
    df = df.copy()
    idade = (df["appointment_date"] - df["date_of_birth"]).dt.days / 365.25
    df["idade_consulta"] = idade.where(idade.between(0, 100)).round(1)  # ≥100 anos: data de nascimento inválida
    faixas = pd.cut(df["idade_consulta"], [0, 6, 12, 18, 60, np.inf], right=False,
                    labels=["0-5", "6-11", "12-17", "18-59", "60+"])
    df["faixa_etaria"] = faixas.astype("string").fillna(MISSING)
    return df.drop(columns=["age", "under_12_years_old", "over_60_years_old"], errors="ignore")


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    date = df["appointment_date"]
    dow = date.dt.dayofweek
    df["dia_semana"] = np.where(dow >= 5, "fim_de_semana", dow.clip(upper=4).map(dict(enumerate(WEEKDAYS))))
    hora = df["appointment_time"].str[:2].astype(int)
    # 17h e 18h ficam separadas: são o horário com mais e o com menos faltas
    df["faixa_horaria"] = pd.cut(hora, [0, 9, 12, 15, 17, 18, 24], right=False,
                                 labels=["07-08h", "09-11h", "12-14h", "15-16h", "17h", "18h"]).astype("string")
    df["mes"] = date.dt.month.astype(str).str.zfill(2)
    df["estacao"] = date.dt.month.map(SEASONS)
    df["periodo"] = pd.cut(date.dt.year, [0, 2019, 2021, np.inf], labels=["pre_pandemia", "pandemia", "pos_2022"]
                           ).astype("string")
    return df


def group_categories(df: pd.DataFrame, min_city: int = 300) -> pd.DataFrame:
    """Agrupa categorias raras e torna a ausência uma categoria explícita (faltantes são informativos)."""
    df = df.copy()
    cadastro = ["date_of_birth", "entry_service_date", "city", "disability"]  # antes de preencher as ausências
    df["cadastro_ausente"] = df[cadastro].isna().all(axis=1).astype(int).astype(str)

    esp = df["specialty"].where(df["specialty"].isin(SPECIALTY_MAIN) | df["specialty"].isna(), "outras")
    df["specialty_grp"] = esp.fillna(MISSING)

    counts = df["city"].value_counts()
    df["city_grp"] = df["city"].where(df["city"].isin(counts[counts >= min_city].index) | df["city"].isna(),
                                      "outras").fillna(MISSING)

    letra = df["icd"].str.upper().str.extract(r"([A-Z])", expand=False)
    df["cid_capitulo"] = letra.map(CID_CHAPTERS).where(letra.isna() | letra.isin(list(CID_CHAPTERS)), "outros"
                                                       ).fillna(MISSING)

    df["gender"] = df["gender"].where(df["gender"].isin(["M", "F"]), MISSING)
    df["disability"] = df["disability"].fillna(MISSING)
    for col in ["heat_intensity", "rain_intensity"]:
        df[col] = df[col].fillna(MISSING)
    df["patient_needs_companion"] = df["patient_needs_companion"].astype(str)
    return df


def add_service_time(df: pd.DataFrame) -> pd.DataFrame:
    """Dias desde a entrada no serviço; datas de entrada posteriores à consulta são tratadas como ausentes."""
    df = df.copy()
    dias = (df["appointment_date"] - df["entry_service_date"]).dt.days
    df["dias_em_servico"] = dias.where(dias >= 0)
    return df


def add_patient_history(df: pd.DataFrame) -> pd.DataFrame:
    """Histórico de comparecimento usando apenas dias anteriores ao da consulta.

    Exige `patient_id` (NaN quando o paciente não pode ser identificado: histórico fica ausente).
    """
    df = df.copy()
    known = df["patient_id"].notna()

    dias = (df[known].groupby(["patient_id", "appointment_date"])[TARGET]
            .agg(consultas_no_dia="size", faltas_no_dia="sum").reset_index()
            .sort_values(["patient_id", "appointment_date"]))
    g = dias.groupby("patient_id")
    # cumsum inclui o próprio dia; subtraí-lo deixa só o passado
    dias["n_agendamentos_anteriores"] = g["consultas_no_dia"].cumsum() - dias["consultas_no_dia"]
    dias["n_faltas_anteriores"] = g["faltas_no_dia"].cumsum() - dias["faltas_no_dia"]
    dias["dias_desde_ultima_consulta"] = g["appointment_date"].diff().dt.days
    dias["faltou_ultima"] = (g["faltas_no_dia"].shift() > 0).map({True: "sim", False: "nao"})
    dias.loc[dias["n_agendamentos_anteriores"] == 0, "faltou_ultima"] = "sem_historico"

    cols = ["consultas_no_dia", "n_agendamentos_anteriores", "n_faltas_anteriores",
            "dias_desde_ultima_consulta", "faltou_ultima"]
    df = df.drop(columns=cols, errors="ignore").merge(
        dias[["patient_id", "appointment_date"] + cols], on=["patient_id", "appointment_date"], how="left")

    prev = df["n_agendamentos_anteriores"]
    df["taxa_faltas_anteriores"] = (df["n_faltas_anteriores"] / prev).where(prev > 0)
    df["faltou_ultima"] = df["faltou_ultima"].fillna(MISSING)  # sem ID: histórico desconhecido
    return df


def build_features(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Base completa: carga, correção do clima, ID aproximado e todas as variáveis de FEATURES."""
    df = add_patient_proxy(fix_weather(load_raw())) if df is None else df
    df = add_age(df)
    df = add_calendar(df)
    df = group_categories(df)
    df = add_service_time(df)
    df = add_patient_history(df)
    return df


class Discretizer(BaseEstimator, TransformerMixin):
    """Discretiza variáveis contínuas por quantis aprendidos no treino; NaN vira a categoria `ausente`.

    Cortes repetidos (variáveis com muitos zeros, como contagens) são unificados, então uma variável pode
    terminar com menos de `n_bins` faixas. Colunas não listadas em `columns` passam inalteradas.
    """

    def __init__(self, columns=None, n_bins: int = 5):
        self.columns = columns
        self.n_bins = n_bins

    def fit(self, X: pd.DataFrame, y=None):
        cols = self.columns if self.columns is not None else list(X.columns)
        qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        self.edges_ = {c: np.unique(X[c].dropna().quantile(qs).to_numpy()) for c in cols}
        self.labels_ = {}
        for c, edges in self.edges_.items():
            bins = np.concatenate([[-np.inf], edges, [np.inf]])
            self.labels_[c] = [self._label(lo, hi) for lo, hi in zip(bins[:-1], bins[1:])] + [MISSING]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, edges in self.edges_.items():
            bins = np.concatenate([[-np.inf], edges, [np.inf]])
            X[col] = pd.cut(X[col], bins, labels=self.labels_[col][:-1]).astype("string").fillna(MISSING)
        return X

    @staticmethod
    def _label(lo, hi):
        fmt = lambda v: f"{v:.3g}"
        if np.isinf(lo):
            return f"<={fmt(hi)}"
        if np.isinf(hi):
            return f">{fmt(lo)}"
        return f"({fmt(lo)}, {fmt(hi)}]"
