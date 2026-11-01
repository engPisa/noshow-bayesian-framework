"""Correção e preenchimento das variáveis de clima com o histórico do Open-Meteo (reanálise ERA5).

Na base original, os dias sem medição meteorológica receberam valores de preenchimento
(`heat_intensity = "heavy_cold"`, `rain_intensity = "no_rain"`) e as colunas `rainy_day_before`
e `storm_day_before` só variam nesses dias. Validação contra o Open-Meteo (969 dias medidos):
temperatura média r = 0,977 e máxima r = 0,939; chuva horária r = 0,45–0,60, insuficiente para imputar.
Por isso, apenas a temperatura dos dias ausentes é preenchida; a chuva permanece ausente.
"""

import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

OPENMETEO_PATH = Path(__file__).resolve().parents[2] / "data" / "external" / "openmeteo_itajai_diario.csv"

# Centro de Itajaí/SC, município com 41% dos agendamentos.
LATITUDE, LONGITUDE = -26.9078, -48.6619
TIMEZONE = "America/Sao_Paulo"

# Regras reproduzidas a partir da base (100% de concordância nos dias medidos).
HEAT_BINS = [-np.inf, 15, 22, 27, 32, np.inf]  # sobre max_temp_day (°C)
HEAT_LABELS = ["heavy_cold", "cold", "mild", "warm", "heavy_warm"]
RAIN_BINS = [-np.inf, 1, 2.5, 10, np.inf]  # sobre max_rain_day (mm/h, maior chuva horária do dia)
RAIN_LABELS = ["no_rain", "weak", "moderate", "heavy"]

WEATHER_ARTIFACT_COLS = ["rainy_day_before", "storm_day_before"]


def heat_category(max_temp: pd.Series) -> pd.Series:
    return pd.cut(max_temp, HEAT_BINS, labels=HEAT_LABELS).astype("string")


def rain_category(max_rain: pd.Series) -> pd.Series:
    return pd.cut(max_rain, RAIN_BINS, labels=RAIN_LABELS).astype("string")


def download_openmeteo(start: str = "2016-06-01", end: str = "2022-02-28", path: Path = OPENMETEO_PATH) -> pd.DataFrame:
    """Baixa o histórico horário e agrega por dia no mesmo formato das colunas da base."""
    url = ("https://archive-api.open-meteo.com/v1/archive"
           f"?latitude={LATITUDE}&longitude={LONGITUDE}&start_date={start}&end_date={end}"
           f"&hourly=temperature_2m,precipitation&timezone={TIMEZONE.replace('/', '%2F')}")
    with urlopen(url, timeout=120) as resp:
        hourly = pd.DataFrame(json.load(resp)["hourly"])

    hourly["date"] = pd.to_datetime(hourly["time"]).dt.normalize()
    daily = hourly.groupby("date").agg(
        average_temp_day=("temperature_2m", "mean"),
        max_temp_day=("temperature_2m", "max"),
        average_rain_day=("precipitation", "mean"),
        max_rain_day=("precipitation", "max"),
        total_rain_day=("precipitation", "sum"),
    ).round(2)

    path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(path)
    return daily


def load_openmeteo(path: Path = OPENMETEO_PATH) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"], index_col="date")


def fix_weather(df: pd.DataFrame, openmeteo: pd.DataFrame | None = None) -> pd.DataFrame:
    """Remove os valores de preenchimento e completa a temperatura dos dias sem medição.

    Espera `appointment_date` já convertida para datetime. Adiciona `clima_fonte` ("base" | "open-meteo").
    """
    om = load_openmeteo() if openmeteo is None else openmeteo
    df = df.drop(columns=WEATHER_ARTIFACT_COLS, errors="ignore").copy()

    missing = df["average_temp_day"].isna()
    df.loc[missing, ["rain_intensity", "heat_intensity"]] = np.nan

    dates = df.loc[missing, "appointment_date"]
    for col in ["average_temp_day", "max_temp_day"]:
        df.loc[missing, col] = dates.map(om[col]).to_numpy()
    filled = missing & df["max_temp_day"].notna()
    df.loc[filled, "heat_intensity"] = heat_category(df.loc[filled, "max_temp_day"])

    df["clima_fonte"] = np.where(filled, "open-meteo", np.where(missing, pd.NA, "base"))
    return df
