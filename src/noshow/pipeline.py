"""Framework de normalização e balanceamento: escalonamento + SMOTENC antes da Rede Bayesiana.

Ordem (Seção 3.4, item 2): imputação → escalonamento → SMOTENC → volta à escala original → discretização.
Tudo é ajustado só no treino; o teste nunca é reamostrado.

- O SMOTENC não aceita ausentes: contínuas são imputadas pela mediana do treino e ganham um indicador de
  ausência categórico. Os pontos sintéticos herdam o indicador por moda dos vizinhos, e onde ele vale 1 o
  valor volta a ser NaN (estado "ausente" na rede).
- O escalonador só afeta a distância usada pelo SMOTENC para escolher vizinhos; depois a escala original é
  restaurada para que as faixas da discretização continuem legíveis (escalonadores lineares não mudam faixas
  de quantis).
"""

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from noshow.data import LEAKAGE_COLS, TARGET

SCALERS = {"minmax": MinMaxScaler, "standard": StandardScaler, "robust": RobustScaler}

# Controle "dados brutos": colunas originais sem limpeza nem engenharia (só sai o que é vazamento ou data crua).
RAW_CONTINUOUS = ["age", "average_temp_day", "max_temp_day", "average_rain_day", "max_rain_day"]
RAW_EXCLUDE = LEAKAGE_COLS + [TARGET, "appointment_date", "date_of_birth", "entry_service_date"]


def raw_matrix(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Base original como chega: categóricas viram texto (NaN vira o estado 'nan'), contínuas ficam numéricas."""
    cols = [c for c in raw.columns if c not in RAW_EXCLUDE]
    categoricas = [c for c in cols if c not in RAW_CONTINUOUS]
    X = raw[cols].copy()
    for c in categoricas:  # no pandas 3, astype(str) mantém NaN: o estado "nan" precisa ser explícito
        X[c] = X[c].astype("string").fillna("nan").astype(str)
    return X, categoricas, list(RAW_CONTINUOUS)


def smotenc_resample(X: pd.DataFrame, y: np.ndarray, categorical: list[str], continuous: list[str],
                     scaler: str | None = "robust", k_neighbors: int = 5, sampling_strategy=1.0,
                     seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    """Reamostra o treino com SMOTENC e devolve as variáveis no formato de entrada (com NaN restaurados)."""
    X = X[categorical + continuous].copy()
    flags = [f"__ausente_{c}" for c in continuous]
    for c, f in zip(continuous, flags):
        X[f] = X[c].isna().astype(int).astype(str)
    medianas = X[continuous].median()
    num = X[continuous].fillna(medianas).to_numpy(dtype=float)

    sc = SCALERS[scaler]().fit(num) if scaler else None
    X[continuous] = sc.transform(num) if sc else num
    X[categorical] = X[categorical].astype(str)

    cat_cols = categorical + flags
    smote = SMOTENC(categorical_features=[X.columns.get_loc(c) for c in cat_cols],
                    k_neighbors=k_neighbors, sampling_strategy=sampling_strategy, random_state=seed)
    Xr, yr = smote.fit_resample(X, y)
    Xr = pd.DataFrame(Xr, columns=X.columns)

    valores = Xr[continuous].to_numpy(dtype=float)
    Xr[continuous] = sc.inverse_transform(valores) if sc else valores
    for c, f in zip(continuous, flags):
        Xr.loc[Xr[f] == "1", c] = np.nan
    Xr[continuous] = Xr[continuous].astype(float)
    return Xr.drop(columns=flags), np.asarray(yr)


def classification_report_at(y_true, p, threshold: float) -> dict[str, float]:
    """Métricas de matriz de confusão para um limiar de decisão."""
    pred = p >= threshold
    tp = int((pred & (y_true == 1)).sum()); fp = int((pred & (y_true == 0)).sum())
    fn = int((~pred & (y_true == 1)).sum()); tn = int((~pred & (y_true == 0)).sum())
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    especificidade = tn / (tn + fp) if tn + fp else 0.0
    return {"accuracy": (tp + tn) / len(y_true), "recall": recall, "precision": precision,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "balanced_accuracy": (recall + especificidade) / 2, "pct_alertas": float(pred.mean())}
