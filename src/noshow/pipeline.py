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
from noshow.weather import heat_category, rain_category

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
                     seed: int = 42, repair_derived: bool = False,
                     coherent_history: bool = False) -> tuple[pd.DataFrame, np.ndarray]:
    """Reamostra o treino com SMOTENC e devolve as variáveis no formato de entrada (com NaN restaurados)."""
    original = X[categorical + continuous].copy()
    history_cols = {"n_faltas_anteriores", "n_agendamentos_anteriores", "taxa_faltas_anteriores"}
    derive_history = coherent_history and history_cols.issubset(continuous)
    output_continuous = continuous
    if derive_history:
        # A razão não participa da distância nem recebe uma interpolação independente.
        continuous = [c for c in continuous if c != "taxa_faltas_anteriores"]
    X = original[categorical + continuous].copy()
    flags = [f"__ausente_{c}" for c in continuous]
    for c, f in zip(continuous, flags):
        X[f] = X[c].isna().astype(int).astype(str)
    # Uma coluna inteiramente ausente no treino não deve impedir a reamostragem.
    # O zero é só operacional: o indicador restaura NaN após a interpolação.
    medianas = X[continuous].median().fillna(0.0)
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
    if repair_derived:
        # Só os sintéticos: os registros reais, inclusive suas ausências, são preservados.
        idx = Xr.index[len(X):]
        for numeric, category, fn in [("max_temp_day", "heat_intensity", heat_category),
                                       ("max_rain_day", "rain_intensity", rain_category)]:
            if numeric in continuous and category in categorical:
                Xr.loc[idx, category] = fn(Xr.loc[idx, numeric]).fillna("ausente").to_numpy()
        if "idade_consulta" in continuous and "faixa_etaria" in categorical:
            ages = pd.cut(Xr.loc[idx, "idade_consulta"], [0, 6, 12, 18, 60, np.inf], right=False,
                          labels=["0-5", "6-11", "12-17", "18-59", "60+"])
            Xr.loc[idx, "faixa_etaria"] = ages.astype("string").fillna("ausente").to_numpy()
    if derive_history:
        idx = Xr.index[len(X):]
        parents = ["n_agendamentos_anteriores", "n_faltas_anteriores"]
        counts = Xr.loc[idx, parents].clip(lower=0)
        # A ausência de qualquer contagem impede construir uma razão conhecida.
        counts.loc[counts.isna().any(axis=1), parents] = np.nan
        counts["n_faltas_anteriores"] = np.minimum(counts.n_faltas_anteriores, counts.n_agendamentos_anteriores)
        Xr.loc[idx, parents] = counts
        Xr["taxa_faltas_anteriores"] = np.nan
        Xr.loc[idx, "taxa_faltas_anteriores"] = (
            counts.n_faltas_anteriores / counts.n_agendamentos_anteriores.where(counts.n_agendamentos_anteriores > 0))
        # As contagens sintéticas são interpolações contínuas; não são eventos observados.
        Xr.loc[Xr.index[:len(X)], "taxa_faltas_anteriores"] = original.taxa_faltas_anteriores.to_numpy()
    return Xr[categorical + output_continuous], np.asarray(yr)


def restore_class_prior(p, original_rate: float, sampled_rate: float):
    """Corrige as odds pela alteração da proporção de classes no treino.

    Preserva a ordenação; não promete aumentar PR-AUC. A hipótese de manutenção
    das distribuições por classe é aproximada no SMOTENC, devendo-se medir Brier.
    """
    if not 0 < original_rate < 1 or not 0 < sampled_rate < 1:
        raise ValueError("As duas prevalências precisam estar estritamente entre 0 e 1.")
    a = np.asarray(p) * original_rate / sampled_rate
    b = (1 - np.asarray(p)) * (1 - original_rate) / (1 - sampled_rate)
    return a / (a + b)


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


def classification_report_budget(y_true, p, fraction: float = 0.10) -> dict[str, float]:
    """Recall/precisão esperados para k=ceil(fração*n) alertas de maior risco.

    Empates no ponto de corte recebem seleção uniforme: usa-se a esperança do
    número de acertos, sem favorecer a ordem das linhas nem desempatar pelo alvo.
    O orçamento é fixado antes de examinar o teste.
    """
    y, p = np.asarray(y_true), np.asarray(p)
    if not 0 < fraction <= 1 or not len(y) or len(y) != len(p) or not np.isfinite(p).all():
        raise ValueError("Orçamento, rótulos e probabilidades inválidos.")
    k = max(1, int(np.ceil(fraction * len(y))))
    cutoff = np.partition(p, len(p) - k)[len(p) - k]
    above, tied = p > cutoff, p == cutoff
    remaining = k - int(above.sum())
    tp = float(y[above].sum() + remaining * y[tied].mean())
    return {"recall": tp / y.sum() if y.sum() else np.nan, "precision": tp / k,
            "pct_alertas": k / len(y), "n_alertas": k, "empates_no_corte": int(tied.sum())}
