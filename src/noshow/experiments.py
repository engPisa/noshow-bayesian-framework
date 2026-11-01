"""Experimentos controlados da H1/H3: dados brutos × framework (engenharia, escalonamento, SMOTENC)."""

import time

import numpy as np
import pandas as pd

from noshow.bayes import DiscreteBN, learn_structure, markov_blanket
from noshow.data import TARGET
from noshow.evaluation import metrics
from noshow.features import Discretizer
from noshow.pipeline import classification_report_at, smotenc_resample


def fit_predict(Xtr: pd.DataFrame, ytr: np.ndarray, Xte: pd.DataFrame, categorical: list[str],
                continuous: list[str], scoring: str = "aic-d") -> tuple[np.ndarray, list[str], int]:
    """Discretiza (cortes do treino), aprende a estrutura, reduz ao Manto e prevê o teste."""
    disc = Discretizer(columns=continuous).fit(Xtr)
    A = disc.transform(Xtr[categorical + continuous]).astype(str)
    A[TARGET] = np.asarray(ytr).astype(int).astype(str)
    B = disc.transform(Xte[categorical + continuous]).astype(str)
    # domínio das categóricas = valores possíveis da variável (não usa rótulos do teste)
    estados = {c: sorted(set(A[c]) | set(B[c])) for c in categorical}
    estados.update(disc.labels_)
    estados[TARGET] = ["0", "1"]

    grafo = learn_structure(A, scoring=scoring)
    mb = markov_blanket(grafo)
    bn = DiscreteBN(grafo.subgraph(mb + [TARGET]).copy()).fit(A[mb + [TARGET]], estados)
    return bn.predict_proba(B)[:, 1], mb, bn.n_params()


def run_config(nome: str, Xtr, ytr, Xte, yte, categorical, continuous, scaler="sem", smote=False,
               scoring="aic-d", seed=42) -> dict:
    """Executa uma configuração do framework e devolve métricas de ranking e de limiar."""
    t0 = time.time()
    n_treino = len(Xtr)
    prevalencia = float(np.mean(ytr))  # do treino original, antes de qualquer reamostragem
    if smote:
        Xtr, ytr = smotenc_resample(Xtr, ytr, categorical, continuous,
                                    scaler=None if scaler == "sem" else scaler, seed=seed)
    p, mb, n_params = fit_predict(Xtr, ytr, Xte, categorical, continuous, scoring)
    linha = {"config": nome, "escalonador": scaler, "smotenc": smote, "n_treino": n_treino,
             "n_treino_final": len(Xtr), "manto": ", ".join(mb), "n_params": n_params,
             "p_media": float(p.mean()), "segundos": time.time() - t0, **metrics(yte, p)}
    # limiar 0,5 (padrão de um classificador) e limiar = prevalência do treino (alerta acima do risco médio)
    for rotulo, limiar in [("050", 0.5), ("prev", prevalencia)]:
        linha.update({f"{k}@{rotulo}": v for k, v in classification_report_at(yte, p, limiar).items()})
    return linha
