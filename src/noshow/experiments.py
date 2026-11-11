"""Experimentos controlados da H1/H3: dados brutos × framework (engenharia, escalonamento, SMOTENC)."""

import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from noshow.bayes import DiscreteBN, learn_structure, markov_blanket
from noshow.data import TARGET
from noshow.evaluation import metrics
from noshow.features import Discretizer
from noshow.pipeline import classification_report_at, classification_report_budget, restore_class_prior, smotenc_resample
from noshow.preprocessing import FoldCategories


def prepare_discrete(Xtr, Xte, categorical, continuous, n_bins=5, encoder=None):
    """Ajusta categorias no treino e cortes no treino recebido; nunca inspeciona o teste no fit."""
    encoder = FoldCategories(categorical).fit(Xtr) if encoder is None else encoder
    A, B = encoder.transform(Xtr), encoder.transform(Xte)
    disc = Discretizer(columns=continuous, n_bins=n_bins).fit(A)
    A = disc.transform(A[categorical + continuous]).astype(str)
    B = disc.transform(B[categorical + continuous]).astype(str)
    states = {**encoder.states_, **disc.labels_, TARGET: ["0", "1"]}
    return A, B, states


def _predict_candidates(Xtr, ytr, Xte, categorical, continuous, scoring, ess_values,
                        n_bins=5, scaler="sem", smote=False, sampling_strategy=1.0,
                        k_neighbors=5, repair_derived=False, coherent_history=False, correct_prior=False, seed=42):
    # O agrupamento e o domínio são os do treino REAL, antes de fabricar amostras.
    encoder = FoldCategories(categorical).fit(Xtr)
    Xtr = encoder.transform(Xtr)
    original_rate = float(np.mean(ytr))
    if smote:
        Xtr, ytr = smotenc_resample(
            Xtr, ytr, categorical, continuous, scaler=None if scaler == "sem" else scaler,
            sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, seed=seed,
            repair_derived=repair_derived, coherent_history=coherent_history)
    A, B, states = prepare_discrete(Xtr, Xte, categorical, continuous, n_bins=n_bins, encoder=encoder)
    A[TARGET] = np.asarray(ytr).astype(int).astype(str)
    graph = learn_structure(A, scoring=scoring)
    mb = markov_blanket(graph)
    reduced = graph.subgraph(mb + [TARGET]).copy()
    results = {}
    for ess in ess_values:
        bn = DiscreteBN(reduced, ess=ess).fit(A[mb + [TARGET]], states)
        p = bn.predict_proba(B)[:, 1]
        if smote and correct_prior:
            p = restore_class_prior(p, original_rate, float(np.mean(ytr)))
        results[float(ess)] = (p, mb, bn.n_params())
    return results, len(Xtr)


def fit_predict(Xtr: pd.DataFrame, ytr: np.ndarray, Xte: pd.DataFrame, categorical: list[str],
                continuous: list[str], scoring: str = "aic-d", *, ess: float = 1.0,
                n_bins: int = 5) -> tuple[np.ndarray, list[str], int]:
    """Discretiza (cortes do treino), aprende a estrutura, reduz ao Manto e prevê o teste."""
    results, _ = _predict_candidates(Xtr, ytr, Xte, categorical, continuous, scoring, [ess], n_bins=n_bins)
    return results[float(ess)]


def run_config(nome: str, Xtr, ytr, Xte, yte, categorical, continuous, scaler="sem", smote=False,
               scoring="aic-d", seed=42, *, ess=1.0, n_bins=5, sampling_strategy=1.0,
               k_neighbors=5, repair_derived=False, correct_prior=False,
               inner_split=None, ess_grid=(1.0, 10.0, 100.0), n_bins_grid=None,
               sampling_grid=None, coherent_history=False) -> dict:
    """Executa uma configuração do framework e devolve métricas de ranking e de limiar."""
    t0 = time.time()
    n_treino = len(Xtr)
    prevalencia = float(np.mean(ytr))  # do treino original, antes de qualquer reamostragem
    ytr = np.asarray(ytr)
    options = dict(n_bins=n_bins, scaler=scaler, smote=smote, sampling_strategy=sampling_strategy,
                   k_neighbors=k_neighbors, repair_derived=repair_derived, coherent_history=coherent_history,
                   correct_prior=correct_prior, seed=seed)
    validation_ap = np.nan
    selection = []
    if inner_split is None and (n_bins_grid is not None or sampling_grid is not None):
        raise ValueError("A seleção de faixas/balanceamento exige uma divisão interna do treino.")
    if inner_split is not None:
        it, iv = (np.asarray(idx, dtype=int) for idx in inner_split)
        if (not len(it) or not len(iv) or np.intersect1d(it, iv).size
                or min(it.min(), iv.min()) < 0 or max(it.max(), iv.max()) >= len(Xtr)):
            raise ValueError("Divisão interna inválida; use índices locais, não sobrepostos, do treino externo.")
        bins_values = list(n_bins_grid) if n_bins_grid is not None else [n_bins]
        ratios = list(sampling_grid) if smote and sampling_grid is not None else [sampling_strategy]
        for bins in bins_values:
            for ratio in ratios:
                if bins < 2 or not 0 < ratio <= 1:
                    raise ValueError("Grade inválida: faixas >= 2 e proporções em (0, 1].")
                if smote:
                    # Verifica viabilidade apenas nos dois conjuntos de TREINO.
                    min_ratio = max(float(z.mean() / (1 - z.mean())) for z in [ytr[it], ytr])
                    if ratio <= min_ratio:
                        continue
                candidate_options = {**options, "n_bins": bins, "sampling_strategy": ratio}
                candidates, _ = _predict_candidates(Xtr.iloc[it], ytr[it], Xtr.iloc[iv], categorical, continuous,
                                                     scoring, ess_grid, **candidate_options)
                for e, (p, _, _) in candidates.items():
                    selection.append({"n_bins": bins, "sampling_strategy": ratio, "ess": e,
                                      "pr_auc": float(average_precision_score(ytr[iv], p))})
        if not selection:
            raise ValueError("Nenhuma combinação viável na grade interna.")
        # Empate: primeira combinação da grade predefinida. Não consulta o teste.
        best = max(selection, key=lambda row: row["pr_auc"])
        ess, n_bins, sampling_strategy = best["ess"], best["n_bins"], best["sampling_strategy"]
        validation_ap = best["pr_auc"]
        options.update(n_bins=n_bins, sampling_strategy=sampling_strategy)
    candidates, n_final = _predict_candidates(Xtr, ytr, Xte, categorical, continuous, scoring, [ess], **options)
    p, mb, n_params = candidates[float(ess)]
    linha = {"config": nome, "escalonador": scaler, "smotenc": smote, "n_treino": n_treino,
             "n_treino_final": n_final, "manto": ", ".join(mb), "n_params": n_params,
             "ess": ess, "n_bins": n_bins, "pr_auc_interna": validation_ap,
             "sampling_strategy": sampling_strategy if smote else np.nan,
             "k_neighbors": k_neighbors if smote else np.nan,
             "repair_derived": bool(smote and repair_derived), "correct_prior": bool(smote and correct_prior),
             "coherent_history": bool(smote and coherent_history and
                                      {"n_faltas_anteriores", "n_agendamentos_anteriores", "taxa_faltas_anteriores"}.issubset(continuous)),
             "n_candidatos": len(selection),
             "selecao_interna": selection,
             "p_media": float(p.mean()), "segundos": time.time() - t0, **metrics(yte, p)}
    # limiar 0,5 (padrão de um classificador) e limiar = prevalência do treino (alerta acima do risco médio)
    for rotulo, limiar in [("050", 0.5), ("prev", prevalencia)]:
        linha.update({f"{k}@{rotulo}": v for k, v in classification_report_at(yte, p, limiar).items()})
    linha.update({f"{k}@top10": v for k, v in classification_report_budget(yte, p, 0.10).items()})
    return linha
