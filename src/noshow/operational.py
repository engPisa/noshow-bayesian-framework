"""Avaliação operacional (notebook 06): várias origens no tempo, métrica de decisão e desempenho por subgrupo."""

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from noshow.benchmark import inner_splits, prior_shift, recent_rate
from noshow.evaluation import patient_groups
from noshow.registry import _fit_models

VOLUMES = (0.05, 0.10, 0.20)


def rolling_origins(dates: pd.Series, origens: list[str], horizonte_meses: int = 6):
    """Para cada origem: treino = tudo antes dela; teste = os `horizonte_meses` seguintes."""
    for o in pd.to_datetime(origens):
        fim = o + pd.DateOffset(months=horizonte_meses)
        tr = np.where(dates < o)[0]
        te = np.where((dates >= o) & (dates < fim))[0]
        if len(te):
            yield o, tr, te


def decision_table(y: np.ndarray, p: np.ndarray, volumes=VOLUMES) -> pd.DataFrame:
    """Alertando os `v` de maior risco: faltas detectadas (recall), precisão e ganho sobre sortear."""
    linhas, base = [], y.mean()
    ordem = np.argsort(-p, kind="stable")
    for v in volumes:
        k = max(1, int(round(v * len(y))))
        acertos = y[ordem[:k]].sum()
        linhas.append({"volume": v, "alertas_por_1000": 1000 * k / len(y), "recall": acertos / y.sum(),
                       "precisao": acertos / k, "ganho": (acertos / k) / base,
                       "faltas_alcancadas_por_1000": 1000 * acertos / len(y),
                       "faltas_ao_acaso_por_1000": 1000 * v * base})
    return pd.DataFrame(linhas)


def subgroup_auc(y: np.ndarray, p: np.ndarray, grupos: pd.Series, minimo: int = 200) -> pd.DataFrame:
    """ROC-AUC por subgrupo (grupos com menos de `minimo` registros ou sem as duas classes ficam sem AUC)."""
    linhas = []
    for g, idx in pd.Series(range(len(y))).groupby(grupos.to_numpy()).groups.items():
        idx = np.asarray(idx)
        ok = len(idx) >= minimo and len(np.unique(y[idx])) == 2
        linhas.append({"grupo": g, "n": len(idx), "taxa": y[idx].mean(),
                       "roc_auc": roc_auc_score(y[idx], p[idx]) if ok else np.nan})
    return pd.DataFrame(linhas).set_index("grupo")


def run_origins(bases: dict[str, pd.DataFrame], configs: dict, origens: list[str], horizonte_meses: int = 6,
                n_trials: int = 15, verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Avalia cada configuração em cada origem. `configs`: nome → {base, categoricas, continuas, ajuste_nivel,
    rotulo_limpo}. Hiperparâmetros escolhidos na validação interna temporal de cada origem."""
    warnings.filterwarnings("ignore")
    ref = bases["aproximado"]
    y = ref.no_show.to_numpy()
    grupos = patient_groups(ref.patient_id)
    metricas, predicoes, t0 = [], [], time.time()
    for origem, tr0, te in rolling_origins(ref.appointment_date, origens, horizonte_meses):
        for nome, cfg in configs.items():
            base = bases[cfg["base"]]
            tr = tr0[(base.falta_avisada.to_numpy()[tr0] == 0)] if cfg.get("rotulo_limpo") else tr0
            cat, cont = [c for c in cfg["categoricas"] if c != "periodo"], cfg["continuas"]
            X = base[cat + cont]
            internas = inner_splits(base.appointment_date.iloc[tr], y[tr], grupos[tr], "temporal")
            res, _ = _fit_models(X.iloc[tr], y[tr], X.iloc[te], cat, cont, internas, n_trials)
            nivel = cfg.get("ajuste_nivel")
            if nivel:
                p_tr, p_novo = y[tr].mean(), recent_rate(base.appointment_date, y, tr, origem, **nivel)
            for modelo, (p, info) in res.items():
                if nivel:
                    p = prior_shift(p, p_tr, p_novo)
                metricas.append({"origem": origem.date(), "config": nome, "modelo": modelo, "n_treino": len(tr),
                                 "n_teste": len(te), "taxa_teste": y[te].mean(),
                                 "roc_auc": roc_auc_score(y[te], p), "pr_auc": average_precision_score(y[te], p),
                                 "brier": brier_score_loss(y[te], p), "p_media": float(p.mean()),
                                 "manto": info.get("manto")})
                predicoes.append(pd.DataFrame({"origem": origem.date(), "config": nome, "modelo": modelo,
                                               "linha": te, "y": y[te], "p": p}))
        if verbose:
            print(f"origem {origem.date()}: treino {len(tr0):,} | teste {len(te):,} ({time.time() - t0:.0f}s)",
                  flush=True)
    return pd.DataFrame(metricas), pd.concat(predicoes, ignore_index=True)
