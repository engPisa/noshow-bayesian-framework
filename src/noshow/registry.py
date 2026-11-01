"""Registro de experimentos 70/30. Cada versão vira um notebook `notebooks/05_experimento_vN.ipynb`.

Para um novo experimento: acrescente uma entrada em EXPERIMENTOS (descrição, o que mudou, base e variáveis)
e crie o notebook da versão a partir do anterior. Os resultados ficam em `resultados/experimentos/vN/`.
"""

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from noshow.benchmark import (fit_lightgbm, fit_logistic, group_holdout, inner_splits, logistic_model,
                              temporal_holdout, to_lgbm)
from noshow.evaluation import patient_groups
from noshow.experiments import fit_predict
from noshow.features import CATEGORICAL, CATEGORICAL_V2, CONTINUOUS, CONTINUOUS_V2

RAIZ = Path(__file__).resolve().parents[2]
PROCESSED = RAIZ / "data" / "processed"
RESULTADOS = RAIZ / "resultados" / "experimentos"

EXPERIMENTOS = {
    "v1": {
        "titulo": "Linha de base 70/30",
        "mudanca": "Variáveis do notebook 02 (25 candidatas) com ID exato, avaliadas em 70/30 agrupado e temporal.",
        "base": "exato", "categoricas": CATEGORICAL, "continuas": CONTINUOUS,
    },
    "v2": {
        "titulo": "Histórico recente, especialidade, feriados e ordem no dia",
        "mudanca": "Acrescenta faltas nas últimas 3 e 5 consultas, sequência de faltas, taxa nos últimos 90 dias, "
                   "taxa na mesma especialidade, ordem da consulta no dia e proximidade de feriados.",
        "base": "exato", "categoricas": CATEGORICAL_V2, "continuas": CONTINUOUS_V2,
    },
    "v3": {
        "titulo": "Ligação aproximada de pacientes",
        "mudanca": "Mesmas variáveis da v2, com o ID aproximado: registros sem data de nascimento são ligados pela "
                   "chave gênero + cidade + data de entrada + deficiência (cobertura de ID de 78% para 89%).",
        "base": "aproximado", "categoricas": CATEGORICAL_V2, "continuas": CONTINUOUS_V2,
    },
}
MODELOS = ["RB (BIC)", "RB (AIC)", "Regressão logística", "LightGBM"]


def load_bases() -> dict[str, pd.DataFrame]:
    bases = {nome: pd.read_parquet(PROCESSED / f"features_v2_{nome}.parquet") for nome in ("exato", "aproximado")}
    assert bases["exato"].appointment_date.equals(bases["aproximado"].appointment_date)
    return bases


def splits(bases: dict[str, pd.DataFrame]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """As mesmas divisões em todas as versões, para as comparações serem pareadas.

    Os grupos usam a ligação aproximada: cada grupo exato está contido em um grupo aproximado, então nenhum
    paciente fica nos dois lados em nenhuma versão.
    """
    grupos = patient_groups(bases["aproximado"].patient_id)
    return {"agrupada": group_holdout(grupos, 0.3), "temporal": temporal_holdout(bases["exato"].appointment_date, 0.7)}


def _fit_models(Xtr, ytr, Xte, cat, cont, internas, n_trials, fixos=None):
    """Ajusta os quatro modelos e prevê o teste. Com `fixos`, reusa os hiperparâmetros (C e LightGBM) em vez de
    escolhê-los de novo; a estrutura da Rede Bayesiana é sempre reaprendida."""
    import lightgbm as lgb

    res = {}
    for crit in ["bic-d", "aic-d"]:
        p, mb, n_params = fit_predict(Xtr, ytr, Xte, cat, cont, scoring=crit)
        res[f"RB ({crit[:3].upper()})"] = (p, {"manto": ", ".join(mb), "n_params": n_params})
    if fixos is None:
        lr, info_lr = fit_logistic(Xtr, ytr, cat, cont, internas)
        lgbm, categorias, info_gb = fit_lightgbm(Xtr, ytr, cat, cont, internas, n_trials=n_trials)
        fixos = {"C": info_lr["C"], "lgbm": {k: v for k, v in info_gb.items() if k != "auc_interna"}}
    else:
        lr = logistic_model(cat, cont, fixos["C"]).fit(Xtr, ytr)
        Xg, categorias = to_lgbm(Xtr, cat)
        lgbm = lgb.LGBMClassifier(**fixos["lgbm"], random_state=42, verbose=-1).fit(Xg, ytr)
        info_lr, info_gb = {"C": fixos["C"]}, dict(fixos["lgbm"])
    res["Regressão logística"] = (lr.predict_proba(Xte)[:, 1], info_lr)
    res["LightGBM"] = (lgbm.predict_proba(to_lgbm(Xte, cat, categorias)[0])[:, 1], info_gb)
    return res, fixos


def run(versao: str, n_trials: int = 30, verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Executa a versão nas duas divisões e salva métricas, predições e hiperparâmetros."""
    warnings.filterwarnings("ignore")
    cfg = EXPERIMENTOS[versao]
    bases = load_bases()
    base = bases[cfg["base"]]
    y = base.no_show.to_numpy()
    grupos = patient_groups(bases["aproximado"].patient_id)

    metricas, predicoes, params, t0 = [], [], {}, time.time()
    for divisao, (tr, te) in splits(bases).items():
        cat = [c for c in cfg["categoricas"] if not (divisao == "temporal" and c == "periodo")]  # constante no treino
        cont = cfg["continuas"]
        X = base[cat + cont]
        Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y[tr], y[te]
        internas = inner_splits(base.appointment_date.iloc[tr], ytr, grupos[tr],
                                "temporal" if divisao == "temporal" else "grupo")
        res, _ = _fit_models(Xtr, ytr, Xte, cat, cont, internas, n_trials)

        for modelo, (p, info) in res.items():
            metricas.append({"versao": versao, "divisao": divisao, "modelo": modelo, "n_treino": len(tr),
                             "n_teste": len(te), "n_variaveis": len(cat + cont),
                             "roc_auc": roc_auc_score(yte, p), "pr_auc": average_precision_score(yte, p),
                             "brier": brier_score_loss(yte, p), "p_media": float(np.mean(p)),
                             **{k: info[k] for k in ("manto", "n_params", "C", "auc_interna") if k in info}})
            predicoes.append(pd.DataFrame({"versao": versao, "divisao": divisao, "modelo": modelo,
                                           "linha": te, "grupo": grupos[te], "y": yte, "p": p}))
            params[f"{divisao}|{modelo}"] = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in info.items()}
        if verbose:
            print(f"{divisao:9s} " + " | ".join(f"{m} {roc_auc_score(yte, p):.3f}" for m, (p, _) in res.items())
                  + f"  ({time.time() - t0:.0f}s)", flush=True)

    saida = RESULTADOS / versao
    saida.mkdir(parents=True, exist_ok=True)
    M, P = pd.DataFrame(metricas), pd.concat(predicoes, ignore_index=True)
    M.to_csv(saida / "metricas.csv", index=False)
    P.to_parquet(saida / "predicoes.parquet", index=False)
    (saida / "parametros.json").write_text(json.dumps({"config": {k: v for k, v in cfg.items()}, "modelos": params},
                                                      indent=1, ensure_ascii=False, default=str))
    return M, P


def load(versao: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    saida = RESULTADOS / versao
    return pd.read_csv(saida / "metricas.csv"), pd.read_parquet(saida / "predicoes.parquet")
