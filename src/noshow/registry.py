"""Registro de experimentos 70/30. Cada versão vira um notebook `notebooks/05_experimento_vN.ipynb`.

Para um novo experimento: acrescente uma entrada em EXPERIMENTOS (descrição, o que mudou, base e variáveis)
e crie o notebook da versão a partir do anterior. `compara_com` indica a versão-base quando ela não é a
imediatamente anterior (ex.: a v6 é construída sobre a v4). Os resultados ficam em `resultados/experimentos/vN/`.
"""

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from noshow.benchmark import (fit_lightgbm, fit_logistic, group_holdout, inner_splits, logistic_model, prior_shift,
                              recent_rate, temporal_holdout, to_lgbm)
from noshow.evaluation import patient_groups
from noshow.experiments import fit_predict
from noshow.features import (CATEGORICAL, CATEGORICAL_V2, CATEGORICAL_V3, CONTINUOUS, CONTINUOUS_V2, CONTINUOUS_V3,
                             FEATURES_V3, build_features)

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
    "v4": {
        "titulo": "Recalibração trimestral",
        "mudanca": "Mesmos dados e variáveis da v3. Na divisão temporal, os modelos são retreinados a cada trimestre do "
                   "teste com tudo o que já era conhecido no início do trimestre (treino + desfechos dos trimestres "
                   "anteriores; janela expansiva). Hiperparâmetros escolhidos uma vez, só no treino, como na v3. "
                   "A divisão agrupada não tem ordem no tempo e fica igual à v3.",
        "base": "aproximado", "categoricas": CATEGORICAL_V2, "continuas": CONTINUOUS_V2,
        "recalibracao": "trimestral",
    },
    "v5": {
        "titulo": "Recalibração trimestral com janela deslizante de 12 meses",
        "mudanca": "Igual à v4, mas cada retreino usa só os agendamentos dos 12 meses anteriores ao trimestre, em vez "
                   "de todo o passado, para acompanhar a deriva mais rápido. As variáveis de histórico do paciente "
                   "continuam calculadas com todo o passado dele; a janela limita apenas as linhas de treino.",
        "base": "aproximado", "categoricas": CATEGORICAL_V2, "continuas": CONTINUOUS_V2,
        "recalibracao": "trimestral", "janela_meses": 12,
    },
    "v6": {
        "titulo": "Ajuste do nível de risco (deslocamento da priori)",
        "mudanca": "Igual à v4 (recalibração trimestral, janela expansiva), mas em cada retreino as probabilidades são "
                   "corrigidas para a taxa de faltas dos últimos 6 meses conhecidos (correção de Saerens; mínimo de "
                   "500 registros). A correção é monotônica: muda o nível de risco, não a ordenação no trimestre.",
        "base": "aproximado", "categoricas": CATEGORICAL_V2, "continuas": CONTINUOUS_V2,
        "recalibracao": "trimestral", "ajuste_nivel": {"meses": 6, "minimo": 500},
        "compara_com": "v4",  # construída sobre a v4; a v5 (janela de 12 meses) foi descartada
    },
    "v7": {
        "titulo": "Histórico suavizado, recência e partida a frio",
        "mudanca": "Igual à v6, com 5 variáveis novas: taxa de faltas suavizada (Beta-Binomial, priori causal), taxa "
                   "ponderada pela recência (meia-vida de 90 dias), indicador de primeira consulta e taxas da "
                   "população por especialidade e por cidade (só dias anteriores).",
        "base": "aproximado", "categoricas": CATEGORICAL_V3, "continuas": CONTINUOUS_V3,
        "recalibracao": "trimestral", "ajuste_nivel": {"meses": 6, "minimo": 500},
    },
    "v8": {
        "titulo": "Rótulo limpo (sem cancelamentos avisados no treino)",
        "mudanca": "Igual à v7, mas os no-shows cujo motivo indica cancelamento ou aviso (`falta_avisada`, 5,4% das "
                   "faltas) saem do treino. O teste mantém o rótulo original, para comparar com a v7.",
        "base": "aproximado", "categoricas": CATEGORICAL_V3, "continuas": CONTINUOUS_V3,
        "recalibracao": "trimestral", "ajuste_nivel": {"meses": 6, "minimo": 500}, "rotulo_limpo": True,
    },
}
MODELOS = ["RB (BIC)", "RB (AIC)", "Regressão logística", "LightGBM"]


BASE_COLS = ["appointment_date", "patient_id", "no_show", "falta_avisada"] + list(FEATURES_V3)


def build_bases() -> None:
    """Gera as bases de modelagem (ID exato e aproximado) a partir da base bruta em `data/raw/`.

    `falta_avisada` vem do motivo da falta: só serve para limpar o rótulo de treino (v8), nunca como variável.
    """
    PROCESSED.mkdir(parents=True, exist_ok=True)
    for nome, relaxed in (("exato", False), ("aproximado", True)):
        build_features(relaxed_id=relaxed)[BASE_COLS].to_parquet(PROCESSED / f"modelagem_{nome}.parquet", index=False)


def load_bases() -> dict[str, pd.DataFrame]:
    if not all((PROCESSED / f"modelagem_{n}.parquet").exists() for n in ("exato", "aproximado")):
        build_bases()
    bases = {nome: pd.read_parquet(PROCESSED / f"modelagem_{nome}.parquet") for nome in ("exato", "aproximado")}
    assert bases["exato"].appointment_date.equals(bases["aproximado"].appointment_date)
    return bases


def splits(bases: dict[str, pd.DataFrame]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """As mesmas divisões em todas as versões, para as comparações serem pareadas.

    Os grupos usam a ligação aproximada: cada grupo exato está contido em um grupo aproximado, então nenhum
    paciente fica nos dois lados em nenhuma versão.
    """
    grupos = patient_groups(bases["aproximado"].patient_id)
    return {"agrupada": group_holdout(grupos, 0.3), "temporal": temporal_holdout(bases["exato"].appointment_date, 0.7)}


def _tune(Xtr, ytr, cat, cont, internas, n_trials):
    """Escolhe os hiperparâmetros da logística e do LightGBM só com o treino original (validação interna)."""
    _, info_lr = fit_logistic(Xtr, ytr, cat, cont, internas)
    _, _, info_gb = fit_lightgbm(Xtr, ytr, cat, cont, internas, n_trials=n_trials)
    return {"C": info_lr["C"], "auc_interna_lr": info_lr["auc_interna"],
            "lgbm": {k: v for k, v in info_gb.items() if k != "auc_interna"}, "auc_interna_lgbm": info_gb["auc_interna"]}


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


def recalibration_blocks(dates: pd.Series, tr: np.ndarray, te: np.ndarray, freq: str = "Q",
                         janela_meses: int | None = None):
    """Para cada trimestre do teste: (rótulo, índices de treino, posições em `te` a prever).

    O treino de um trimestre é o treino original + as linhas de teste com data anterior ao início do trimestre.
    Com `janela_meses`, só entram as linhas dos últimos `janela_meses` meses antes do início do trimestre.
    """
    datas_te = dates.iloc[te].reset_index(drop=True)
    periodos = datas_te.dt.to_period(freq)
    for q in sorted(periodos.unique()):
        conhecidos = np.concatenate([tr, te[(datas_te < q.start_time).to_numpy()]])
        if janela_meses is not None:
            inicio = q.start_time - pd.DateOffset(months=janela_meses)
            conhecidos = conhecidos[(dates.iloc[conhecidos] >= inicio).to_numpy()]
        yield q, conhecidos, np.where(periodos == q)[0]


def _fit_recalibrado(base, X, y, tr, te, cat, cont, internas, n_trials, verbose, janela_meses=None,
                     ajuste_nivel=None, valido=None):
    """Retreina a cada trimestre do teste com o treino + os desfechos dos trimestres anteriores."""
    preds = {m: np.empty(len(te)) for m in MODELOS}
    info_final, mantos = {}, []
    # hiperparâmetros escolhidos uma vez, com o treino original; os retreinos só reajustam os modelos
    fixos = _tune(X.iloc[tr], y[tr], cat, cont, internas, n_trials)
    for q, tr_q, pos in recalibration_blocks(base.appointment_date, tr, te, janela_meses=janela_meses):
        if valido is not None:  # rótulo limpo: registros conhecidos inválidos também saem do retreino
            tr_q = tr_q[valido[tr_q]]
        res, _ = _fit_models(X.iloc[tr_q], y[tr_q], X.iloc[te[pos]], cat, cont, None, n_trials, fixos)
        nivel = {}
        if ajuste_nivel:
            nivel = {"prev_treino": float(y[tr_q].mean()),
                     "prev_recente": recent_rate(base.appointment_date, y, tr_q, q.start_time, **ajuste_nivel)}
        for m, (p, info) in res.items():
            preds[m][pos] = prior_shift(p, nivel["prev_treino"], nivel["prev_recente"]) if nivel else p
            info_final[m] = info
        mantos.append({"trimestre": str(q), "n_treino": len(tr_q), "faltas_treino": int(y[tr_q].sum()),
                       "n_teste": len(pos), **nivel,
                       **{m: res[m][1]["manto"] for m in MODELOS if m.startswith("RB")}})
        if verbose:
            print(f"  recalibração {q}: treino {len(tr_q):,} | teste {len(pos):,}", flush=True)
    info_final["Regressão logística"]["auc_interna"] = fixos["auc_interna_lr"]
    info_final["LightGBM"]["auc_interna"] = fixos["auc_interna_lgbm"]
    for m in MODELOS:
        info_final[m] = {**info_final[m], "recalibracoes": len(mantos), "mantos_por_trimestre": mantos}
    return {m: (preds[m], info_final[m]) for m in MODELOS}


def run(versao: str, n_trials: int = 30, verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Executa a versão nas duas divisões e salva métricas, predições e hiperparâmetros."""
    warnings.filterwarnings("ignore")
    cfg = EXPERIMENTOS[versao]
    bases = load_bases()
    base = bases[cfg["base"]]
    y = base.no_show.to_numpy()
    grupos = patient_groups(bases["aproximado"].patient_id)

    metricas, predicoes, params, t0 = [], [], {}, time.time()
    # rótulo limpo (v8): no-shows avisados saem do treino; o teste mantém o rótulo original
    valido = (base.falta_avisada == 0).to_numpy() if cfg.get("rotulo_limpo") else None
    for divisao, (tr, te) in splits(bases).items():
        if valido is not None:
            tr = tr[valido[tr]]
        cat = [c for c in cfg["categoricas"] if not (divisao == "temporal" and c == "periodo")]  # constante no treino
        cont = cfg["continuas"]
        X = base[cat + cont]
        Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y[tr], y[te]
        internas = inner_splits(base.appointment_date.iloc[tr], ytr, grupos[tr],
                                "temporal" if divisao == "temporal" else "grupo")
        if divisao == "temporal" and cfg.get("recalibracao") == "trimestral":
            res = _fit_recalibrado(base, X, y, tr, te, cat, cont, internas, n_trials, verbose,
                                   janela_meses=cfg.get("janela_meses"), ajuste_nivel=cfg.get("ajuste_nivel"),
                                   valido=valido)
        else:
            res, _ = _fit_models(Xtr, ytr, Xte, cat, cont, internas, n_trials)

        for modelo, (p, info) in res.items():
            metricas.append({"versao": versao, "divisao": divisao, "modelo": modelo, "n_treino": len(tr),
                             "n_teste": len(te), "n_variaveis": len(cat + cont),
                             "roc_auc": roc_auc_score(yte, p), "pr_auc": average_precision_score(yte, p),
                             "brier": brier_score_loss(yte, p), "p_media": float(np.mean(p)),
                             **{k: info[k] for k in ("manto", "n_params", "C", "auc_interna", "recalibracoes")
                                if k in info}})
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
