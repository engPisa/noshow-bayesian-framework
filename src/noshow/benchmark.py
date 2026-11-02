"""Divisões 70/30, modelos de comparação (regressão logística, LightGBM) e comparação por bootstrap pareado."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------------- divisões
def group_holdout(groups: np.ndarray, test_frac: float = 0.3, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Sorteia grupos (pacientes) até o teste reunir `test_frac` das linhas; nenhum grupo fica nos dois lados."""
    unicos, contagem = np.unique(groups, return_counts=True)
    ordem = np.random.default_rng(seed).permutation(len(unicos))
    acumulado = np.cumsum(contagem[ordem])
    teste = set(unicos[ordem[: np.searchsorted(acumulado, test_frac * len(groups)) + 1]])
    no_teste = np.isin(groups, list(teste))
    return np.where(~no_teste)[0], np.where(no_teste)[0]


def temporal_holdout(dates: pd.Series, train_frac: float = 0.7) -> tuple[np.ndarray, np.ndarray]:
    """Treino = datas mais antigas até completar `train_frac` das linhas; um mesmo dia nunca é partido."""
    corte = dates.sort_values().iloc[int(train_frac * len(dates))]
    treino = (dates < corte).to_numpy()
    return np.where(treino)[0], np.where(~treino)[0]


def inner_splits(dates: pd.Series, y: np.ndarray, groups: np.ndarray, mode: str, seed: int = 42):
    """Validação interna, só com o treino: CV agrupada (modo 'grupo') ou últimos 25% do tempo ('temporal')."""
    if mode == "temporal":
        tr, va = temporal_holdout(dates.reset_index(drop=True), train_frac=0.75)
        return [(tr, va)]
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    return list(cv.split(np.zeros(len(y)), y, groups))


# ---------------------------------------------------------------- modelos
def logistic_model(categorical: list[str], continuous: list[str], C: float = 1.0):
    """Regressão logística com one-hot, imputação pela mediana + indicador de ausência e padronização."""
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), categorical),
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), continuous),
    ])
    return make_pipeline(prep, LogisticRegression(C=C, max_iter=2000))


def fit_logistic(Xtr, ytr, categorical, continuous, splits) -> tuple[object, dict]:
    melhor, melhor_auc = None, -1
    for C in [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
        aucs = [roc_auc_score(ytr[va], logistic_model(categorical, continuous, C).fit(Xtr.iloc[tr], ytr[tr])
                              .predict_proba(Xtr.iloc[va])[:, 1]) for tr, va in splits]
        if np.mean(aucs) > melhor_auc:
            melhor, melhor_auc = C, np.mean(aucs)
    return logistic_model(categorical, continuous, melhor).fit(Xtr, ytr), {"C": melhor, "auc_interna": melhor_auc}


def to_lgbm(X: pd.DataFrame, categorical: list[str], categorias: dict | None = None):
    """Categóricas como `category` com o domínio do treino (categorias novas no teste viram NaN)."""
    X = X.copy()
    categorias = categorias or {c: sorted(X[c].astype(str).unique()) for c in categorical}
    for c in categorical:
        X[c] = pd.Categorical(X[c].astype(str), categories=categorias[c])
    return X, categorias


def fit_lightgbm(Xtr, ytr, categorical, continuous, splits, n_trials: int = 30, seed: int = 42):
    """LightGBM com hiperparâmetros escolhidos pelo Optuna na validação interna (ROC-AUC média)."""
    import lightgbm as lgb
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    X, categorias = to_lgbm(Xtr[categorical + continuous], categorical)

    def objetivo(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 800, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            num_leaves=trial.suggest_int("num_leaves", 4, 64, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 20, 400, log=True),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
            cat_smooth=trial.suggest_float("cat_smooth", 1, 100, log=True),
        )
        aucs = []
        for tr, va in splits:
            m = lgb.LGBMClassifier(**params, random_state=seed, verbose=-1).fit(X.iloc[tr], ytr[tr])
            aucs.append(roc_auc_score(ytr[va], m.predict_proba(X.iloc[va])[:, 1]))
        return np.mean(aucs)

    estudo = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    estudo.optimize(objetivo, n_trials=n_trials)
    modelo = lgb.LGBMClassifier(**estudo.best_params, random_state=seed, verbose=-1).fit(X, ytr)
    return modelo, categorias, {**estudo.best_params, "auc_interna": estudo.best_value}


# ---------------------------------------------------------------- comparação
def grouped_bootstrap(y: np.ndarray, preds: dict[str, np.ndarray], groups: np.ndarray, n_boot: int = 1000,
                      seed: int = 42) -> dict[str, np.ndarray]:
    """Reamostra pacientes (com reposição) e recalcula ROC-AUC e PR-AUC de cada modelo nas mesmas amostras."""
    rng = np.random.default_rng(seed)
    unicos, inverso = np.unique(groups, return_inverse=True)
    linhas_por_grupo = np.split(np.argsort(inverso), np.cumsum(np.bincount(inverso))[:-1])
    out = {f"{m}|{met}": np.empty(n_boot) for m in preds for met in ("roc_auc", "pr_auc")}
    for b in range(n_boot):
        idx = np.concatenate([linhas_por_grupo[g] for g in rng.integers(0, len(unicos), len(unicos))])
        for m, p in preds.items():
            out[f"{m}|roc_auc"][b] = roc_auc_score(y[idx], p[idx])
            out[f"{m}|pr_auc"][b] = average_precision_score(y[idx], p[idx])
    return out


def paired_difference(boot: dict, a: str, b: str, metric: str = "roc_auc") -> dict[str, float]:
    """IC 95% e p bilateral da diferença a − b, a partir das mesmas reamostragens."""
    d = boot[f"{a}|{metric}"] - boot[f"{b}|{metric}"]
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return {"diferença": float(d.mean()), "ic_inf": float(np.quantile(d, 0.025)),
            "ic_sup": float(np.quantile(d, 0.975)), "p": float(min(p, 1.0))}


# ---------------------------------------------------------------- ajuste do nível de risco
def prior_shift(p: np.ndarray, prevalencia_treino: float, prevalencia_nova: float) -> np.ndarray:
    """Correção de Saerens et al. (2002): ajusta P(y=1|x) a uma nova prevalência sem reestimar o modelo.

    É monotônica em `p`, então preserva a ordenação; muda só o nível de risco.
    """
    a = p * prevalencia_nova / prevalencia_treino
    b = (1 - p) * (1 - prevalencia_nova) / (1 - prevalencia_treino)
    return a / (a + b)


def recent_rate(dates: pd.Series, y: np.ndarray, conhecidos: np.ndarray, inicio: pd.Timestamp,
                meses: int = 6, minimo: int = 500) -> float:
    """Taxa de no-show dos registros conhecidos nos `meses` anteriores a `inicio`; se houver menos de `minimo`,
    usa os `minimo` registros mais recentes antes de `inicio`."""
    datas = dates.iloc[conhecidos].to_numpy()
    anteriores = conhecidos[datas < np.datetime64(inicio)]
    datas = dates.iloc[anteriores].to_numpy()
    janela = anteriores[datas >= np.datetime64(inicio - pd.DateOffset(months=meses))]
    if len(janela) < minimo:
        janela = anteriores[np.argsort(datas, kind="stable")[-minimo:]]
    return float(np.mean(y[janela]))
