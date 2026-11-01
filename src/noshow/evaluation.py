"""Validação cruzada agrupada por paciente e métricas para classificação desbalanceada."""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


def patient_groups(patient_id: pd.Series) -> np.ndarray:
    """Grupos para a validação: o paciente; registros sem ID viram grupos unitários."""
    ids = patient_id.astype("Int64")
    sem_id = ids.isna().to_numpy()
    grupos = ids.fillna(-1).to_numpy(dtype=np.int64)
    grupos[sem_id] = -np.arange(1, sem_id.sum() + 1)
    return grupos


def repeated_group_folds(y, groups, n_splits: int = 5, n_repeats: int = 3, seed: int = 42):
    """Gera (repetição, fold, índices de treino, índices de teste). Um paciente nunca aparece nos dois lados."""
    for r in range(n_repeats):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        for f, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y, groups)):
            yield r, f, tr, te


def metrics(y_true, p) -> dict[str, float]:
    return {"roc_auc": roc_auc_score(y_true, p), "pr_auc": average_precision_score(y_true, p),
            "brier": brier_score_loss(y_true, p)}


def paired_wilcoxon(a: pd.Series, b: pd.Series, alternative: str = "two-sided") -> dict[str, float]:
    """Wilcoxon pareado entre dois modelos avaliados nos mesmos folds."""
    d = np.asarray(a) - np.asarray(b)
    res = stats.wilcoxon(a, b, alternative=alternative, zero_method="zsplit")
    return {"diferença_mediana": float(np.median(d)), "pares_a_maior": int((d > 0).sum()), "pares": len(d),
            "p": float(res.pvalue)}
