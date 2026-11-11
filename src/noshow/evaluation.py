"""Validação cruzada agrupada por paciente e métricas para classificação desbalanceada."""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from noshow.data import LEAKAGE_COLS, TARGET


def record_groups(records: pd.DataFrame) -> np.ndarray:
    """Equivalência dos campos originais, excluindo alvo e motivo da falta.

    Inclui as datas e horários: não agrupa consultas de dias diferentes só porque
    têm os mesmos atributos. Não tenta identificar pessoas nem remove linhas.
    """
    columns = [c for c in records if c not in {TARGET, *LEAKAGE_COLS}]
    if not columns:
        raise ValueError("Forneça os campos originais do agendamento, além do desfecho.")
    # Factorize compara as chaves completas e trata ausências iguais; não depende
    # somente da igualdade de um hash de 64 bits.
    return pd.MultiIndex.from_frame(records[columns]).factorize(sort=False)[0]


def patient_groups(patient_id: pd.Series, records: pd.DataFrame | None = None) -> np.ndarray:
    """Agrupa pacientes e, com `records`, também agendamentos indistinguíveis.

    Forneça o CSV original, alinhado a `patient_id`, para a proteção contra
    sobreposição de registros. A união é transitiva e conserva os grupos de
    pacientes conhecidos. Sem `records`, mantém a API legada (um grupo por linha
    sem ID), usada pelos experimentos posteriores ao núcleo 01–04.
    """
    ids = patient_id.astype("Int64")
    sem_id = ids.isna().to_numpy()
    grupos = ids.fillna(-1).to_numpy(dtype=np.int64)
    grupos[sem_id] = -np.arange(1, sem_id.sum() + 1)
    if records is None:
        return grupos
    if len(records) != len(ids) or not records.index.equals(ids.index):
        raise ValueError("Registros e IDs precisam estar alinhados, com os mesmos índices e tamanho.")

    parent = np.arange(len(ids))
    size = np.ones(len(ids), dtype=int)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        a, b = find(a), find(b)
        if a == b:
            return
        if size[a] < size[b]:
            a, b = b, a
        parent[b] = a
        size[a] += size[b]

    for keys in (grupos, record_groups(records)):
        first = {}
        for i, key in enumerate(keys):
            if key in first:
                union(first[key], i)
            else:
                first[key] = i
    roots = np.fromiter((find(i) for i in range(len(ids))), dtype=np.int64, count=len(ids))
    return pd.factorize(roots, sort=False)[0]


def repeated_group_folds(y, groups, n_splits: int = 5, n_repeats: int = 3, seed: int = 42):
    """Gera (repetição, fold, treino, teste), mantendo cada grupo inteiro."""
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
