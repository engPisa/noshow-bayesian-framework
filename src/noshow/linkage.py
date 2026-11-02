"""Ligação de pacientes pela agenda semanal (notebook 07) — experimento com resultado negativo documentado.

Hipótese: na reabilitação, as terapias são semanais e o paciente mantém o horário (90% das consultas da mesma
especialidade 7 dias depois caem no mesmo horário). Um registro sem ID poderia ser ligado ao paciente que ocupa o
mesmo lugar na agenda nas semanas vizinhas. Limite: 48% dos agendamentos dividem data, horário, especialidade e
gênero com outro paciente (vários terapeutas no mesmo horário), então o lugar na agenda não identifica a pessoa.
"""

from collections import Counter

import numpy as np
import pandas as pd

AGENDA = ["appointment_time", "gender", "specialty", "patient_needs_companion"]


def schedule_key(df: pd.DataFrame) -> pd.Series:
    """Lugar na agenda: horário, gênero, especialidade e acompanhante (o dia da semana vem do deslocamento)."""
    return (df["appointment_time"].astype(str) + "|" + df["gender"].astype(str) + "|"
            + df["specialty"].fillna("?").astype(str) + "|" + df["patient_needs_companion"].astype(str))


def link_by_schedule(df: pd.DataFrame, ids: pd.Series, deslocamentos=(7, -7, 14, -14), min_votos: int = 2,
                     excluir_conflito: bool = True, max_passos: int = 20) -> pd.Series:
    """Preenche `ids` ausentes com o paciente que ocupa o mesmo lugar na agenda em semanas vizinhas.

    Um registro é ligado só se um único paciente aparece naquele lugar (em `min_votos` ou mais deslocamentos) e,
    com `excluir_conflito`, se esse paciente não tem outra consulta no mesmo dia. Ligações novas servem de base
    para o passo seguinte (propagação). Devolve os IDs preenchidos.
    """
    ids = ids.astype("Float64").copy()
    chave = schedule_key(df)
    datas = df["appointment_date"]
    for _ in range(max_passos):
        conhecidos = ids.notna()
        indice: dict[tuple, Counter] = {}
        for d in deslocamentos:
            alvo = datas[conhecidos] + pd.Timedelta(days=d)
            for k, dd, p in zip(chave[conhecidos], alvo, ids[conhecidos]):
                indice.setdefault((k, dd), Counter())[p] += 1
        ocupado = set(zip(ids[conhecidos], datas[conhecidos]))
        novos = {}
        for i in ids.index[~conhecidos]:
            votos = indice.get((chave[i], datas[i]))
            if not votos or len(votos) != 1:
                continue
            p, n = next(iter(votos.items()))
            if n >= min_votos and not (excluir_conflito and (p, datas[i]) in ocupado):
                novos[i] = p
        if not novos:
            break
        ids.loc[list(novos)] = pd.Series(novos, dtype="Float64")
    return ids


def mask_windows(ids: pd.Series, dates: pd.Series, frac: float = 0.3, dias: int = 90, seed: int = 0) -> pd.Series:
    """Simula registros sem ID: para uma fração dos pacientes, esconde o ID numa janela aleatória de `dias`."""
    rng = np.random.default_rng(seed)
    pacientes = ids.dropna().unique()
    escolhidos = rng.choice(pacientes, int(frac * len(pacientes)), replace=False)
    mascara = pd.Series(False, index=ids.index)
    for p in escolhidos:
        d = dates[ids == p]
        inicio = d.iloc[rng.integers(len(d))]
        mascara |= (ids == p) & dates.between(inicio, inicio + pd.Timedelta(days=dias))
    return mascara


def validate_linkage(df: pd.DataFrame, ids: pd.Series, mascara: pd.Series, **regra) -> dict[str, float]:
    """Esconde os IDs em `mascara`, religa pela agenda e mede cobertura e precisão contra o ID verdadeiro."""
    religado = link_by_schedule(df, ids.mask(mascara), **regra)
    ligados = mascara & religado.notna()
    return {"escondidos": int(mascara.sum()), "religados": int(ligados.sum()),
            "cobertura": float(ligados.sum() / mascara.sum()) if mascara.any() else np.nan,
            "precisao": float((religado[ligados] == ids[ligados]).mean()) if ligados.any() else np.nan}
