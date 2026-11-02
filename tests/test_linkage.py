import numpy as np
import pandas as pd

from noshow.linkage import link_by_schedule, mask_windows, validate_linkage


def _agenda(rows):
    """(id, data, horário, especialidade) → agenda com gênero e acompanhante fixos."""
    df = pd.DataFrame(rows, columns=["id", "appointment_date", "appointment_time", "specialty"])
    df["appointment_date"] = pd.to_datetime(df["appointment_date"])
    df["gender"], df["patient_needs_companion"] = "M", 0
    return df


def test_liga_quando_um_unico_paciente_ocupa_o_horario_nas_semanas_vizinhas():
    df = _agenda([(1, "2020-01-06", "14:00", "fono"), (None, "2020-01-13", "14:00", "fono"),
                  (1, "2020-01-20", "14:00", "fono")])
    assert link_by_schedule(df, df["id"]).tolist() == [1, 1, 1]


def test_nao_liga_quando_o_horario_e_dividido_ou_o_candidato_ja_esta_no_dia():
    dividido = _agenda([(1, "2020-01-06", "14:00", "fono"), (2, "2020-01-06", "14:00", "fono"),
                        (None, "2020-01-13", "14:00", "fono"), (1, "2020-01-20", "14:00", "fono"),
                        (2, "2020-01-20", "14:00", "fono")])
    assert pd.isna(link_by_schedule(dividido, dividido["id"]).iloc[2])
    conflito = _agenda([(1, "2020-01-06", "14:00", "fono"), (1, "2020-01-13", "14:00", "fono"),
                        (None, "2020-01-13", "14:00", "fono"), (1, "2020-01-20", "14:00", "fono")])
    assert pd.isna(link_by_schedule(conflito, conflito["id"]).iloc[2])  # o paciente 1 já está naquele dia


def test_validacao_mede_cobertura_e_precisao():
    datas = pd.date_range("2020-01-06", periods=20, freq="7D")
    df = pd.concat([_agenda([(1, d, "14:00", "fono") for d in datas]),
                    _agenda([(2, d, "15:00", "fisio") for d in datas])], ignore_index=True)
    mascara = pd.Series(False, index=df.index)
    mascara.iloc[[5, 6, 25]] = True
    r = validate_linkage(df, df["id"].astype(float), mascara)
    assert r["escondidos"] == 3 and r["cobertura"] == 1.0 and r["precisao"] == 1.0


def test_mascara_esconde_janelas_de_uma_fracao_dos_pacientes():
    ids = pd.Series(np.repeat(np.arange(10.0), 20))
    datas = pd.Series(np.tile(pd.date_range("2020-01-01", periods=20, freq="7D"), 10))
    m = mask_windows(ids, datas, frac=0.3, dias=28, seed=1)
    assert ids[m].nunique() == 3 and m.groupby(ids).sum().max() <= 5
