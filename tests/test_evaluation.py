import numpy as np
import pandas as pd
import pytest

from noshow.evaluation import patient_groups, record_groups, repeated_group_folds


def test_registros_iguais_sem_id_ficam_juntos_sem_usar_o_desfecho():
    records = pd.DataFrame({"appointment_date": ["2020-01-01"] * 3 + ["2020-01-02"],
                            "appointment_time": ["09:00", "09:00", "10:00", "09:00"],
                            "city": [None] * 4, "no_show": [0, 1, 0, 0],
                            "no_show_reason": [None, "cancelado", None, None]})
    pid = pd.Series([None] * 4, dtype="Int64")
    groups = patient_groups(pid, records=records)
    assert groups[0] == groups[1]
    assert len(set(groups)) == 3  # outro horário ou outra data são distintos
    changed = records.assign(no_show=1-records.no_show, no_show_reason="qualquer motivo")
    np.testing.assert_array_equal(groups, patient_groups(pid, records=changed))


def test_uniao_preserva_pacientes_e_e_transitiva():
    # O registro ambíguo liga dois grupos, sem dividir as demais consultas deles.
    pid = pd.Series([1, 1, None, 2, 2, None], dtype="Int64")
    records = pd.DataFrame({"appointment_date": ["a", "b", "b", "b", "c", "d"]})
    groups = patient_groups(pid, records=records)
    assert len(set(groups[:5])) == 1
    assert groups[5] != groups[0]


def test_grupos_independem_da_ordem_e_do_indice_das_linhas():
    pid = pd.Series([1, 1, None, None, 2, 2], dtype="Int64")
    records = pd.DataFrame({"date": [1, 2, 3, 3, 4, 5], "value": [None] * 6})
    groups = patient_groups(pid, records=records)
    order = [4, 2, 0, 5, 3, 1]
    shuffled = patient_groups(pid.iloc[order], records=records.iloc[order])
    np.testing.assert_array_equal(groups[order] == groups[order][:, None],
                                  shuffled == shuffled[:, None])


def test_divisoes_externas_e_internas_nao_separam_registros_iguais():
    records = pd.DataFrame({"appointment_date": np.repeat(np.arange(30), 2),
                            "no_show": [0, 1] * 30})
    pid = pd.Series([None] * len(records), dtype="Int64")
    groups = patient_groups(pid, records=records)
    signatures = record_groups(records)
    y = records.no_show.to_numpy()
    for _, _, tr, te in repeated_group_folds(y, groups, n_splits=3, n_repeats=2):
        assert not set(signatures[tr]) & set(signatures[te])
        _, _, it, iv = next(repeated_group_folds(y[tr], groups[tr], n_splits=3, n_repeats=1))
        assert not set(signatures[tr[it]]) & set(signatures[tr[iv]])


def test_rejeita_registros_desalinhados_ou_sem_campos_de_entrada():
    pid = pd.Series([None, None], dtype="Int64")
    with pytest.raises(ValueError, match="alinhados"):
        patient_groups(pid, records=pd.DataFrame({"date": [1, 2]}, index=[1, 0]))
    with pytest.raises(ValueError, match="campos originais"):
        record_groups(pd.DataFrame({"no_show": [0, 1], "no_show_reason": [None, "faltou"]}))
