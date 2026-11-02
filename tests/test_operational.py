import numpy as np
import pandas as pd
import pytest

from noshow.operational import decision_table, rolling_origins, subgroup_auc


def test_origens_treinam_so_com_o_passado():
    datas = pd.Series(pd.to_datetime(["2019-01-10", "2019-03-10", "2019-07-10", "2019-12-31", "2020-01-02"]))
    blocos = list(rolling_origins(datas, ["2019-07-01", "2020-01-01", "2023-01-01"], horizonte_meses=6))
    assert len(blocos) == 2                                          # origem sem teste é ignorada
    for origem, tr, te in blocos:
        assert (datas.iloc[tr] < origem).all() and (datas.iloc[te] >= origem).all()
    assert blocos[0][2].tolist() == [2, 3]                           # 6 meses: até 2019-12-31


def test_tabela_de_decisao():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05])
    t = decision_table(y, p, volumes=(0.1, 0.3)).set_index("volume")
    assert t.loc[0.1, "recall"] == 0.5 and t.loc[0.1, "precisao"] == 1.0 and t.loc[0.1, "ganho"] == pytest.approx(5)
    assert t.loc[0.3, "recall"] == 1.0 and t.loc[0.3, "precisao"] == pytest.approx(2 / 3)


def test_auc_por_subgrupo():
    y = np.array([1, 0, 1, 0, 1, 1])
    p = np.array([0.9, 0.1, 0.2, 0.8, 0.5, 0.6])
    t = subgroup_auc(y, p, pd.Series(["a", "a", "b", "b", "c", "c"]), minimo=2)
    assert t.loc["a", "roc_auc"] == 1.0 and t.loc["b", "roc_auc"] == 0.0
    assert np.isnan(t.loc["c", "roc_auc"])                           # só uma classe


def test_run_origins_ponta_a_ponta():
    from noshow.operational import run_origins
    rng = np.random.default_rng(0)
    n = 1200
    y = (rng.random(n) < 0.15).astype(int)
    base = pd.DataFrame({
        "appointment_date": pd.date_range("2018-01-01", periods=n, freq="D"),
        "patient_id": pd.array(rng.integers(0, 200, n), dtype="Int64"),
        "no_show": y, "falta_avisada": ((rng.random(n) < 0.1) & (y == 1)).astype(int),
        "cat": rng.choice(["a", "b"], n), "x": rng.normal(size=n) + y,
    })
    configs = {"simples": {"base": "aproximado", "categoricas": ["cat"], "continuas": ["x"]},
               "completo": {"base": "aproximado", "categoricas": ["cat"], "continuas": ["x"],
                            "ajuste_nivel": {"meses": 6, "minimo": 50}, "rotulo_limpo": True}}
    M, P = run_origins({"aproximado": base}, configs, ["2019-06-01", "2020-01-01"], n_trials=2, verbose=False)
    assert len(M) == 2 * 2 * 4                                        # origens × configs × modelos
    assert M.roc_auc.between(0, 1).all() and P.p.between(0, 1).all()
    assert (M.groupby(["origem", "config"]).n_treino.first().loc[:, "completo"]
            < M.groupby(["origem", "config"]).n_treino.first().loc[:, "simples"]).all()  # rótulo limpo tira linhas


def test_auc_por_origem_nao_mistura_periodos():
    from noshow.operational import auc_by_origin
    # dentro de cada origem a ordenação é perfeita; juntando as origens, o nível diferente a estragaria
    s = pd.DataFrame({"origem": ["a"] * 4 + ["b"] * 4, "y": [0, 0, 1, 1] * 2,
                      "p": [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]})
    assert auc_by_origin(s, minimo=1).loc["todos", "roc_auc"] == 1.0
    g = auc_by_origin(s, grupos=pd.Series(["x", "x", "x", "x", "y", "y", "y", "y"]), minimo=1)
    assert g.loc["x", "origens"] == 1 and g.loc["y", "n"] == 4


def test_decisao_por_origem_alerta_dentro_do_periodo():
    from noshow.operational import decision_by_origin
    s = pd.DataFrame({"origem": ["a"] * 10 + ["b"] * 10, "y": ([1] + [0] * 9) * 2,
                      "p": list(np.linspace(0.9, 0.1, 10)) + list(np.linspace(0.09, 0.01, 10))})
    t = decision_by_origin(s, volumes=(0.1,))
    assert t.loc[0.1, "recall"] == 1.0                                # cada período alerta o seu 1º colocado
