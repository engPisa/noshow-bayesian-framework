import networkx as nx
import numpy as np
import pandas as pd

from noshow.experiments import fit_predict, prepare_discrete, run_config
from noshow.preprocessing import FoldCategories, UNKNOWN


def test_cidades_sao_agrupadas_pelas_frequencias_do_treino():
    # city_grp simula um agrupamento global anterior: a coluna original deve prevalecer.
    train = pd.DataFrame({"city": ["A", "A", "B"], "city_grp": ["outras"] * 3})
    enc = FoldCategories(["city_grp"], min_city=2).fit(train)
    test = pd.DataFrame({"city": ["A", "B", "C", None], "city_grp": ["A"] * 4})
    assert enc.transform(test).city_grp.tolist() == ["A", "outras", "outras", "ausente"]
    assert enc.cities_ == {"A"}


def test_dominio_nao_depende_das_categorias_do_teste():
    train = pd.DataFrame({"cat": ["a", "b"], "x": [1.0, 2.0]})
    test = pd.DataFrame({"cat": ["inedita"], "x": [100.0]})
    _, out, states = prepare_discrete(train, test, ["cat"], ["x"])
    assert out.cat.iloc[0] == UNKNOWN
    assert "inedita" not in states["cat"]
    _, _, other_states = prepare_discrete(train, train, ["cat"], ["x"])
    assert states == other_states


def test_acrescentar_linha_de_teste_nao_altera_predicao_existente(monkeypatch):
    monkeypatch.setattr("noshow.experiments.learn_structure", lambda *a, **k: nx.DiGraph([("cat", "no_show")]))
    X = pd.DataFrame({"cat": ["a"] * 20 + ["b"] * 20})
    y = np.array([1] * 15 + [0] * 5 + [0] * 20)
    a = fit_predict(X, y, pd.DataFrame({"cat": ["a"]}), ["cat"], [])[0]
    b = fit_predict(X, y, pd.DataFrame({"cat": ["a", "inedita"]}), ["cat"], [])[0]
    np.testing.assert_array_equal(a, b[:1])
    assert np.isfinite(b).all()


def test_escolha_de_ess_nao_depende_dos_rotulos_do_teste(monkeypatch):
    monkeypatch.setattr("noshow.experiments.learn_structure", lambda *a, **k: nx.DiGraph([("cat", "no_show")]))
    X = pd.DataFrame({"cat": ["a", "b"] * 20})
    y = np.array([0, 1, 1, 0] * 10)
    options = dict(inner_split=(np.arange(24), np.arange(24, 40)))
    a = run_config("test", X, y, X.iloc[:4], y[:4], ["cat"], [], **options)
    b = run_config("test", X, y, X.iloc[:4], 1 - y[:4], ["cat"], [], **options)
    assert (a["ess"], a["pr_auc_interna"], a["p_media"]) == (b["ess"], b["pr_auc_interna"], b["p_media"])


def test_busca_de_faixas_e_balanceamento_ignora_rotulos_externos(monkeypatch):
    monkeypatch.setattr("noshow.experiments.learn_structure", lambda *a, **k: nx.DiGraph([("cat", "no_show")]))
    X = pd.DataFrame({"cat": ["a", "b"] * 40, "x": np.linspace(0, 10, 80)})
    y = np.array([0, 0, 0, 1] * 20)
    options = dict(inner_split=(np.arange(64), np.arange(64, 80)),
                   n_bins_grid=(3, 5), sampling_grid=(0.5, 1.0), smote=True)
    a = run_config("test", X, y, X.iloc[:8], y[:8], ["cat"], ["x"], **options)
    b = run_config("test", X, y, X.iloc[:8], 1-y[:8], ["cat"], ["x"], **options)
    for key in ["ess", "n_bins", "sampling_strategy", "selecao_interna", "p_media"]:
        assert a[key] == b[key]
    assert a["n_candidatos"] == 12
