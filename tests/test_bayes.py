import networkx as nx
import numpy as np
import pandas as pd
import pytest

from noshow.bayes import DiscreteBN, markov_blanket, naive_bayes_graph
from noshow.evaluation import patient_groups, repeated_group_folds


def test_manto_de_markov_inclui_pais_filhos_e_copais():
    g = nx.DiGraph([("A", "T"), ("T", "C"), ("B", "C"), ("C", "D"), ("E", "A")])
    assert markov_blanket(g, "T") == ["A", "B", "C"]


def test_cpt_com_priori_bdeu_soma_um_e_nunca_zera():
    X = pd.DataFrame({"x": ["a", "a", "b"], "no_show": ["0", "1", "0"]})
    bn = DiscreteBN(nx.DiGraph([("x", "no_show")]), ess=1.0).fit(X, states={"x": ["a", "b", "c"], "no_show": ["0", "1"]})
    tabela = bn.cpt_frame()
    assert np.allclose(tabela.sum(axis=1), 1)
    assert (tabela.to_numpy() > 0).all()
    assert tabela.loc["a", "1"] == pytest.approx((1 + 1 / 6) / (2 + 1 / 3))


def test_predicao_com_pais_e_igual_a_cpt():
    X = pd.DataFrame({"x": ["a"] * 8 + ["b"] * 8, "no_show": ["1"] * 6 + ["0"] * 2 + ["1"] * 1 + ["0"] * 7})
    bn = DiscreteBN(nx.DiGraph([("x", "no_show")]), ess=0.0).fit(X)
    p = bn.predict_proba(pd.DataFrame({"x": ["a", "b"]}))[:, 1]
    assert p == pytest.approx([6 / 8, 1 / 8])


def test_naive_bayes_reproduz_bayes_manual():
    X = pd.DataFrame({"x": ["a", "a", "b", "b"], "z": ["u", "v", "u", "u"], "no_show": ["1", "1", "0", "0"]})
    bn = DiscreteBN(naive_bayes_graph(["x", "z"]), ess=0.0).fit(X)
    p = bn.predict_proba(pd.DataFrame({"x": ["a"], "z": ["u"]}))[0, 1]
    # P(1)P(a|1)P(u|1) / Σ_y ...  com P(a|0) = 0  →  1
    assert p == pytest.approx(1.0)


def test_estado_fora_do_dominio_gera_erro_claro():
    X = pd.DataFrame({"x": ["a", "b"], "no_show": ["0", "1"]})
    bn = DiscreteBN(nx.DiGraph([("x", "no_show")])).fit(X)
    with pytest.raises(ValueError, match="fora do domínio"):
        bn.predict_proba(pd.DataFrame({"x": ["zzz"]}))


def test_n_params_conta_parametros_livres():
    X = pd.DataFrame({"x": list("abcabc"), "no_show": list("010101")})
    bn = DiscreteBN(nx.DiGraph([("x", "no_show")])).fit(X)
    assert bn.n_params() == (3 - 1) + 3 * (2 - 1)


def test_folds_nunca_dividem_um_paciente():
    pid = pd.Series([1, 1, 2, 2, 3, 3, None, None, 4, 4] * 3, dtype="Int64")
    y = np.array([0, 1] * 15)
    grupos = patient_groups(pid)
    assert len(set(grupos[pid.isna().to_numpy()])) == pid.isna().sum()  # sem ID: um grupo por linha
    for _, _, tr, te in repeated_group_folds(y, grupos, n_splits=3, n_repeats=2):
        assert not set(grupos[tr]) & set(grupos[te])
