import numpy as np
import pandas as pd

from noshow.registry import EXPERIMENTOS, recalibration_blocks


def test_recalibracao_so_usa_o_que_ja_era_conhecido():
    datas = pd.Series(pd.to_datetime(["2019-01-10", "2019-02-10",            # treino
                                      "2019-04-05", "2019-05-20",            # teste, 2º trimestre
                                      "2019-08-01", "2020-01-15"]))          # teste, 3º trim. e 2020-Q1
    tr, te = np.array([0, 1]), np.array([2, 3, 4, 5])
    blocos = list(recalibration_blocks(datas, tr, te))
    assert [str(q) for q, _, _ in blocos] == ["2019Q2", "2019Q3", "2020Q1"]
    for q, tr_q, pos in blocos:
        alvo = datas.iloc[te[pos]]
        assert (datas.iloc[tr_q] < q.start_time).all()                       # nada do próprio trimestre ou futuro
        assert alvo.dt.to_period("Q").eq(q).all()
    assert blocos[2][1].tolist() == [0, 1, 2, 3, 4]                          # 2020-Q1 conhece todo o passado
    assert sorted(np.concatenate([te[pos] for _, _, pos in blocos])) == te.tolist()  # cada linha prevista 1 vez


def test_registro_tem_campos_obrigatorios():
    for v, cfg in EXPERIMENTOS.items():
        assert {"titulo", "mudanca", "base", "categoricas", "continuas"} <= set(cfg), v


def test_janela_deslizante_descarta_o_que_e_antigo():
    datas = pd.Series(pd.to_datetime(["2018-01-10", "2019-02-10",            # treino
                                      "2019-04-05", "2020-05-20"]))          # teste
    tr, te = np.array([0, 1]), np.array([2, 3])
    blocos = {str(q): (tr_q, pos) for q, tr_q, pos in recalibration_blocks(datas, tr, te, janela_meses=12)}
    assert blocos["2019Q2"][0].tolist() == [1]          # 2018-01 fica fora da janela de 2018-04 a 2019-03
    assert blocos["2020Q2"][0].tolist() == [2]          # só 2019-04-05 está entre 2019-04 e 2020-03
    sem_janela = {str(q): tr_q for q, tr_q, _ in recalibration_blocks(datas, tr, te)}
    assert sem_janela["2020Q2"].tolist() == [0, 1, 2]


def test_recalibracao_com_janela_roda_ponta_a_ponta():
    """Regressão: a escolha de hiperparâmetros usava índices do treino original sobre o treino da janela."""
    from noshow.benchmark import inner_splits
    from noshow.registry import MODELOS, _fit_recalibrado

    rng = np.random.default_rng(0)
    n = 900
    datas = pd.Series(pd.date_range("2018-01-01", periods=n, freq="D"))
    y = (rng.random(n) < 0.2).astype(int)
    base = pd.DataFrame({"appointment_date": datas})
    X = pd.DataFrame({"cat": rng.choice(["a", "b"], n), "x": rng.normal(size=n) + y})
    tr, te = np.arange(600), np.arange(600, n)
    internas = inner_splits(datas.iloc[tr], y[tr], np.arange(600), "temporal")
    valido = rng.random(n) > 0.05
    for janela, ajuste, val in ((None, None, None), (6, None, None), (None, {"meses": 3, "minimo": 50}, valido)):
        res = _fit_recalibrado(base, X, y, tr, te, ["cat"], ["x"], internas, n_trials=2, verbose=False,
                               janela_meses=janela, ajuste_nivel=ajuste, valido=val)
        assert set(res) == set(MODELOS)
        assert all(len(p) == len(te) and np.isfinite(p).all() and (p > 0).all() and (p < 1).all()
                   for p, _ in res.values())
        if ajuste:
            blocos = res["RB (BIC)"][1]["mantos_por_trimestre"]
            assert all({"prev_treino", "prev_recente"} <= set(b) for b in blocos)
