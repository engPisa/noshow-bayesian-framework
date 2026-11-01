import numpy as np
import pandas as pd
import pytest

from noshow.benchmark import group_holdout, grouped_bootstrap, paired_difference, temporal_holdout


def test_group_holdout_respeita_fracao_e_grupos():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(500), rng.integers(1, 60, 500))
    tr, te = group_holdout(groups, 0.3)
    assert not set(groups[tr]) & set(groups[te])
    assert len(te) / len(groups) == pytest.approx(0.3, abs=0.03)
    assert len(tr) + len(te) == len(groups)


def test_temporal_holdout_nao_parte_um_dia():
    datas = pd.Series(pd.to_datetime(["2020-01-01"] * 4 + ["2020-01-02"] * 3 + ["2020-01-03"] * 3))
    tr, te = temporal_holdout(datas, 0.7)
    assert datas.iloc[tr].max() < datas.iloc[te].min()
    assert set(datas.iloc[te].dt.day) == {3}


def test_bootstrap_pareado_detecta_modelo_melhor():
    rng = np.random.default_rng(1)
    n = 2000
    y = (rng.random(n) < 0.2).astype(int)
    bom = y + rng.normal(0, 0.7, n)
    ruim = rng.normal(0, 1, n)
    groups = np.arange(n) // 4
    boot = grouped_bootstrap(y, {"bom": bom, "ruim": ruim}, groups, n_boot=200)
    d = paired_difference(boot, "bom", "ruim")
    assert d["ic_inf"] > 0 and d["p"] < 0.01
