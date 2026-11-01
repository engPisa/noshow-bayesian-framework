import numpy as np
import pandas as pd
import pytest

from noshow.data import TARGET
from noshow.pipeline import classification_report_at, raw_matrix, smotenc_resample


def _treino(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.15).astype(int)
    X = pd.DataFrame({
        "cat": rng.choice(["a", "b", "c"], n),
        "x": rng.normal(10, 2, n) + 3 * y,
        "z": rng.exponential(100, n),
    })
    X.loc[rng.random(n) < 0.3, "z"] = np.nan
    return X, y


@pytest.mark.parametrize("scaler", [None, "minmax", "standard", "robust"])
def test_smotenc_balanceia_e_preserva_formato(scaler):
    X, y = _treino()
    Xr, yr = smotenc_resample(X, y, ["cat"], ["x", "z"], scaler=scaler)
    assert list(Xr.columns) == ["cat", "x", "z"]
    assert (yr == 1).sum() == (yr == 0).sum()
    assert set(Xr["cat"]) <= {"a", "b", "c"}
    # originais preservados no topo, na escala original
    np.testing.assert_allclose(Xr["x"].to_numpy()[: len(X)], X["x"].to_numpy())


def test_smotenc_restaura_ausentes_e_mantem_escala():
    X, y = _treino()
    Xr, yr = smotenc_resample(X, y, ["cat"], ["x", "z"], scaler="robust")
    sinteticos = Xr.iloc[len(X):]
    assert sinteticos["z"].isna().any()  # ausência herdada dos vizinhos
    assert sinteticos["x"].between(X["x"].min(), X["x"].max()).all()  # interpolação dentro do intervalo observado


def test_smotenc_e_deterministico():
    X, y = _treino()
    a = smotenc_resample(X, y, ["cat"], ["x", "z"], seed=1)[0]
    b = smotenc_resample(X, y, ["cat"], ["x", "z"], seed=1)[0]
    pd.testing.assert_frame_equal(a, b)


def test_raw_matrix_exclui_vazamento_e_datas():
    raw = pd.DataFrame({"no_show_reason": ["x"], TARGET: [1], "appointment_date": ["01/01/2020"],
                        "date_of_birth": ["01/01/2010"], "entry_service_date": ["01/01/2019"],
                        "specialty": [np.nan], "age": [10.0], "average_temp_day": [20.0], "max_temp_day": [25.0],
                        "average_rain_day": [0.0], "max_rain_day": [0.0]})
    X, cat, cont = raw_matrix(raw)
    assert "no_show_reason" not in X and TARGET not in X and "appointment_date" not in X
    assert cat == ["specialty"] and X.loc[0, "specialty"] == "nan"
    assert "age" in cont


def test_classification_report_at():
    y = np.array([1, 1, 0, 0, 0])
    p = np.array([0.9, 0.2, 0.8, 0.1, 0.1])
    r = classification_report_at(y, p, 0.5)
    assert r["recall"] == 0.5 and r["precision"] == 0.5 and r["accuracy"] == 0.6
    assert r["pct_alertas"] == pytest.approx(0.4)
