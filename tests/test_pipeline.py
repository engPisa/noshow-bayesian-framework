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


def test_correcao_da_priori_recupera_prevalencia_e_preserva_ranking():
    from noshow.pipeline import restore_class_prior
    p = np.array([0.1, 0.5, 0.9])
    corrected = restore_class_prior(p, original_rate=0.1, sampled_rate=0.5)
    assert corrected[1] == pytest.approx(0.1)
    assert np.array_equal(np.argsort(p), np.argsort(corrected))
    np.testing.assert_allclose(restore_class_prior(p, 0.1, 0.1), p)


def test_sinteticos_preservam_relacoes_de_idade_e_clima():
    from noshow.weather import rain_category, heat_category
    X, y = _treino()
    X = X.rename(columns={"x": "max_temp_day", "z": "max_rain_day"})
    X["heat_intensity"] = heat_category(X.max_temp_day).fillna("ausente")
    X["rain_intensity"] = rain_category(X.max_rain_day).fillna("ausente")
    X["idade_consulta"] = np.linspace(1, 90, len(X))
    X["faixa_etaria"] = pd.cut(X.idade_consulta, [0, 6, 12, 18, 60, np.inf], right=False,
                               labels=["0-5", "6-11", "12-17", "18-59", "60+"]).astype(str)
    cat = ["cat", "heat_intensity", "rain_intensity", "faixa_etaria"]
    cont = ["max_temp_day", "max_rain_day", "idade_consulta"]
    out, _ = smotenc_resample(X, y, cat, cont, repair_derived=True)
    pd.testing.assert_frame_equal(out.iloc[:len(X)].reset_index(drop=True), X[cat + cont], check_dtype=False)
    synthetic = out.iloc[len(X):]
    assert synthetic.rain_intensity.tolist() == rain_category(synthetic.max_rain_day).fillna("ausente").tolist()
    assert synthetic.heat_intensity.tolist() == heat_category(synthetic.max_temp_day).fillna("ausente").tolist()
    expected = pd.cut(synthetic.idade_consulta, [0, 6, 12, 18, 60, np.inf], right=False,
                      labels=["0-5", "6-11", "12-17", "18-59", "60+"]).astype(str)
    assert synthetic.faixa_etaria.tolist() == expected.tolist()


def test_coluna_totalmente_ausente_nao_impede_smotenc():
    X, y = _treino()
    X["z"] = np.nan
    out, _ = smotenc_resample(X, y, ["cat"], ["x", "z"])
    assert out.z.isna().all()


def test_historico_sintetico_recalcula_razao_e_preserva_reais():
    X, y = _treino()
    X = X.rename(columns={"x": "n_agendamentos_anteriores", "z": "n_faltas_anteriores"})
    X.n_agendamentos_anteriores = np.arange(len(X), dtype=float)
    X.n_faltas_anteriores = np.floor(X.n_agendamentos_anteriores * 0.2)
    X.loc[:10, ["n_agendamentos_anteriores", "n_faltas_anteriores"]] = np.nan
    X["taxa_faltas_anteriores"] = X.n_faltas_anteriores / X.n_agendamentos_anteriores.where(X.n_agendamentos_anteriores > 0)
    cont = ["n_agendamentos_anteriores", "n_faltas_anteriores", "taxa_faltas_anteriores"]
    out, _ = smotenc_resample(X, y, ["cat"], cont, coherent_history=True, sampling_strategy=0.5)
    pd.testing.assert_frame_equal(out.iloc[:len(X)], X[["cat"] + cont], check_dtype=False)
    synth = out.iloc[len(X):]
    expected = synth.n_faltas_anteriores / synth.n_agendamentos_anteriores.where(synth.n_agendamentos_anteriores > 0)
    np.testing.assert_allclose(synth.taxa_faltas_anteriores, expected, equal_nan=True)
    assert synth.taxa_faltas_anteriores.dropna().between(0, 1).all()
    # Alterar a razão redundante não deve mudar os vizinhos nem as contagens sintéticas.
    changed = X.assign(taxa_faltas_anteriores=np.linspace(0, 1, len(X)))
    other, _ = smotenc_resample(changed, y, ["cat"], cont, coherent_history=True, sampling_strategy=0.5)
    np.testing.assert_allclose(synth[cont[:2]], other.iloc[len(X):][cont[:2]], equal_nan=True)


def test_orcamento_fixo_independe_da_ordem_dos_empates():
    from noshow.pipeline import classification_report_budget
    y, p = np.array([1, 0, 1, 0]), np.array([0.8, 0.8, 0.8, 0.1])
    a = classification_report_budget(y, p, fraction=0.25)
    idx = [2, 3, 1, 0]
    b = classification_report_budget(y[idx], p[idx], fraction=0.25)
    assert a == b
    assert a["n_alertas"] == 1
    assert a["precision"] == pytest.approx(2 / 3)
    assert a["recall"] == pytest.approx(1 / 3)
