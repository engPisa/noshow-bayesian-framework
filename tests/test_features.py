import numpy as np
import pandas as pd
import pytest

from noshow.features import Discretizer, add_age, add_patient_history, add_service_time
from noshow.weather import fix_weather


def _agenda(rows):
    df = pd.DataFrame(rows, columns=["patient_id", "appointment_date", "no_show"])
    df["appointment_date"] = pd.to_datetime(df["appointment_date"])
    df["patient_id"] = df["patient_id"].astype("Int64")
    return df


def test_historico_usa_apenas_dias_anteriores():
    df = add_patient_history(_agenda([
        (1, "2020-01-06", 1),
        (1, "2020-01-13", 0),
        (1, "2020-01-13", 1),  # duas consultas no mesmo dia
        (1, "2020-01-20", 0),
    ]))
    assert df["n_agendamentos_anteriores"].tolist() == [0, 1, 1, 3]
    assert df["n_faltas_anteriores"].tolist() == [0, 1, 1, 2]
    assert df["consultas_no_dia"].tolist() == [1, 2, 2, 1]
    assert df["faltou_ultima"].tolist() == ["sem_historico", "sim", "sim", "sim"]


def test_falta_no_mesmo_dia_nao_entra_no_historico():
    df = add_patient_history(_agenda([(1, "2020-01-06", 1), (1, "2020-01-06", 1)]))
    assert (df["n_faltas_anteriores"] == 0).all()
    assert df["taxa_faltas_anteriores"].isna().all()


def test_historico_nao_mistura_pacientes_e_ignora_ordem_das_linhas():
    df = add_patient_history(_agenda([
        (2, "2020-01-13", 0),
        (1, "2020-01-13", 0),
        (2, "2020-01-06", 1),
        (1, "2020-01-06", 0),
    ]))
    p2 = df[df.patient_id == 2].set_index("appointment_date")
    assert p2.loc["2020-01-13", "n_faltas_anteriores"] == 1
    assert p2.loc["2020-01-13", "dias_desde_ultima_consulta"] == 7
    p1 = df[df.patient_id == 1].set_index("appointment_date")
    assert p1.loc["2020-01-13", "taxa_faltas_anteriores"] == 0


def test_paciente_sem_id_fica_com_historico_ausente():
    df = add_patient_history(_agenda([(None, "2020-01-06", 1), (1, "2020-01-06", 0)]))
    sem_id = df[df.patient_id.isna()].iloc[0]
    assert pd.isna(sem_id["n_agendamentos_anteriores"])
    assert sem_id["faltou_ultima"] == "ausente"


def test_idade_calculada_na_data_da_consulta():
    df = pd.DataFrame({
        "appointment_date": pd.to_datetime(["2016-06-20", "2021-06-20", "2021-06-20"]),
        "date_of_birth": pd.to_datetime(["2010-06-20", "2010-06-20", "1900-01-01"]),
        "age": [11.0, 11.0, 122.0],
    })
    out = add_age(df)
    assert out["idade_consulta"].iloc[0] == pytest.approx(6.0, abs=0.01)
    assert out["faixa_etaria"].tolist() == ["6-11", "6-11", "ausente"]  # >100 anos: data inválida
    assert "age" not in out


def test_dias_em_servico_negativo_vira_ausente():
    df = pd.DataFrame({"appointment_date": pd.to_datetime(["2020-01-10", "2020-01-10"]),
                       "entry_service_date": pd.to_datetime(["2020-01-01", "2020-02-01"])})
    assert add_service_time(df)["dias_em_servico"].tolist()[0] == 9
    assert np.isnan(add_service_time(df)["dias_em_servico"].tolist()[1])


def test_discretizer_aprende_cortes_so_no_fit_e_trata_nan():
    treino = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, np.nan]})
    teste = pd.DataFrame({"x": [100.0, np.nan]})
    disc = Discretizer(columns=["x"], n_bins=2).fit(treino)
    out = disc.transform(teste)["x"].tolist()
    assert out == [">2.5", "ausente"]
    assert disc.edges_["x"].tolist() == [2.5]  # cortes vêm do treino, não do teste


def test_discretizer_unifica_cortes_repetidos():
    disc = Discretizer(columns=["x"], n_bins=5).fit(pd.DataFrame({"x": [0, 0, 0, 0, 0, 0, 0, 1, 2, 3]}))
    assert len(disc.edges_["x"]) < 4


def test_fix_weather_preenche_temperatura_e_mantem_chuva_ausente():
    om = pd.DataFrame({"average_temp_day": [25.0], "max_temp_day": [30.0]},
                      index=pd.to_datetime(["2022-02-01"]))
    df = pd.DataFrame({
        "appointment_date": pd.to_datetime(["2022-02-01", "2021-07-01"]),
        "average_temp_day": [np.nan, 12.0], "max_temp_day": [np.nan, 14.0],
        "max_rain_day": [np.nan, 0.0],
        "heat_intensity": ["heavy_cold", "heavy_cold"], "rain_intensity": ["no_rain", "no_rain"],
        "rainy_day_before": [0, 1], "storm_day_before": [0, 1],
    })
    out = fix_weather(df, om)
    assert out.loc[0, "heat_intensity"] == "warm" and out.loc[0, "clima_fonte"] == "open-meteo"
    assert pd.isna(out.loc[0, "rain_intensity"]) and pd.isna(out.loc[0, "max_rain_day"])
    assert out.loc[1, "heat_intensity"] == "heavy_cold" and out.loc[1, "clima_fonte"] == "base"
    assert "rainy_day_before" not in out


def test_cadastro_ausente_detecta_bloco_vazio_antes_de_preencher():
    from noshow.features import group_categories
    df = pd.DataFrame({
        "specialty": ["physiotherapy", None], "city": ["ITAJAÍ", None], "icd": [None, None],
        "gender": ["M", "F"], "disability": ["motor", None],
        "date_of_birth": pd.to_datetime(["2010-01-01", None]), "entry_service_date": pd.to_datetime(["2019-01-01", None]),
        "heat_intensity": ["mild", "mild"], "rain_intensity": ["no_rain", None], "patient_needs_companion": [1, 0],
    })
    out = group_categories(df)
    assert out["cadastro_ausente"].tolist() == ["0", "1"]
    assert out["disability"].tolist() == ["motor", "ausente"]
