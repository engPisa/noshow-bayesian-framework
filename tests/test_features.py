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


def test_discretizer_coluna_sem_valores_no_treino():
    disc = Discretizer(columns=["x"]).fit(pd.DataFrame({"x": [np.nan, np.nan]}))
    assert disc.edges_["x"].size == 0
    assert disc.transform(pd.DataFrame({"x": [np.nan, 5.0]})).x.tolist() == ["ausente", "<=inf"]


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


def _agenda_v2(rows):
    df = pd.DataFrame(rows, columns=["patient_id", "appointment_date", "appointment_time", "specialty_grp", "no_show"])
    df["appointment_date"] = pd.to_datetime(df["appointment_date"])
    df["patient_id"] = df["patient_id"].astype("Int64")
    return df


def test_historico_recente_sequencia_e_janela_de_90_dias():
    from noshow.features import add_recent_history
    df = add_recent_history(_agenda_v2([
        (1, "2020-01-06", "08:00", "fono", 1),
        (1, "2020-01-13", "08:00", "fono", 1),
        (1, "2020-01-20", "08:00", "fono", 0),
        (1, "2020-01-27", "08:00", "fono", 1),
        (1, "2020-06-01", "08:00", "fono", 0),  # > 90 dias depois das anteriores
    ]))
    assert np.isnan(df["faltas_ult3"].iloc[0])  # primeiro dia: sem histórico
    assert df["faltas_ult3"].tolist()[1:] == [1, 2, 2, 2]
    assert df["faltas_seguidas"].tolist()[1:] == [1, 2, 0, 1]
    assert df["taxa_faltas_90d"].iloc[3] == pytest.approx(2 / 3)  # não inclui o próprio dia
    assert np.isnan(df["taxa_faltas_90d"].iloc[4])  # nada nos 90 dias anteriores


def test_historico_por_especialidade_e_ordem_no_dia():
    from noshow.features import add_recent_history
    df = add_recent_history(_agenda_v2([
        (1, "2020-01-06", "08:00", "fono", 1),
        (1, "2020-01-06", "09:00", "fisio", 0),
        (1, "2020-01-13", "10:00", "fisio", 0),
        (1, "2020-01-13", "08:00", "fono", 0),
    ]))
    d = df.set_index(["appointment_date", "specialty_grp"])
    assert d.loc[(pd.Timestamp("2020-01-13"), "fono"), "taxa_faltas_especialidade"] == 1.0
    assert d.loc[(pd.Timestamp("2020-01-13"), "fisio"), "taxa_faltas_especialidade"] == 0.0
    assert np.isnan(d.loc[(pd.Timestamp("2020-01-06"), "fono"), "taxa_faltas_especialidade"])
    assert d.loc[(pd.Timestamp("2020-01-13"), "fono"), "ordem_no_dia"] == 1
    assert d.loc[(pd.Timestamp("2020-01-13"), "fisio"), "ordem_no_dia"] == 2


def test_proximidade_de_feriado():
    from noshow.features import holiday_proximity
    datas = pd.Series(pd.to_datetime(["2019-12-24", "2019-12-23", "2019-11-01", "2019-11-04", "2019-07-10"]))
    # 24/12 é ponto facultativo (véspera de Natal); 02/11 é Finados
    assert holiday_proximity(datas).tolist() == ["feriado", "vespera", "vespera", "semana_de_feriado", "nao"]


def test_ligacao_aproximada_de_pacientes():
    from noshow.data import add_patient_proxy
    base = dict(gender="M", city="ITAJAÍ", disability="motor")
    df = pd.DataFrame([
        {**base, "date_of_birth": pd.Timestamp("2010-01-01"), "entry_service_date": pd.Timestamp("2019-01-01")},
        {**base, "date_of_birth": pd.NaT, "entry_service_date": pd.Timestamp("2019-01-01")},   # liga ao paciente 1
        {**base, "date_of_birth": pd.NaT, "entry_service_date": pd.Timestamp("2020-05-05")},   # paciente novo
        {**base, "date_of_birth": pd.NaT, "entry_service_date": pd.NaT},                       # sem chave
    ])
    exato = add_patient_proxy(df)
    assert exato["patient_id"].isna().tolist() == [False, True, True, True]
    out = add_patient_proxy(df, relaxed=True)
    assert out["patient_id"].iloc[1] == out["patient_id"].iloc[0]
    assert pd.notna(out["patient_id"].iloc[2]) and out["patient_id"].iloc[2] != out["patient_id"].iloc[0]
    assert pd.isna(out["patient_id"].iloc[3])
    assert out["id_qualidade"].tolist() == ["exato", "aproximado", "aproximado", "ausente"]


def _agenda_v3(rows):
    """(paciente, data, especialidade, cidade, no_show) → base com o histórico básico já calculado."""
    from noshow.features import add_patient_history
    df = pd.DataFrame(rows, columns=["patient_id", "appointment_date", "specialty_grp", "city_grp", "no_show"])
    df["appointment_date"] = pd.to_datetime(df["appointment_date"])
    df["patient_id"] = df["patient_id"].astype("Int64")
    return add_patient_history(df)


def test_taxas_da_populacao_e_suavizacao_usam_so_o_passado():
    from noshow.features import add_cold_start_and_smoothing
    df = add_cold_start_and_smoothing(_agenda_v3([
        (1, "2020-01-06", "fono", "A", 1),
        (2, "2020-01-06", "fisio", "A", 0),
        (1, "2020-01-13", "fono", "A", 0),
        (3, "2020-01-13", "fono", "B", 0),   # paciente novo
    ]), k=2)
    d = df.set_index(["patient_id", "appointment_date"])
    novo = d.loc[(3, pd.Timestamp("2020-01-13"))]
    assert novo["taxa_populacao_especialidade"] == 1.0          # única consulta anterior de fono faltou
    assert np.isnan(novo["taxa_populacao_cidade"])              # cidade B sem passado
    assert novo["primeira_consulta"] == "sim"
    assert novo["taxa_faltas_suavizada"] == pytest.approx(0.5)  # sem histórico → priori (1 falta em 2)
    antigo = d.loc[(1, pd.Timestamp("2020-01-13"))]
    assert antigo["taxa_faltas_suavizada"] == pytest.approx((1 + 2 * 0.5) / (1 + 2))
    assert np.isnan(d.loc[(1, pd.Timestamp("2020-01-06")), "taxa_populacao_especialidade"])  # nada antes do 1º dia


def test_taxa_ponderada_pela_recencia():
    from noshow.features import add_cold_start_and_smoothing
    df = add_cold_start_and_smoothing(_agenda_v3([
        (1, "2020-01-01", "fono", "A", 1),
        (1, "2020-03-31", "fono", "A", 0),   # 90 dias depois
        (1, "2020-04-01", "fono", "A", 0),
    ]), meia_vida=90)
    r = df["taxa_faltas_recencia"].tolist()
    assert np.isnan(r[0])
    assert r[1] == pytest.approx(1.0)
    peso_falta, peso_ok = 2 ** (-91 / 90), 2 ** (-1 / 90)       # a falta antiga pesa menos
    assert r[2] == pytest.approx(peso_falta / (peso_falta + peso_ok))


def test_falta_avisada_marca_so_no_show_com_motivo_de_cancelamento():
    from noshow.features import add_label_quality
    df = pd.DataFrame({"no_show": [1, 1, 0, 1], "no_show_reason": ["Desmarcou", "doente", "avisou", None]})
    assert add_label_quality(df)["falta_avisada"].tolist() == [1, 0, 0, 0]
