# Dados externos

## openmeteo_itajai_diario.csv

Histórico meteorológico diário de Itajaí/SC (lat -26.9078, lon -48.6619), jun/2016 a fev/2022.

- **Fonte:** Open-Meteo Historical Weather API (reanálise ERA5/ECMWF) — https://open-meteo.com/en/docs/historical-weather-api
- **Licença:** CC BY 4.0 (citar Open-Meteo e Copernicus/ERA5)
- **Gerado por:** `noshow.weather.download_openmeteo()` — dados horários agregados por dia
- **Colunas:** `average_temp_day` e `max_temp_day` (°C, média e máxima horária); `average_rain_day` e
  `max_rain_day` (mm/h, média e máxima horária); `total_rain_day` (mm)

**Uso no projeto:** preencher a temperatura dos 32 dias (1.016 agendamentos) sem medição na base original.
Validação nos 969 dias medidos: temperatura média r = 0,977 (erro absoluto mediano 0,5 °C), máxima r = 0,939;
`heat_intensity` recalculada concorda em 79% (100% a no máximo uma categoria). A chuva **não** é imputada:
r = 0,45–0,60 e o ERA5 subestima picos horários (nenhum dos 49 dias de chuva forte da base aparece como forte).

## validacao_clima_base_x_openmeteo.csv

Comparação dia a dia (1.001 datas da base) entre as colunas de clima da base original e os valores do Open-Meteo
(sufixo `om_`). Gerado durante a validação de out/2026; serve como evidência da decisão de preenchimento.
