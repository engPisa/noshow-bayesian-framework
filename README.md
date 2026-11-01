# Predição de No-show com Redes Bayesianas — TCC

Framework de normalização e engenharia de dados para predição de *no-show* em sistemas de agendamento, com
Redes Bayesianas e delimitação de variáveis pelo Manto de Markov.

**Autores:** Cesar Augusto da Silva Pisa e Natan Felipe Kainak Arend · **Orientador:** Prof. Vinicius Tessele
· Engenharia de Software, Faculdade Donaduzzi (2026)

## Estrutura

```
data/raw/            base original (não versionada; ver Instalação)
data/external/       histórico meteorológico do Open-Meteo + validação (ver README próprio)
data/processed/      base de variáveis gerada pelo notebook 02 (não versionada)
src/noshow/          o framework
  data.py              carga e ID aproximado do paciente
  weather.py           correção e preenchimento do clima
  features.py          engenharia de variáveis e discretização
  bayes.py             estrutura, Manto de Markov, CPTs e inferência
  evaluation.py        validação cruzada agrupada por paciente e métricas
  pipeline.py          escalonamento + SMOTENC (com ausentes) e métricas por limiar
  experiments.py       configurações dos experimentos H1/H3
  benchmark.py         divisões 70/30, regressão logística, LightGBM (Optuna) e bootstrap por paciente
  registry.py          registro dos experimentos 70/30 (v1, v2, v3, ...)
notebooks/
  00_eda.ipynb                  análise exploratória (formato do desafio de EDA)
  01_diagnostico.ipynb          qualidade, vazamento e estrutura de validação
  02_engenharia_variaveis.ipynb variáveis candidatas → data/processed/features.parquet
  03_rede_bayesiana.ipynb       estrutura, Manto de Markov e H2
  04_experimentos.ipynb         dados brutos × framework (H1/H3), validação cruzada e temporal
  05_experimento_vN.ipynb       um notebook por experimento 70/30 (agrupado e temporal)
scripts/run_h3.py    calcula os experimentos do notebook 04 (~25 min)
scripts/novo_experimento.py  gera o notebook de um experimento registrado
resultados/          métricas por fold dos experimentos
tests/               testes automatizados (pytest)
```

## Instalação

Requer [uv](https://docs.astral.sh/uv/) e Python 3.12.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt -e .
.venv/bin/python -m ipykernel install --user --name tcc-noshow --display-name "TCC no-show (.venv)" \
    --env PYTHONHASHSEED 0
.venv/bin/nbstripout --install --attributes .gitattributes   # limpa as saídas dos notebooks no commit
```

A base de dados **não é versionada**. Coloque o arquivo `medical-appointments-no-show-en.csv` em `data/raw/`
antes de executar os notebooks.

No VS Code/Jupyter, selecione o kernel **"TCC no-show (.venv)"**. Os notebooks devem ser executados em ordem
(00 → 04); o 02 gera a base usada pelo 03 e pelo 04. O notebook 04 carrega os resultados de
`resultados/`; para recalculá-los: `PYTHONHASHSEED=0 .venv/bin/python scripts/run_h3.py`.

## Registrar um novo experimento

Cada experimento 70/30 tem o próprio notebook, e todos usam as mesmas divisões, para que as comparações sejam
pareadas.

1. Acrescente uma entrada `"vN"` em `EXPERIMENTOS` (`src/noshow/registry.py`): título, o que muda, base e variáveis.
2. `.venv/bin/python scripts/novo_experimento.py vN` → cria `notebooks/05_experimento_vN.ipynb`.
3. Execute o notebook: ele roda o experimento, salva em `resultados/experimentos/vN/` e compara com a versão
   anterior.
4. Escreva a análise em `resultados/experimentos/vN/registro.md`, rode o passo 2 de novo e reexecute o notebook.

| Versão | Experimento |
|---|---|
| v1 | Linha de base 70/30 (variáveis do notebook 02) |
| v2 | + histórico recente, por especialidade, feriados, ordem no dia |
| v3 | + ligação aproximada de pacientes |
| v4 | + recalibração trimestral na divisão temporal |
| v5 | recalibração com janela deslizante de 12 meses |

## Contribuição

- Commits em [Conventional Commits](https://www.conventionalcommits.org/pt-br/), **somente o header** (sem body),
  por exemplo `feat(features): adiciona histórico recente`. O CI rejeita commits fora do padrão.
- Cada entrega vira uma issue (templates em `.github/ISSUE_TEMPLATE`) dentro da milestone da fase e um PR para a
  `main` seguindo `.github/PULL_REQUEST_TEMPLATE/default.md`, fechando a issue com `Closes #n`.
- Os notebooks são versionados **sem saídas** (filtro `nbstripout`).
- A base de dados e as predições linha a linha não são versionadas.

## Testes

```bash
.venv/bin/python -m pytest tests
```

## Reprodutibilidade

- **`PYTHONHASHSEED=0`**: o Hill Climbing do pgmpy desempata pela ordem de iteração de conjuntos, que depende
  dessa semente. O comando de instalação acima já a fixa no kernel. Sem ela, arestas periféricas da rede podem
  variar entre execuções (o Manto de Markov não muda).
- Validação cruzada com sementes fixas (`seed=42` e seguintes) em `noshow.evaluation.repeated_group_folds`.
- O histórico do Open-Meteo pode ser baixado de novo com `noshow.weather.download_openmeteo()`.
