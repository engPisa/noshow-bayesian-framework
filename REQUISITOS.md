# Matriz de rastreabilidade dos requisitos

Gerada por `scripts/rastreabilidade.py` a partir de `requisitos.yml` (fonte única). Não edite à mão.

**88 itens** · 88 entregues · 0 em andamento · **44 com teste automatizado** (50%); os demais são verificados por execução de notebook ou revisão.

Prefixos: `RF` funcional · `RNF` não funcional · `CA` critério de tarefa técnica.

| ID | Requisito | Issue | Implementação | Verificação | Entrega |
|---|---|---|---|---|---|
| **RF-001** | Converter datas `dd/mm/aaaa` e tratar datas inválidas como ausentes | #1 | `src/noshow/data.py::load_raw` | notebook 01_diagnostico §4 | #15 |
| **RF-002** | Converter textos em branco para ausentes e o alvo para 0/1 | #1 | `src/noshow/data.py::load_raw` | notebook 01_diagnostico §1 | #15 |
| **RF-003** | Reconstruir `patient_id` a partir de atributos estáveis do cadastro | #1 | `src/noshow/data.py::add_patient_proxy` | `tests/test_features.py::test_ligacao_aproximada_de_pacientes`<br>notebook 01_diagnostico §5 | #15 |
| **RF-004** | Marcar `no_show_reason` como coluna de vazamento | #1 | `src/noshow/data.py::LEAKAGE_COLS` | `tests/test_pipeline.py::test_raw_matrix_exclui_vazamento_e_datas`<br>notebook 01_diagnostico §3 | #15 |
| **RF-005** | Remover os valores de preenchimento e as colunas de dia anterior | #2 | `src/noshow/weather.py::fix_weather` | `tests/test_features.py::test_fix_weather_preenche_temperatura_e_mantem_chuva_ausente` | #15 |
| **RF-006** | Preencher a temperatura dos dias ausentes com o histórico do Open-Meteo (ERA5) | #2 | `src/noshow/weather.py::fix_weather`<br>`src/noshow/weather.py::load_openmeteo` | `tests/test_features.py::test_fix_weather_preenche_temperatura_e_mantem_chuva_ausente` | #15 |
| **RF-007** | Validar a fonte externa contra os dias medidos antes de usá-la | #2 | `data/external/validacao_clima_base_x_openmeteo.csv` | notebook 00_eda Atividade 3 (Sobre o clima) | #15 |
| **RF-008** | Registrar a origem de cada valor em `clima_fonte` | #2 | `src/noshow/weather.py::fix_weather` | `tests/test_features.py::test_fix_weather_preenche_temperatura_e_mantem_chuva_ausente` | #15 |
| **RNF-001** | Download reproduzível do histórico (`download_openmeteo`) | #2 | `src/noshow/weather.py::download_openmeteo`<br>`data/external/README.md` | execução manual documentada em data/external/README.md | #15 |
| **RF-009** | Notebook de diagnóstico: vazamento, deriva temporal, ausentes informativos, pacientes recorrentes | #3 | `notebooks/01_diagnostico.ipynb` | execução completa do notebook | #15 |
| **RF-010** | Notebook de EDA com 8 atividades e checkpoints preenchidos com resultados reais | #3 | `notebooks/00_eda.ipynb` | execução completa do notebook | #15 |
| **RF-011** | Testes de hipótese com tamanho de efeito | #3 | `notebooks/00_eda.ipynb Atividade 7` | execução completa do notebook | #15 |
| **RF-012** | Idade na data da consulta e faixas etárias | #4 | `src/noshow/features.py::add_age` | `tests/test_features.py::test_idade_calculada_na_data_da_consulta` | #16 |
| **RF-013** | Histórico do paciente usando apenas dias anteriores | #4 | `src/noshow/features.py::add_patient_history` | `tests/test_features.py::test_historico_usa_apenas_dias_anteriores`<br>`tests/test_features.py::test_falta_no_mesmo_dia_nao_entra_no_historico`<br>`tests/test_features.py::test_historico_nao_mistura_pacientes_e_ignora_ordem_das_linhas`<br>`tests/test_features.py::test_paciente_sem_id_fica_com_historico_ausente` | #16 |
| **RF-014** | Ausência como categoria explícita (`cadastro_ausente`) | #4 | `src/noshow/features.py::group_categories` | `tests/test_features.py::test_cadastro_ausente_detecta_bloco_vazio_antes_de_preencher` | #16 |
| **RF-015** | Calendário (dia, faixa horária com 17h e 18h separadas, mês, estação, período) | #4 | `src/noshow/features.py::add_calendar` | notebook 02_engenharia_variaveis §3 | #16 |
| **RF-016** | Discretizador por quantis ajustado só no treino | #4 | `src/noshow/features.py::Discretizer` | `tests/test_features.py::test_discretizer_aprende_cortes_so_no_fit_e_trata_nan`<br>`tests/test_features.py::test_discretizer_unifica_cortes_repetidos` | #16 |
| **RNF-002** | Testes cobrindo vazamento do histórico | #4 | `tests/test_features.py` | CI (pytest) | #16 |
| **RF-017** | Exemplo do histórico com um paciente real | #5 | `notebooks/02_engenharia_variaveis.ipynb §4` | verificações de sanidade no notebook | #16 |
| **RF-018** | Controle do efeito do histórico pela deriva temporal | #5 | `notebooks/02_engenharia_variaveis.ipynb §4` | execução completa do notebook | #16 |
| **RF-019** | Catálogo, associação com o alvo e matriz de redundância | #5 | `notebooks/02_engenharia_variaveis.ipynb §5–§7` | notebook confere que a base salva é idêntica à do módulo | #16 |
| **RF-020** | Hill Climbing (BIC/AIC) com conhecimento especialista | #6 | `src/noshow/bayes.py::learn_structure` | notebook 03_rede_bayesiana §1 | #17 |
| **RF-021** | Manto de Markov (pais, filhos e pais dos filhos) | #6 | `src/noshow/bayes.py::markov_blanket` | `tests/test_bayes.py::test_manto_de_markov_inclui_pais_filhos_e_copais` | #17 |
| **RF-022** | CPTs com priori BDeu e inferência vetorizada sobre o Manto | #6 | `src/noshow/bayes.py::DiscreteBN` | `tests/test_bayes.py::test_cpt_com_priori_bdeu_soma_um_e_nunca_zera`<br>`tests/test_bayes.py::test_predicao_com_pais_e_igual_a_cpt`<br>`tests/test_bayes.py::test_naive_bayes_reproduz_bayes_manual`<br>`tests/test_bayes.py::test_estado_fora_do_dominio_gera_erro_claro`<br>`tests/test_bayes.py::test_n_params_conta_parametros_livres` | #17 |
| **RF-023** | Validação cruzada repetida agrupada por paciente e Wilcoxon pareado | #6 | `src/noshow/evaluation.py::repeated_group_folds`<br>`src/noshow/evaluation.py::patient_groups`<br>`src/noshow/evaluation.py::paired_wilcoxon` | `tests/test_bayes.py::test_folds_nunca_dividem_um_paciente` | #17 |
| **RF-024** | Rede completa, rede do Manto e Naive Bayes completo em 5 × 3 folds | #7 | `notebooks/03_rede_bayesiana.ipynb §5`<br>`resultados/h2_validacao_cruzada.csv` | notebook confere métricas idênticas entre rede completa e Manto | #17 |
| **RF-025** | ROC-AUC, PR-AUC e Brier com Wilcoxon pareado | #7 | `src/noshow/evaluation.py::metrics`<br>`src/noshow/evaluation.py::paired_wilcoxon` | notebook 03_rede_bayesiana §5 | #17 |
| **RF-026** | Sensibilidade ao critério de pontuação e ao limite de pais | #7 | `notebooks/03_rede_bayesiana.ipynb §2 e §6` | execução completa do notebook | #17 |
| **RNF-003** | Reprodutibilidade com `PYTHONHASHSEED=0` | #7 | `README.md (Reprodutibilidade)`<br>`kernel do projeto` | duas execuções seguidas do notebook 03 com resultado idêntico | #17 |
| **RF-027** | MinMax, Standard e Robust ajustados só no treino | #8 | `src/noshow/pipeline.py::smotenc_resample` | `tests/test_pipeline.py::test_smotenc_balanceia_e_preserva_formato` | #18 |
| **RF-028** | SMOTENC com imputação e indicadores de ausência herdados pelos sintéticos | #8 | `src/noshow/pipeline.py::smotenc_resample` | `tests/test_pipeline.py::test_smotenc_restaura_ausentes_e_mantem_escala`<br>`tests/test_pipeline.py::test_smotenc_e_deterministico` | #18 |
| **RF-029** | Controle com as colunas originais da base | #8 | `src/noshow/pipeline.py::raw_matrix` | `tests/test_pipeline.py::test_raw_matrix_exclui_vazamento_e_datas` | #18 |
| **RF-030** | Métricas de matriz de confusão por limiar | #8 | `src/noshow/pipeline.py::classification_report_at` | `tests/test_pipeline.py::test_classification_report_at` | #18 |
| **RF-031** | 7 configurações × 15 folds agrupados por paciente | #9 | `src/noshow/experiments.py::run_config`<br>`scripts/run_h3.py`<br>`resultados/h3_validacao_cruzada.csv` | notebook 04_experimentos §2 | #18 |
| **RF-032** | Wilcoxon com correção de Holm | #9 | `notebooks/04_experimentos.ipynb §2` | execução completa do notebook | #18 |
| **RF-033** | Validação temporal (treino até 2020, teste 2021–2022) | #9 | `scripts/run_h3.py::main`<br>`resultados/h3_validacao_temporal.csv` | notebook 04_experimentos §5 | #18 |
| **RF-034** | Explicação do efeito do SMOTENC sobre a estrutura da rede | #9 | `notebooks/04_experimentos.ipynb §3` | execução completa do notebook | #18 |
| **RF-035** | Faltas nas últimas 3/5 consultas, sequência de faltas e taxa em 90 dias | #10 | `src/noshow/features.py::add_recent_history` | `tests/test_features.py::test_historico_recente_sequencia_e_janela_de_90_dias` | #19 |
| **RF-036** | Taxa de faltas na mesma especialidade e ordem da consulta no dia | #10 | `src/noshow/features.py::add_recent_history` | `tests/test_features.py::test_historico_por_especialidade_e_ordem_no_dia` | #19 |
| **RF-037** | Proximidade de feriados nacionais/SC e pontos facultativos | #10 | `src/noshow/features.py::holiday_proximity` | `tests/test_features.py::test_proximidade_de_feriado` | #19 |
| **RF-038** | Ligação aproximada sem data de nascimento com `id_qualidade` | #10 | `src/noshow/data.py::add_patient_proxy` | `tests/test_features.py::test_ligacao_aproximada_de_pacientes` | #19 |
| **RF-039** | Divisão agrupada por paciente (30% das linhas) e temporal (sem partir um dia) | #11 | `src/noshow/benchmark.py::group_holdout`<br>`src/noshow/benchmark.py::temporal_holdout` | `tests/test_benchmark.py::test_group_holdout_respeita_fracao_e_grupos`<br>`tests/test_benchmark.py::test_temporal_holdout_nao_parte_um_dia` | #19 |
| **RF-040** | Regressão logística e LightGBM com Optuna na validação interna | #11 | `src/noshow/benchmark.py::fit_logistic`<br>`src/noshow/benchmark.py::fit_lightgbm`<br>`src/noshow/benchmark.py::inner_splits` | `tests/test_registry.py::test_recalibracao_com_janela_roda_ponta_a_ponta` | #19 |
| **RF-041** | Bootstrap reamostrando pacientes e diferença pareada | #11 | `src/noshow/benchmark.py::grouped_bootstrap`<br>`src/noshow/benchmark.py::paired_difference` | `tests/test_benchmark.py::test_bootstrap_pareado_detecta_modelo_melhor` | #19 |
| **RF-042** | Registro de versões com título, mudança, base e variáveis | #12 | `src/noshow/registry.py::EXPERIMENTOS` | `tests/test_registry.py::test_registro_tem_campos_obrigatorios` | #19 |
| **RF-043** | Gerador de notebooks `05_experimento_vN` | #12 | `scripts/novo_experimento.py` | notebooks v1–v5 gerados e executados | #19 |
| **RF-044** | Registro escrito de cada versão em `resultados/experimentos/vN/registro.md` | #12 | `resultados/experimentos/vN/registro.md` | seção "Registro do experimento" de cada notebook | #19 |
| **RF-045** | Experimentos v1 (linha de base), v2 (variáveis) e v3 (ID aproximado) | #12 | `notebooks/05_experimento_v1.ipynb`<br>`notebooks/05_experimento_v2.ipynb`<br>`notebooks/05_experimento_v3.ipynb` | execução completa dos notebooks | #19 |
| **RF-046** | Retreino trimestral com janela expansiva (v4) | #13 | `src/noshow/registry.py::recalibration_blocks`<br>`src/noshow/registry.py::_fit_recalibrado` | `tests/test_registry.py::test_recalibracao_so_usa_o_que_ja_era_conhecido` | #19 |
| **RF-047** | Retreino com janela deslizante de 12 meses (v5) | #13 | `src/noshow/registry.py::recalibration_blocks` | `tests/test_registry.py::test_janela_deslizante_descarta_o_que_e_antigo` | #19 |
| **RF-048** | Hiperparâmetros escolhidos uma vez, só com o treino original | #13 | `src/noshow/registry.py::_tune` | `tests/test_registry.py::test_recalibracao_com_janela_roda_ponta_a_ponta` | #19 |
| **CA-001** | Instalação reproduzível em uma máquina nova | #14 | `README.md (Instalação)` | comandos executados no ambiente do projeto | #20 |
| **CA-002** | Indica que a base de dados não é versionada e onde colocá-la | #14 | `README.md (Instalação)` | revisão do README | #20 |
| **CA-003** | Descreve o fluxo issue → PR → main com conventional commits | #14 | `README.md (Contribuição)` | revisão do README | #20 |
| **CA-004** | Lista as versões v1–v5 | #14 | `README.md (Registrar um novo experimento)` | revisão do README | #20 |
| **CA-005** | As bases são geradas por código versionado | #21 | `src/noshow/registry.py::build_bases` | bases regeneradas do zero a partir de data/raw | #27 |
| **CA-006** | As colunas existentes ficam idênticas às usadas nos experimentos v1–v5 | #21 | `src/noshow/registry.py::BASE_COLS` | comparação das 36 colunas existentes nas duas bases (idênticas) | #27 |
| **CA-007** | `load_bases` gera as bases quando ausentes | #21 | `src/noshow/registry.py::load_bases` | execução dos notebooks 05 sem bases prévias | #27 |
| **RF-049** | Corrigir a priori em cada retreino com a taxa dos últimos 6 meses conhecidos (correção de Saerens) | #22 | `src/noshow/benchmark.py::prior_shift`<br>`src/noshow/benchmark.py::recent_rate`<br>`src/noshow/registry.py::_fit_recalibrado` | `tests/test_benchmark.py::test_prior_shift_preserva_ordem_e_ajusta_nivel`<br>`tests/test_benchmark.py::test_taxa_recente_usa_janela_e_minimo`<br>`tests/test_registry.py::test_recalibracao_com_janela_roda_ponta_a_ponta` | #27 |
| **RF-050** | Manter a ordenação dentro de cada trimestre (a correção é monotônica) | #22 | `src/noshow/benchmark.py::prior_shift` | `tests/test_benchmark.py::test_prior_shift_preserva_ordem_e_ajusta_nivel` | #27 |
| **RF-051** | Curvas de calibração por decil nos notebooks de experimento | #22 | `scripts/novo_experimento.py (Calibração)` | notebook 05_experimento_v6 §2 | #27 |
| **RF-052** | Notebook `05_experimento_v6` com registro | #22 | `notebooks/05_experimento_v6.ipynb`<br>`resultados/experimentos/v6/registro.md` | execução completa do notebook | #27 |
| **RF-053** | Taxa de faltas suavizada (Beta-Binomial) com priori causal | #23 | `src/noshow/features.py::add_cold_start_and_smoothing` | `tests/test_features.py::test_taxas_da_populacao_e_suavizacao_usam_so_o_passado` | #27 |
| **RF-054** | Taxa de faltas ponderada pela recência (decaimento exponencial) | #23 | `src/noshow/features.py::add_cold_start_and_smoothing` | `tests/test_features.py::test_taxa_ponderada_pela_recencia` | #27 |
| **RF-055** | Taxas da população por especialidade e por cidade, calculadas só com o passado (partida a frio) | #23 | `src/noshow/features.py::add_cold_start_and_smoothing` | `tests/test_features.py::test_taxas_da_populacao_e_suavizacao_usam_so_o_passado` | #27 |
| **RF-056** | Indicador de primeira consulta | #23 | `src/noshow/features.py::add_cold_start_and_smoothing` | `tests/test_features.py::test_taxas_da_populacao_e_suavizacao_usam_so_o_passado` | #27 |
| **RNF-004** | Testes de vazamento para todas as variáveis novas | #23 | `tests/test_features.py` | CI (pytest) | #27 |
| **RF-057** | Marcar os no-shows com motivo de cancelamento/aviso na base, fora das variáveis do modelo | #24 | `src/noshow/features.py::add_label_quality` | `tests/test_features.py::test_falta_avisada_marca_so_no_show_com_motivo_de_cancelamento` | #27 |
| **RF-058** | Excluir esses registros apenas do treino | #24 | `src/noshow/registry.py::run`<br>`src/noshow/registry.py::_fit_recalibrado` | `tests/test_registry.py::test_recalibracao_com_janela_roda_ponta_a_ponta`<br>`tests/test_operational.py::test_run_origins_ponta_a_ponta` | #27 |
| **RF-059** | Avaliar no teste com o rótulo original, para comparar com a v7 | #24 | `src/noshow/registry.py::run` | notebook 05_experimento_v8 §3 (comparação pareada com a v7) | #27 |
| **RF-060** | Várias origens temporais (treino com o passado, teste nos 6 meses seguintes) | #25 | `src/noshow/operational.py::rolling_origins`<br>`src/noshow/operational.py::run_origins` | `tests/test_operational.py::test_origens_treinam_so_com_o_passado`<br>`tests/test_operational.py::test_run_origins_ponta_a_ponta` | #27 |
| **RF-061** | Comparar a linha de base com a melhor configuração da fase 7 | #25 | `notebooks/06_avaliacao_operacional.ipynb` | execução completa do notebook | #27 |
| **RF-062** | Faltas detectadas e precisão para volumes de alerta fixos | #25 | `src/noshow/operational.py::decision_table` | `tests/test_operational.py::test_tabela_de_decisao` | #27 |
| **RF-063** | Desempenho por subgrupo (faixa etária, especialidade, histórico, qualidade do ID) | #25 | `src/noshow/operational.py::subgroup_auc` | `tests/test_operational.py::test_auc_por_subgrupo` | #27 |
| **CA-008** | Numeração global única: `RF-NNN` (funcional), `RNF-NNN` (não funcional) e `CA-NNN` (critério de tarefa técnica) | #26 | `requisitos.yml` | scripts/rastreabilidade.py valida unicidade e sequência | #27 |
| **CA-009** | Cada requisito aponta implementação, verificação (teste automatizado ou notebook) e PR de entrega | #26 | `requisitos.yml`<br>`REQUISITOS.md` | scripts/rastreabilidade.py valida campos e testes citados | #27 |
| **CA-010** | Cada cenário de aceite indica os requisitos que verifica | #26 | `requisitos.yml (cenarios)` | scripts/rastreabilidade.py valida que os ids citados existem na issue | #27 |
| **CA-011** | Fonte única versionada (`requisitos.yml`) que gera a matriz `REQUISITOS.md` e sincroniza as issues, inclusive as fechadas | #26 | `scripts/rastreabilidade.py` | `tests/test_rastreabilidade.py::test_rastreabilidade_valida_a_fonte_unica` | #27 |
| **RF-064** | Métricas por origem restritas a pacientes com ID (exato e exato + aproximado) | #28 | `src/noshow/operational.py::auc_by_origin`<br>`notebooks/06_avaliacao_operacional.ipynb §4` | `tests/test_operational.py::test_auc_por_origem_nao_mistura_periodos` | #31 |
| **RF-065** | Faltas alcançadas por volume de alertas em cada cenário | #28 | `src/noshow/operational.py::decision_by_origin`<br>`notebooks/06_avaliacao_operacional.ipynb §4` | `tests/test_operational.py::test_decisao_por_origem_alerta_dentro_do_periodo` | #31 |
| **RF-066** | Ressalva explícita de que o recorte é uma aproximação (sem ID concentrado em 2021–2022) | #28 | `notebooks/06_avaliacao_operacional.ipynb §4` | revisão do notebook | #31 |
| **RF-067** | Curva de aprendizado por paciente: ROC-AUC por número de consultas anteriores, dentro de cada origem | #29 | `src/noshow/operational.py::auc_by_origin`<br>`notebooks/06_avaliacao_operacional.ipynb §5` | `tests/test_operational.py::test_auc_por_origem_nao_mistura_periodos`<br>notebook 06_avaliacao_operacional §5 | #31 |
| **RF-068** | Tempo até o paciente acumular histórico (dias até a 5ª consulta) | #29 | `notebooks/06_avaliacao_operacional.ipynb §5` | execução completa do notebook | #31 |
| **RF-069** | Efeito do volume de dados no modelo ao longo das origens | #29 | `notebooks/06_avaliacao_operacional.ipynb §5` | execução completa do notebook | #31 |
| **RNF-005** | Recomendação para produção: registrar intervenções e manter grupo de controle, para que a retroalimentação não aprenda o efeito do próprio lembrete | #29 | `notebooks/06_avaliacao_operacional.ipynb §5` | revisão do notebook | #31 |
| **RF-070** | Ligação por lugar na agenda (dia da semana, horário, especialidade, gênero, acompanhante) com votos, exclusão de conflitos e propagação | #30 | `src/noshow/linkage.py::link_by_schedule`<br>`src/noshow/linkage.py::schedule_key` | `tests/test_linkage.py::test_liga_quando_um_unico_paciente_ocupa_o_horario_nas_semanas_vizinhas`<br>`tests/test_linkage.py::test_nao_liga_quando_o_horario_e_dividido_ou_o_candidato_ja_esta_no_dia` | #31 |
| **RF-071** | Validação escondendo o ID de pacientes conhecidos (precisão × cobertura) | #30 | `src/noshow/linkage.py::mask_windows`<br>`src/noshow/linkage.py::validate_linkage` | `tests/test_linkage.py::test_validacao_mede_cobertura_e_precisao`<br>`tests/test_linkage.py::test_mascara_esconde_janelas_de_uma_fracao_dos_pacientes` | #31 |
| **RF-072** | Notebook com as regras testadas e a decisão de não aplicar | #30 | `notebooks/07_ligacao_por_agenda.ipynb` | execução completa do notebook | #31 |
