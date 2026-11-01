**Data:** 01/11/2026 · **Variáveis:** 33 · **ID:** aproximado · **Mudança:** recalibração trimestral com **janela
deslizante de 12 meses** (v4: janela expansiva)

**Verificações:** a divisão agrupada reproduz a v4 exatamente. Durante a v5 foi encontrado e corrigido um defeito: a
escolha de hiperparâmetros usava divisões internas do treino original sobre o treino do primeiro trimestre. Na v4 os
dois coincidiam; na v5, não. A escolha agora é um passo próprio, feito com o treino original, como o protocolo
declara. A v4 foi recalculada com o código corrigido e ficou idêntica, e um teste de regressão cobre o caso.

| Divisão temporal | v4 → v5 (ROC-AUC) | Diferença [IC 95%] |
|---|---|---|
| **LightGBM** | 0,568 → 0,534 | **−0,034 [−0,054; −0,016]** |
| RB (BIC) | 0,551 → 0,537 | −0,014 (n.s.) |
| Regressão logística | 0,552 → 0,544 | −0,007 (n.s.) |
| RB (AIC) | 0,527 → 0,531 | +0,004 (n.s.); Brier piora de 0,127 para 0,174 |

**Leitura:**
- **A janela deslizante piora o resultado.** Nenhum modelo melhora, e o LightGBM, o melhor da v4, perde 0,034.
  A hipótese da v4 (acompanhar a deriva mais rápido) não se confirmou.
- **A causa é a falta de dados na janela.** Nos trimestres após a pandemia, a janela de 12 meses fica quase vazia:
  em 2021-T1, o treino tem **904 agendamentos e só 126 faltas**. Com tão poucos casos, a estrutura da rede fica
  instável:

  | Trimestre | Treino (faltas) | Manto da RB (BIC) |
  |---|---|---|
  | 2019-T3 a 2020-T1 | ~8.500 (~1.000) | `estacao` |
  | 2020-T2 a 2020-T4 | 7.351 → 3.312 | `gender` |
  | 2021-T1 | 904 (126) | `faixa_horaria` |
  | 2021-T4 a 2022-T1 | ~8.300 (~1.200) | `taxa_faltas_anteriores` |

  Com janelas pequenas, o histórico do paciente, o único sinal que generaliza, perde lugar no Manto para
  variáveis sem relação estável com a falta (estação, gênero, horário). Ele só volta quando a janela reacumula
  dados, no fim de 2021.
- **O nível de risco fica mais próximo do real em alguns trimestres** (2019-T4: 11,7% previsto × 11,4% real, contra
  9,9% na v4), mas oscila e erra para mais em outros (2021-T1: 15,5% × 12,3%). Atualizar o nível rápido não compensa
  a perda de informação para ordenar os pacientes.

**Conclusão:** para esta base, a **janela expansiva (v4)** é a melhor forma de recalibração. O histórico longo
importa mais do que acompanhar a deriva rapidamente, sobretudo com os vazios de cobertura de 2020.

**Balanço das cinco versões (ROC-AUC, melhor modelo):**

| | Agrupada | Temporal |
|---|---|---|
| v1 — linha de base | LightGBM 0,678 | RB (BIC) 0,559 |
| v2 — variáveis novas | LightGBM 0,684 | RB (BIC) 0,546 |
| v3 — ID aproximado | LightGBM 0,678 | logística 0,555 |
| v4 — recalibração (expansiva) | LightGBM 0,678 | **LightGBM 0,568** |
| v5 — recalibração (12 meses) | LightGBM 0,678 | logística 0,544 |

**Ideia para um experimento futuro (v6):** separar o nível de risco da ordenação. Manter o modelo treinado com a
janela expansiva (v4) e ajustar só o intercepto à taxa de faltas dos últimos meses (*prior shift*). Isso acompanha
a deriva sem jogar fora o histórico.
