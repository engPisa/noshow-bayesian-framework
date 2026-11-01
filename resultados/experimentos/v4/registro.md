**Data:** 01/11/2026 · **Variáveis:** 33 (iguais à v3) · **ID:** aproximado · **Mudança:** recalibração trimestral
(janela expansiva) na divisão temporal; 11 retreinos de 2019-T3 a 2022-T1

**Verificação:** a divisão agrupada, que não mudou, reproduz a v3 **exatamente** (diferença 0,0000 nos quatro modelos).

| Divisão temporal | v3 → v4 (ROC-AUC) | Diferença [IC 95%] | PR-AUC v3 → v4 |
|---|---|---|---|
| **LightGBM** | 0,552 → **0,568** | **+0,016 [+0,005; +0,028]** | 0,171 → **0,182** (+0,011, *p* = 0,02) |
| **RB (BIC)** | 0,534 → 0,551 | **+0,017 [+0,006; +0,028]** | 0,150 → 0,161 (+0,011, *p* < 0,001) |
| RB (AIC) | 0,506 → 0,527 | +0,021 [+0,007; +0,035] | 0,144 → 0,151 |
| Regressão logística | 0,555 → 0,552 | −0,003 (n.s.) | 0,163 → 0,168 (n.s.) |

**Leitura:**
- **A recalibração melhora três dos quatro modelos**, com ganho pequeno, mas consistente e significativo: cerca de
  +0,02 de ROC-AUC e +0,01 de PR-AUC. O LightGBM passa a ser o melhor modelo na divisão temporal (0,568).
- **O ganho vem do nível de risco, e não de uma ordenação melhor dentro de cada período.** A taxa de faltas sobe de
  11% (2019) para 17% (2022). Retreinando, o modelo acompanha parte dessa subida, e a AUC calculada sobre todo o
  teste melhora porque os períodos de risco maior recebem probabilidades maiores.
- **Mas a correção é parcial.** Risco médio previsto × real em 2022: LightGBM 12,3% × 17,2% (v3: 9,1%) e
  RB (BIC) 12,8% × 17,2% (v3: 11,6%). A janela expansiva mantém os anos antigos, de taxa baixa, no treino e dilui a
  atualização.
- **O Manto da RB (BIC) muda ao longo do tempo:** `taxa_faltas_90d` até 2021-T2 e `taxa_faltas_anteriores` a partir
  de 2021-T3. Com mais histórico acumulado, a taxa de longo prazo volta a ser a melhor resumidora. O modelo
  continua com uma única variável em todos os 11 retreinos.
- **Para o serviço:** reestimar o modelo periodicamente é barato (a RB é uma contagem; o LightGBM leva segundos com
  hiperparâmetros fixos) e deve fazer parte da operação.

**Balanço das quatro versões (ROC-AUC, melhor modelo):**

| | Agrupada | Temporal |
|---|---|---|
| v1 — linha de base | LightGBM 0,678 | RB (BIC) 0,559 |
| v2 — variáveis novas | LightGBM 0,684 | RB (BIC) 0,546 |
| v3 — ID aproximado | LightGBM 0,678 | logística 0,555 |
| v4 — recalibração | LightGBM 0,678 | **LightGBM 0,568** |

**Próximo experimento sugerido (v5):** **janela deslizante** (retreinar só com os últimos 12 meses), para acompanhar
a deriva com mais rapidez. O risco é ter menos dados: o período de 2020 tem pouquíssimos registros.
