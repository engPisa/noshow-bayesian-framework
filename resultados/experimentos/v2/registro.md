**Data:** 01/11/2026 · **Variáveis:** 33 (+8: histórico recente, por especialidade, sequência de faltas, feriados,
ordem no dia) · **ID:** exato

| Divisão | Modelo | v1 → v2 (ROC-AUC) | Diferença [IC 95%] |
|---|---|---|---|
| Agrupada | LightGBM | 0,678 → 0,684 | +0,006 [+0,000; +0,013] |
| Agrupada | Regressão logística | 0,641 → 0,643 | +0,002 (n.s.) |
| Agrupada | RB (BIC) / RB (AIC) | sem mudança | 0 (o Manto não mudou) |
| Temporal | RB (BIC) | 0,558 → 0,546 | −0,012 (n.s.) |
| Temporal | LightGBM | 0,516 → 0,500 | −0,016 [−0,029; −0,002] |

**Leitura:**
- **As variáveis novas não trouxeram ganho relevante.** O único ganho com IC positivo é marginal (+0,006 no
  LightGBM agrupado), e a PR-AUC desse mesmo modelo caiu. Na divisão temporal, os modelos ficaram iguais ou piores.
- **O histórico recente substitui, mas não soma.** Na temporal, a RB (BIC) trocou `taxa_faltas_anteriores` por
  `taxa_faltas_90d`. As variáveis novas carregam a mesma informação que a taxa acumulada (correlação alta, AUC
  univariada 0,54–0,58), então mais variáveis do mesmo tipo só dão mais chance de sobreajuste.
- **O gargalo não é a forma de resumir o histórico, e sim quem tem histórico.** Na divisão temporal, uma grande
  parte do teste não tem ID e, portanto, não tem histórico nenhum.

**Próximo experimento (v3):** aumentar a cobertura de ID com a ligação aproximada de pacientes.
