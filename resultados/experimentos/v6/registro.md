> **Histórico exploratório (fases 7 e 8).** Este notebook foi produzido com o protocolo anterior, **antes da
> correção da sobreposição entre treino e teste** (PR #36) e sem a seleção interna de faixas, ESS e balanceamento.
> Os números e conclusões abaixo registram o processo de desenvolvimento, mas **não sustentam os resultados do
> artigo**. O protocolo final está nos notebooks `03_rede_bayesiana`, `04_experimentos` e
> `08_validacao_sensibilidade_e_base_externa`, com resultados em `resultados/artigo_sem_sobreposicao/`.

**Data:** 01/11/2026 · **Base de comparação:** v4 (a v5 foi descartada) · **Mudança:** correção da priori (Saerens)
para a taxa dos últimos 6 meses conhecidos, em cada retreino trimestral

| Divisão temporal | v4 → v6 (ROC-AUC) | Diferença [IC 95%] |
|---|---|---|
| LightGBM | 0,568 → 0,553 | **−0,015 [−0,021; −0,008]** |
| RB (BIC) | 0,551 → 0,537 | **−0,014 [−0,022; −0,005]** |
| Regressão logística | 0,552 → 0,540 | **−0,012 [−0,018; −0,005]** |
| RB (AIC) | 0,527 → 0,524 | −0,003 (n.s.) |

A divisão agrupada não muda (não há ordem no tempo para corrigir).

**Leitura:**
- **A hipótese não se confirmou: o ajuste piorou a ordenação e não melhorou a calibração como um todo.**
- **A correção exagera.** Risco médio previsto pelo LightGBM × real: 2019 15,4% × 11,0% (v4: 10,0%); 2020
  19,5% × 13,9%; 2022 18,3% × 17,2% (v4: 12,3%). Ela aproxima os anos finais e afasta os iniciais.
- **A causa é a volatilidade da taxa mensal.** Fev/2019 chegou a 28% e 2020-T2 a 47%. Uma janela de 6 meses
  captura esses picos e os projeta sobre o trimestre seguinte, que não os repete.
- **Ela não muda decisões por volume de alertas.** A correção é monotônica: dentro de um trimestre, os 10% de maior
  risco continuam os mesmos. O efeito na AUC vem só do nível entre trimestres. O notebook 06 confirma, em 7 origens,
  métricas idênticas com e sem ajuste.

**Conclusão:** para esta base, corrigir o nível com a taxa recente é instável. Se a mensageria decidir por **volume**
de alertas (os N de maior risco), o nível é irrelevante. Se decidir por **limiar** de probabilidade, a calibração deve
usar uma janela mais longa ou suavizada, e isso fica como experimento futuro.
