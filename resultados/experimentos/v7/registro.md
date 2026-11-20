> **Histórico exploratório (fases 7 e 8).** Este notebook foi produzido com o protocolo anterior, **antes da
> correção da sobreposição entre treino e teste** (PR #36) e sem a seleção interna de faixas, ESS e balanceamento.
> Os números e conclusões abaixo registram o processo de desenvolvimento, mas **não sustentam os resultados do
> artigo**. O protocolo final está nos notebooks `03_rede_bayesiana`, `04_experimentos` e
> `08_validacao_sensibilidade_e_base_externa`, com resultados em `resultados/artigo_sem_sobreposicao/`.

**Data:** 01/11/2026 · **Base de comparação:** v6 · **Mudança:** 5 variáveis novas: taxa de faltas suavizada
(Beta-Binomial), taxa ponderada pela recência (meia-vida de 90 dias), primeira consulta e taxas da população por
especialidade e por cidade

| Divisão | Modelo | v6 → v7 (ROC-AUC) | Diferença [IC 95%] |
|---|---|---|---|
| **Agrupada** | **LightGBM** | 0,679 → **0,690** | **+0,011 [+0,003; +0,018]** |
| Agrupada | demais | ±0,003 | n.s. |
| Temporal | RB (BIC) / LightGBM | 0,537 → 0,539 / 0,553 → 0,548 | n.s. |
| Temporal | Regressão logística | 0,540 → 0,536 | −0,004 [−0,008; −0,001] |

**Leitura:**
- **Melhor resultado da divisão agrupada até aqui:** LightGBM com ROC-AUC de 0,690 e PR-AUC de 0,212.
- **Na divisão temporal única, nada muda.** Mas a avaliação com 7 origens (notebook 06), mais estável que uma
  divisão só, mostra que as variáveis da fase 7 **superam a linha de base em todas as 7 origens** com o LightGBM
  (+0,016, *p* = 0,016, o menor possível com 7 pares) e em 5 de 7 com a regressão logística (+0,016, *p* = 0,078).
- **A partida a frio não resolveu os pacientes novos:** AUC de 0,497 para quem não tem consultas anteriores
  (notebook 06). As taxas da população por especialidade e cidade são iguais para todos os novos do mesmo grupo e
  não diferenciam quem vai faltar.

**Conclusão:** as variáveis da v7 dão ganho pequeno, mas consistente entre origens. Ficam como o conjunto de
variáveis recomendado.
