> **Histórico exploratório (fases 7 e 8).** Este notebook foi produzido com o protocolo anterior, **antes da
> correção da sobreposição entre treino e teste** (PR #36) e sem a seleção interna de faixas, ESS e balanceamento.
> Os números e conclusões abaixo registram o processo de desenvolvimento, mas **não sustentam os resultados do
> artigo**. O protocolo final está nos notebooks `03_rede_bayesiana`, `04_experimentos` e
> `08_validacao_sensibilidade_e_base_externa`, com resultados em `resultados/artigo_sem_sobreposicao/`.

**Data:** 01/11/2026 · **Base de comparação:** v7 · **Mudança:** os 262 no-shows com motivo de cancelamento ou aviso
(5,4% das faltas) saem do treino; o teste mantém o rótulo original

| Divisão temporal | v7 → v8 (ROC-AUC) | Diferença [IC 95%] |
|---|---|---|
| RB (BIC) | 0,539 → 0,544 | +0,006 (n.s., *p* = 0,09) |
| LightGBM | 0,548 → 0,548 | 0,000 (n.s.) |
| Regressão logística | 0,536 → 0,532 | −0,004 (n.s.); PR-AUC −0,004 [−0,007; −0,002] |
| RB (AIC) | 0,525 → 0,506 | **−0,019 [−0,032; −0,007]** |

**Leitura:**
- **O rótulo limpo não trouxe ganho.** Nenhum modelo melhora de forma significativa, e a RB (AIC) piora.
- **O motivo é a escala.** São 262 registros em 4.832 faltas, concentrados em 2017–2018 e 2021. Tirá-los muda pouco o
  que o modelo aprende; para a RB (AIC), que já tinha a estrutura mais instável, a mudança de amostra alterou o Manto.
- **O ruído do rótulo não é o gargalo.** O teste continua medindo no-show como a base o define, e é isso que o
  serviço observa.

**Conclusão:** manter o rótulo original. Registrar no TCC (Seção 2.1.1) que a base mistura no-show e cancelamento
avisado, que o efeito sobre o modelo foi testado e que ele é desprezível.
