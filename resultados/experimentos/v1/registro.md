**Data:** 01/11/2026 · **Variáveis:** 25 (notebook 02) · **ID:** exato

| Divisão | Melhor modelo | ROC-AUC | Observação (diferença pareada, IC 95%) |
|---|---|---|---|
| Agrupada | LightGBM | 0,678 | +0,066 sobre a RB (BIC), IC inteiramente positivo |
| Temporal | **RB (BIC)** | 0,559 | supera LightGBM (−0,042), logística (−0,019) e RB (AIC) (−0,021); todos com IC abaixo de zero |

**Leitura:**
- **O teto da divisão agrupada é ~0,68**, e não 0,8. Os modelos de maior capacidade ganham da Rede Bayesiana, mas
  o LightGBM se apoia em `mes` e `periodo`, variáveis de calendário e deriva que não ajudam a prever o futuro.
- **Prevendo o futuro, a ordem se inverte:** a RB (BIC), cujo Manto é só `taxa_faltas_anteriores`, é o melhor
  modelo. Os modelos mais flexíveis aprendem padrões de calendário de 2016–2019 que não se repetem.
- Os resultados reproduzem os dos notebooks 03 e 04 (RB AIC ≈ 0,63 na divisão agrupada), agora com uma divisão
  70/30 e intervalos de confiança.

**Próximo experimento (v2):** enriquecer o histórico do paciente, o único sinal que generalizou.
