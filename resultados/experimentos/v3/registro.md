**Data:** 01/11/2026 · **Variáveis:** 33 (iguais à v2) · **ID:** aproximado (cobertura 78% → 89%)

| Divisão | Modelo | v2 → v3 (ROC-AUC) | Diferença [IC 95%] |
|---|---|---|---|
| Temporal | **LightGBM** | 0,500 → 0,552 | **+0,052 [+0,036; +0,069]** |
| Temporal | **Regressão logística** | 0,541 → 0,555 | **+0,014 [+0,004; +0,025]** |
| Temporal | RB (BIC) | 0,546 → 0,534 | −0,013 (n.s.) |
| Agrupada | todos | ±0,006 | sem mudança relevante |

**Leitura:**
- **A ligação aproximada é a primeira mudança com ganho claro na divisão temporal**: +0,05 de ROC-AUC e +0,027 de
  PR-AUC no LightGBM, e +0,014 na regressão logística. Mais pacientes com histórico melhoram justamente os modelos
  que combinam várias variáveis de histórico.
- **A Rede Bayesiana não se beneficiou.** O Manto (BIC) é uma única variável, e a ligação aproximada também junta
  pacientes diferentes (a chave colide em ~26% dos grupos), adicionando ruído ao histórico. Os modelos que combinam
  várias variáveis compensam melhor esse ruído.
- **Na divisão temporal, os três melhores empatam** (logística 0,555, LightGBM 0,552 e RB BIC v1 0,559), com ICs
  sobrepostos. Prever o futuro continua difícil para qualquer modelo com estes dados.
- Na divisão agrupada, a cobertura de ID já era alta nos anos sorteados, e nada muda.

**Balanço das três versões:**

| | Agrupada (melhor) | Temporal (melhor) |
|---|---|---|
| v1 | LightGBM 0,678 | RB (BIC) 0,559 |
| v2 | LightGBM 0,684 | RB (BIC) 0,546 |
| v3 | LightGBM 0,678 | logística 0,555 / LightGBM 0,552 |

**Por que não chegamos a 0,8:** com estas variáveis, o teto fica em ~0,68 (agrupada) e ~0,56 (temporal), e nenhum
algoritmo ou variável derivada do histórico mudou isso. O que falta é **informação que a base não tem**: a data de
marcação (antecedência), confirmação e lembretes, distância até o serviço, e o histórico dos pacientes sem cadastro
(em 2022, 71% dos registros não têm cadastro). Estudos que relatam ~0,8 ou mais costumam ter a antecedência entre os
preditores, ou validam com divisões aleatórias que misturam o mesmo paciente no treino e no teste.

**Próximos experimentos possíveis:**
- **v4:** recalibração periódica (reestimar no último ano antes de prever), atacando a deriva temporal;
- **v5:** variáveis externas, se a fonte dos dados puder fornecer a data de marcação.
