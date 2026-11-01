"""Gera o notebook de um experimento registrado em `noshow.registry.EXPERIMENTOS`.

Uso: .venv/bin/python scripts/novo_experimento.py vN

Cria `notebooks/05_experimento_vN.ipynb`. Depois de executar o notebook, escreva a análise em
`resultados/experimentos/vN/registro.md` e rode o script de novo para incluí-la na seção "Registro do experimento".

Atenção: o notebook é recriado sem saídas. Reexecute-o depois (com RECALCULAR = False ele só recarrega os
resultados salvos, o bootstrap e o SHAP, em ~2 min).
"""
import sys
from pathlib import Path

import nbformat as nbf

from noshow.registry import EXPERIMENTOS, RAIZ, RESULTADOS

versao = sys.argv[1]
saida = RAIZ / "notebooks" / f"05_experimento_{versao}.ipynb"
registro = RESULTADOS / versao / "registro.md"
conclusao = registro.read_text() if registro.exists() else (
    f"> _Registro pendente: escrever a análise em `resultados/experimentos/{versao}/registro.md`._")
cfg = EXPERIMENTOS[versao]
versoes = list(EXPERIMENTOS)
anterior = versoes[versoes.index(versao) - 1] if versoes.index(versao) > 0 else None

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(f"""
# 05 — Experimento {versao}: {cfg['titulo']}

**O que muda nesta versão:** {cfg['mudanca']}

{"**Versão anterior:** `05_experimento_" + anterior + ".ipynb`" if anterior else "**Primeira versão** da série de experimentos 70/30."}

## Protocolo (igual em todas as versões)

- **Duas divisões 70/30**, as mesmas em todas as versões, para as comparações serem pareadas:
  - **agrupada:** pacientes sorteados até o teste reunir 30% dos registros; nenhum paciente nos dois lados;
  - **temporal:** os 70% de registros mais antigos para treino, e os 30% mais recentes para teste (corte em set/2019).
    `periodo` fica de fora, porque é constante no treino.
- **Quatro modelos:**
  - Rede Bayesiana reduzida ao Manto de Markov, com critério BIC e com AIC;
  - **regressão logística** (one-hot + padronização; regularização escolhida na validação interna);
  - **LightGBM** (30 tentativas do Optuna na validação interna).
- **Validação interna só com os 70% de treino:** CV agrupada (3 folds) na divisão agrupada; os últimos 25% do
  período de treino na divisão temporal.
- **Incerteza:** intervalos de 95% por **bootstrap reamostrando pacientes** (1.000 reamostragens) e diferenças
  pareadas entre modelos nas mesmas reamostragens.

Código: `src/noshow/registry.py` (entrada `"{versao}"`), resultados em `resultados/experimentos/{versao}/`.
""")

code(f"""
import json
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from noshow.benchmark import grouped_bootstrap, paired_difference, to_lgbm
from noshow.registry import EXPERIMENTOS, MODELOS, RESULTADOS, load, load_bases, run, splits

warnings.filterwarnings("ignore")
assert os.environ.get("PYTHONHASHSEED") == "0", "Use o kernel do projeto (PYTHONHASHSEED=0); ver README."

VERSAO = "{versao}"
ANTERIOR = {repr(anterior)}
RECALCULAR = False  # True refaz o experimento (~5 min); False reaproveita resultados/experimentos/{versao}/

if RECALCULAR or not (RESULTADOS / VERSAO / "predicoes.parquet").exists():  # predições não são versionadas
    M, P = run(VERSAO)
else:
    M, P = load(VERSAO)

BLUE, ORANGE, GRAY, INK, GREEN = "#2a78d6", "#eb6834", "#8a8984", "#52514e", "#1baf7a"
COR = {{"RB (BIC)": ORANGE, "RB (AIC)": "#eda100", "Regressão logística": BLUE, "LightGBM": GREEN}}
plt.rcParams.update({{
    "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": GRAY, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
}})
print(f"Experimento {{VERSAO}}: {{EXPERIMENTOS[VERSAO]['titulo']}}")
""")

md("## 1. Resultados no teste (30%)")

code("""
tabela = M.set_index(["divisao", "modelo"]).loc[[(d, m) for d in ["agrupada", "temporal"] for m in MODELOS]]
tabela[["n_variaveis", "roc_auc", "pr_auc", "brier", "p_media", "auc_interna", "manto"]].style.format(
    {"roc_auc": "{:.4f}", "pr_auc": "{:.4f}", "brier": "{:.4f}", "p_media": "{:.1%}", "auc_interna": "{:.4f}"},
    na_rep="—")
""")

md("""
## 2. Intervalos de confiança (bootstrap por paciente)

Uma única divisão 70/30 dá um único número por modelo. O bootstrap estima o quanto esse número variaria com outra
amostra de pacientes: reamostra os pacientes do teste com reposição e recalcula as métricas 1.000 vezes.
""")

code("""
BOOT = {}
for divisao in ["agrupada", "temporal"]:
    sub = P[P.divisao == divisao]
    base_ = sub[sub.modelo == MODELOS[0]].sort_values("linha")
    preds = {m: sub[sub.modelo == m].sort_values("linha").p.to_numpy() for m in MODELOS}
    BOOT[divisao] = grouped_bootstrap(base_.y.to_numpy(), preds, base_.grupo.to_numpy(), n_boot=1000)

ic = pd.DataFrame([
    {"divisao": d, "modelo": m, "métrica": met, "média": BOOT[d][f"{m}|{met}"].mean(),
     "ic_inf": np.quantile(BOOT[d][f"{m}|{met}"], 0.025), "ic_sup": np.quantile(BOOT[d][f"{m}|{met}"], 0.975)}
    for d in BOOT for m in MODELOS for met in ["roc_auc", "pr_auc"]])

fig, axes = plt.subplots(2, 2, figsize=(13, 5.5), sharey=True)
for (i, divisao), (j, met) in [(a, b) for a in enumerate(["agrupada", "temporal"]) for b in enumerate(["roc_auc", "pr_auc"])]:
    ax = axes[i, j]
    for k, m in enumerate(MODELOS[::-1]):
        r = ic[(ic.divisao == divisao) & (ic.modelo == m) & (ic["métrica"] == met)].iloc[0]
        ax.plot([r.ic_inf, r.ic_sup], [k, k], color=COR[m], lw=3, solid_capstyle="round")
        ax.plot(r["média"], k, "o", color=COR[m], ms=8, mec="white", mew=2)
        ax.text(r.ic_sup, k, f"  {r['média']:.3f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(MODELOS)), MODELOS[::-1])
    ax.set_title(f"{divisao} — {'ROC-AUC' if met == 'roc_auc' else 'PR-AUC'}")
    if met == "roc_auc":
        ax.axvline(0.5, color=GRAY, ls="--", lw=1)
    ax.grid(axis="y", visible=False)
fig.suptitle("Estimativa e IC 95% no teste", x=0.01, ha="left", fontweight="bold")
plt.tight_layout(); plt.show()
""")

md("""
**Comparação com a Rede Bayesiana (BIC)**, o modelo mais parcimonioso do TCC: diferença pareada de ROC-AUC (outro
modelo − RB BIC), com IC 95%.
""")

code("""
comp = pd.DataFrame([{"divisao": d, "modelo": m, **paired_difference(BOOT[d], m, "RB (BIC)", "roc_auc")}
                     for d in BOOT for m in MODELOS[1:]]).set_index(["divisao", "modelo"])
comp.style.format({"diferença": "{:+.4f}", "ic_inf": "{:+.4f}", "ic_sup": "{:+.4f}", "p": "{:.3f}"}) \\
    .map(lambda v: "font-weight: bold" if v < 0.05 else "", subset=["p"])
""")

if anterior:
    md(f"""
## 3. Comparação com a versão anterior ({anterior})

O mesmo modelo na mesma divisão e com **os mesmos registros de teste**: a diferença mede só o efeito da mudança
desta versão.
""")
    code("""
_, P_ant = load(ANTERIOR)
linhas = []
for divisao in ["agrupada", "temporal"]:
    for m in MODELOS:
        a = P[(P.divisao == divisao) & (P.modelo == m)].sort_values("linha")
        b = P_ant[(P_ant.divisao == divisao) & (P_ant.modelo == m)].sort_values("linha")
        assert (a.linha.to_numpy() == b.linha.to_numpy()).all()
        boot = grouped_bootstrap(a.y.to_numpy(), {"atual": a.p.to_numpy(), "anterior": b.p.to_numpy()},
                                 a.grupo.to_numpy(), n_boot=1000)
        for met in ["roc_auc", "pr_auc"]:
            linhas.append({"divisao": divisao, "modelo": m, "métrica": met,
                           "anterior": boot[f"anterior|{met}"].mean(), "atual": boot[f"atual|{met}"].mean(),
                           **paired_difference(boot, "atual", "anterior", met)})
evolucao = pd.DataFrame(linhas).set_index(["divisao", "modelo", "métrica"])
evolucao.style.format({"anterior": "{:.4f}", "atual": "{:.4f}", "diferença": "{:+.4f}", "ic_inf": "{:+.4f}",
                       "ic_sup": "{:+.4f}", "p": "{:.3f}"}) \\
    .map(lambda v: "font-weight: bold" if v < 0.05 else "", subset=["p"])
""")

md(f"""
## {3 + bool(anterior)}. Evolução das versões registradas

Melhor ROC-AUC de cada versão já executada (todas com as mesmas divisões).
""")

code("""
todas = pd.concat([load(v)[0] for v in EXPERIMENTOS if (RESULTADOS / v / "metricas.csv").exists()
                   and list(EXPERIMENTOS).index(v) <= list(EXPERIMENTOS).index(VERSAO)])
evol = todas.pivot_table(index=["divisao", "modelo"], columns="versao", values="roc_auc")
evol.style.format("{:.4f}").highlight_max(axis=1, color="#d6e6f8")
""")

md(f"""
## {4 + bool(anterior)}. O que os modelos usam

- **Rede Bayesiana:** o Manto de Markov (coluna `manto` na Seção 1) é o modelo inteiro.
- **LightGBM:** importância média das variáveis pelo **SHAP** (contribuição média absoluta para o log-odds),
  reajustado no treino com os hiperparâmetros escolhidos pelo Optuna.
""")

code("""
import lightgbm as lgb
import shap

cfg = EXPERIMENTOS[VERSAO]
bases = load_bases()
base = bases[cfg["base"]]
parametros = json.loads((RESULTADOS / VERSAO / "parametros.json").read_text())["modelos"]
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (divisao, (tr, te)) in zip(axes, splits(bases).items()):
    cat = [c for c in cfg["categoricas"] if not (divisao == "temporal" and c == "periodo")]
    cols = cat + cfg["continuas"]
    Xtr, categorias = to_lgbm(base[cols].iloc[tr], cat)
    hp = {k: v for k, v in parametros[f"{divisao}|LightGBM"].items() if k != "auc_interna"}
    modelo = lgb.LGBMClassifier(**hp, random_state=42, verbose=-1).fit(Xtr, base.no_show.to_numpy()[tr])
    Xte = to_lgbm(base[cols].iloc[te], cat, categorias)[0]
    valores = shap.TreeExplainer(modelo).shap_values(Xte.sample(3000, random_state=0))
    valores = valores[1] if isinstance(valores, list) else valores
    imp = pd.Series(np.abs(valores).mean(axis=0), index=cols).sort_values().tail(12)
    ax.barh(imp.index, imp.values, color=GREEN, height=0.6)
    ax.set_title(f"LightGBM — {divisao}: |SHAP| médio"); ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=8)
plt.tight_layout(); plt.show()
""")

md(f"## {5 + bool(anterior)}. Registro do experimento")
md(conclusao)

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "tcc-noshow", "display_name": "TCC no-show (.venv)", "language": "python"}
nbf.write(nb, saida)
print(f"Gerado: {saida}")
