"""Comparação pareada anterior × ajustado, em 15 folds e dois testes temporais.

Prepare uma vez, distribua jobs independentes, depois consolide:
  PYTHONHASHSEED=0 python scripts/compare_article.py --prepare
  PYTHONHASHSEED=0 python scripts/compare_article.py --jobs 3 --job-index 0
  PYTHONHASHSEED=0 python scripts/compare_article.py --collect
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from noshow.data import RAW_PATH
from noshow.evaluation import patient_groups, repeated_group_folds
from noshow.experiments import run_config
from run_h3 import CONFIGS, RAIZ, inner_split, load_inputs

PROFILES = ("anterior", "ajustado")
METRICS = ("pr_auc", "roc_auc", "brier", "recall@top10", "precision@top10")


def provenance():
    files = sorted((RAIZ / "src/noshow").glob("*.py")) + [Path(__file__).resolve(),
             RAIZ / "scripts/run_h3.py", RAIZ / "scripts/audit_article_splits.py"]
    return dict(source_sha256={str(p.relative_to(RAIZ)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                data_sha256=hashlib.sha256(RAW_PATH.read_bytes()).hexdigest(),
                external_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted((RAIZ / "data/external").glob("*.csv"))},
                python=platform.python_version(), packages={p: importlib.metadata.version(p) for p in
                    ["numpy", "pandas", "scikit-learn", "imbalanced-learn", "pgmpy"]})


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                   default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + "\n")
    temporary.replace(path)


def task_names():
    return [f"cv_r{r}_f{f}" for r in range(3) for f in range(5)] + ["temporal_bic", "temporal_aic"]


def prepare(root):
    current = provenance()
    path = root / "protocolo.json"
    if path.exists():
        if json.loads(path.read_text())["provenance"] != current:
            raise ValueError("Código/dados mudaram. Use outra --saida; resultados não serão misturados.")
        return
    root.mkdir(parents=True, exist_ok=True)
    protocol = dict(provenance=current, tasks=task_names(), primary_metric="average_precision",
                    cv="StratifiedGroupKFold, 5 folds × 3 repetições, sementes 42–44, BIC",
                    grouping="componentes de paciente reconstruído e campos originais iguais, sem alvo/motivo",
                    temporal="treino até 2020, teste 2021–2022; BIC principal e AIC sensibilidade",
                    inner="um holdout do treino: agrupado de 20% na CV; últimos 25% no temporal",
                    ess=[1.0, 10.0, 100.0], baseline=dict(n_bins=5, sampling_strategy=1.0, coherent_history=False),
                    adjusted=dict(n_bins_grid=[3, 5], sampling_grid=[0.25, 0.5, 1.0], coherent_history=True),
                    shared=dict(k_neighbors=5, repair_derived=True, correct_prior=True, observed_weather=True),
                    selection="maior AP interna; empate pela ordem predefinida da grade",
                    budget="10%; k=ceil(0.1*n); acertos esperados sob desempate uniforme",
                    history="sequencial, usando desfechos de dias anteriores a cada consulta",
                    interpretation="avaliação exploratória; períodos já usados no desenvolvimento")
    write_json(path, protocol)
    for relative in current["source_sha256"]:
        target = root / "codigo" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((RAIZ / relative).read_bytes())


def validate(root):
    protocol = json.loads((root / "protocolo.json").read_text())
    if protocol["provenance"] != provenance():
        raise ValueError("Código/dados diferem do protocolo preparado. Use outra --saida.")
    return protocol


def run_job(root, jobs, job_index):
    validate(root)
    df, bases = load_inputs()
    y, groups = df.no_show.to_numpy(), df._validation_group.to_numpy()
    cv = list(repeated_group_folds(y, groups))
    for task in task_names()[job_index::jobs]:
        output = root / f"{task}.json"
        if output.exists():
            print(f"{task}: já concluído", flush=True)
            continue
        temporal = task.startswith("temporal")
        if temporal:
            tr = np.where(df.appointment_date.dt.year <= 2020)[0]
            te = np.where(df.appointment_date.dt.year >= 2021)[0]
            scoring, seed = task.split("_")[1] + "-d", 42
            info = dict(etapa="temporal", criterio=scoring[:3].upper())
        else:
            r, f = int(task[4]), int(task[7])
            _, _, tr, te = cv[r * 5 + f]
            scoring, seed = "bic-d", 42 + r
            info = dict(etapa="cv", rep=r, fold=f, criterio="BIC")
        split = inner_split(y[tr], groups[tr], df.appointment_date.iloc[tr], temporal, seed)
        if temporal:
            assert df.appointment_date.iloc[tr].max() < df.appointment_date.iloc[te].min()
        else:
            assert not set(groups[tr]) & set(groups[te])
        rows, start = [], time.time()
        print(f"INÍCIO {task}", flush=True)
        for code, (name, base, scaler, smote) in CONFIGS.items():
            X, cat, cont = bases[base]
            cat = [c for c in cat if not temporal or c not in {"periodo", "appointment_year"}]
            for profile in PROFILES:
                adjusted = profile == "ajustado"
                row = run_config(name, X.iloc[tr], y[tr], X.iloc[te], y[te], cat, cont, scaler, smote,
                                 scoring=scoring, seed=seed, inner_split=split, repair_derived=True, correct_prior=True,
                                 coherent_history=adjusted, n_bins_grid=(3, 5) if adjusted else None,
                                 sampling_grid=(0.25, 0.5, 1.0) if adjusted else None)
                rows.append({**info, "perfil": profile, "codigo": code, **row})
                print(f"{task} {code:11s} {profile:8s}: AP={row['pr_auc']:.4f} "
                      f"R10={row['recall@top10']:.4f} bins={row['n_bins']} ratio={row['sampling_strategy']} "
                      f"ESS={row['ess']:g}", flush=True)
                write_json(root / f"{task}_partial.json", rows)
        write_json(output, rows)
        print(f"FIM {task}: {time.time()-start:.0f}s", flush=True)


def collect(root):
    protocol = validate(root)
    missing = [t for t in task_names() if not (root / f"{t}.json").exists()]
    if missing:
        raise ValueError(f"Tarefas ainda incompletas: {missing}")
    rows = [row for task in task_names() for row in json.loads((root / f"{task}.json").read_text())]
    data = pd.DataFrame(rows)
    assert len(data) == 238 and not data.duplicated(["etapa", "criterio", "rep", "fold", "perfil", "codigo"]).any()
    selections = [{k: row[k] for k in ["etapa", "criterio", "perfil", "codigo", "selecao_interna"]}
                  | {k: row[k] for k in ["rep", "fold"] if k in row} for row in rows]
    write_json(root / "selecao_interna.json", selections)
    data = data.drop(columns="selecao_interna")
    data.to_csv(root / "metricas_completas.csv", index=False)
    cv = data[data.etapa == "cv"]
    paired = cv[cv.perfil == "anterior"].merge(cv[cv.perfil == "ajustado"], on=["codigo", "rep", "fold"],
                                              suffixes=("_anterior", "_ajustado"), validate="one_to_one")
    summary = []
    for code, group in paired.groupby("codigo", sort=False):
        for metric in METRICS:
            a, b = group[metric+"_anterior"], group[metric+"_ajustado"]
            delta = b-a
            better = -delta if metric == "brier" else delta
            summary.append(dict(codigo=code, metrica=metric, anterior=a.mean(), ajustado=b.mean(),
                                dp_anterior=a.std(), dp_ajustado=b.std(), diferenca=delta.mean(),
                                folds_melhores=int((better > 1e-12).sum()),
                                folds_iguais=int((better.abs() <= 1e-12).sum()), pares=len(group)))
    summary = pd.DataFrame(summary)
    summary.to_csv(root / "comparacao_cv.csv", index=False)
    temp = data[data.etapa == "temporal"]
    temporal = temp[temp.perfil == "anterior"].merge(temp[temp.perfil == "ajustado"],
                on=["codigo", "criterio"], suffixes=("_anterior", "_ajustado"), validate="one_to_one")
    temporal.to_csv(root / "comparacao_temporal.csv", index=False)
    params = cv.groupby(["perfil", "codigo", "n_bins", "sampling_strategy", "ess"], dropna=False).size().reset_index(name="folds")
    params.to_csv(root / "parametros_escolhidos.csv", index=False)
    for profile in PROFILES:
        out = root.parent / ("completo" if profile == "ajustado" else "anterior")
        out.mkdir(parents=True, exist_ok=True)
        mine = data[data.perfil == profile]
        mine[mine.etapa == "cv"].to_csv(out / "h3_validacao_cruzada.csv", index=False)
        mine[mine.etapa == "temporal"].to_csv(out / "h3_validacao_temporal.csv", index=False)
        write_json(out / "protocolo.json", dict(settings={"cv_scoring": "bic-d", "perfil": profile,
                   "grouping": protocol["grouping"],
                   "comparison_protocol": str(root / "protocolo.json")}, pilot=False, provenance=protocol["provenance"],
                   stages={"cv": dict(complete=True, rows=105, file="h3_validacao_cruzada.csv"),
                           "temporal": dict(complete=True, rows=14, file="h3_validacao_temporal.csv")}))
    print(summary[summary.metrica == "pr_auc"].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saida", type=Path, default=RAIZ / "resultados/artigo_sem_sobreposicao/comparacao")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--job-index", type=int, default=0)
    args = parser.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        parser.error("Defina PYTHONHASHSEED=0.")
    if args.jobs < 1 or not 0 <= args.job_index < args.jobs:
        parser.error("Use jobs >= 1 e 0 <= job-index < jobs.")
    if args.prepare:
        prepare(args.saida)
    elif args.collect:
        collect(args.saida)
    else:
        run_job(args.saida, args.jobs, args.job_index)


if __name__ == "__main__":
    main()
