"""Experimentos do artigo (01–04), com pré-processamento isolado por partição.

Uso: PYTHONHASHSEED=0 .venv/bin/python scripts/run_h3.py [temporal] [cv]
Resultados novos ficam em resultados/artigo_sem_sobreposicao/, preservando os históricos.
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

from noshow.data import RAW_PATH, add_patient_proxy, load_raw
from noshow.evaluation import patient_groups, repeated_group_folds
from noshow.experiments import run_config
from noshow.features import (CATEGORICAL, CONTINUOUS, add_age, add_calendar,
                             add_patient_history, add_service_time, group_categories)
from noshow.pipeline import raw_matrix
from noshow.weather import fix_weather

RAIZ = Path(__file__).resolve().parents[1]
CONFIGS = {
    "C0": ("C0 bruto", "raw", "sem", False),
    "C0_smote": ("C0 bruto + SMOTENC (robust)", "raw", "robust", True),
    "C1": ("C1 engenharia", "eng", "sem", False),
    "C3_sem": ("C3 engenharia + SMOTENC (sem escalonador)", "eng", "sem", True),
    "C3_minmax": ("C3 engenharia + SMOTENC (minmax)", "eng", "minmax", True),
    "C3_standard": ("C3 engenharia + SMOTENC (standard)", "eng", "standard", True),
    "C3_robust": ("C3 engenharia + SMOTENC (robust)", "eng", "robust", True),
}
CLIMA = {"average_temp_day", "max_temp_day", "average_rain_day", "max_rain_day",
         "rain_intensity", "heat_intensity", "rainy_day_before", "storm_day_before"}


def load_inputs():
    """Refaz somente as variáveis básicas de 01–02, sem importar variáveis da v7."""
    raw = pd.read_csv(RAW_PATH)
    df = add_patient_proxy(fix_weather(load_raw()))
    for transform in (add_age, add_calendar, group_categories, add_service_time, add_patient_history):
        df = transform(df)
    dates = pd.to_datetime(raw.appointment_date, format="%d/%m/%Y")
    if not dates.reset_index(drop=True).equals(df.appointment_date.reset_index(drop=True)):
        raise ValueError("A ordem dos agendamentos mudou durante a engenharia.")
    if not np.array_equal(raw.no_show.eq("yes").astype(int), df.no_show):
        raise ValueError("Os desfechos deixaram de estar alinhados.")
    df["_validation_group"] = patient_groups(df.patient_id, records=raw)
    X_raw, raw_cat, raw_cont = raw_matrix(raw)
    # city é auxiliar: permite aprender city_grp no treino, sem o agrupamento global da EDA.
    return df, {"raw": (X_raw, raw_cat, raw_cont),
                "eng": (df[CATEGORICAL + CONTINUOUS + ["city"]], CATEGORICAL, CONTINUOUS)}


def inner_split(y, groups, dates, temporal, seed):
    if temporal:
        cut = dates.sort_values().iloc[int(0.75 * len(dates))]
        return np.where(dates < cut)[0], np.where(dates >= cut)[0]
    _, _, tr, va = next(repeated_group_folds(y, groups, n_splits=5, n_repeats=1, seed=seed))
    return tr, va


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("etapas", nargs="*", help="temporal e/ou cv; padrão: ambas")
    parser.add_argument("--saida", type=Path)
    parser.add_argument("--max-folds", type=int, help="Piloto: limite de folds; não produz resultados finais")
    parser.add_argument("--configs", nargs="+", choices=CONFIGS, default=list(CONFIGS))
    parser.add_argument("--sem-clima", action="store_true", help="Retira clima observado dos dois braços")
    parser.add_argument("--sem-ajuste-ess", action="store_true", help="Usa ESS=1 em vez de seleção interna por AP")
    parser.add_argument("--sem-reparo-sinteticos", action="store_true")
    parser.add_argument("--sem-correcao-priori", action="store_true")
    parser.add_argument("--sampling-strategy", type=float, default=1.0)
    parser.add_argument("--k-neighbors", type=int, default=5)
    parser.add_argument("--n-bins", type=int, default=5)
    parser.add_argument("--protocolo", choices=["anterior", "ajustado"], default="ajustado")
    parser.add_argument("--n-bins-grid", nargs="+", type=int, default=[3, 5])
    parser.add_argument("--sampling-grid", nargs="+", type=float, default=[0.25, 0.5, 1.0])
    parser.add_argument("--criterio-cv", choices=["bic", "aic"], default="bic",
                        help="BIC segue a proposta do artigo; AIC é análise de sensibilidade")
    args = parser.parse_args()
    etapas = args.etapas or ["temporal", "cv"]
    if set(etapas) - {"temporal", "cv"}:
        parser.error("Etapas válidas: temporal e cv.")
    if os.environ.get("PYTHONHASHSEED") != "0":
        parser.error("Defina PYTHONHASHSEED=0 para resultados reprodutíveis.")
    if args.max_folds is not None and not 1 <= args.max_folds <= 15:
        parser.error("--max-folds deve estar entre 1 e 15.")
    if not 0 < args.sampling_strategy <= 1 or args.k_neighbors < 1 or args.n_bins < 2:
        parser.error("Proporção deve estar em (0,1], vizinhos >= 1 e n-bins >= 2.")
    pilot = args.max_folds is not None or set(args.configs) != set(CONFIGS)
    folder = "artigo_sem_sobreposicao" if args.protocolo == "ajustado" else "artigo_sem_sobreposicao_anterior"
    saida = args.saida or RAIZ / "resultados" / folder / ("piloto" if pilot else "completo")
    settings = {k: v for k, v in vars(args).items() if k not in {"etapas", "saida", "max_folds"}}
    settings.update(ess_grid=[1.0] if args.sem_ajuste_ess else [1.0, 10.0, 100.0], cv_scoring=f"{args.criterio_cv}-d", temporal_scoring=["aic-d", "bic-d"],
                    grouping="componentes de paciente reconstruído e campos originais iguais, sem alvo/motivo",
                    inner_validation="um holdout agrupado ou últimos 25% do treino temporal",
                    history="sequencial: desfechos anteriores disponíveis antes de cada consulta",
                    pr_auc="sklearn.metrics.average_precision_score")
    manifest_path = saida / "protocolo.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if manifest and manifest["settings"] != settings:
        parser.error("A pasta já contém outro protocolo. Escolha outra --saida para preservar a comparação.")
    saida.mkdir(parents=True, exist_ok=True)
    tracked = list((RAIZ / "src" / "noshow").glob("*.py")) + [Path(__file__).resolve()]
    provenance = dict(python=platform.python_version(),
                      source_sha256={str(p.relative_to(RAIZ)): hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked},
                      data_sha256=hashlib.sha256(RAW_PATH.read_bytes()).hexdigest(),
                      external_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in (RAIZ / "data" / "external").glob("*.csv")},
                      packages={p: importlib.metadata.version(p) for p in
                                ["pandas", "numpy", "scikit-learn", "imbalanced-learn", "pgmpy"]})
    if manifest and any(manifest.get(k) != v for k, v in provenance.items()):
        parser.error("Código, dados ou ambiente mudaram. Escolha outra --saida para não misturar execuções.")
    manifest.update(settings=settings, pilot=pilot, **provenance)
    manifest.setdefault("stages", {})
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    df, bases = load_inputs()
    y, groups = df.no_show.to_numpy(), df._validation_group.to_numpy()
    print(f"{len(df):,} linhas | saída: {saida} | piloto: {pilot}", flush=True)

    def evaluate(tr, te, temporal=False, scoring=None, seed=42):
        scoring = scoring or f"{args.criterio_cv}-d"
        adjusted = args.protocolo == "ajustado"
        split = None if args.sem_ajuste_ess and not adjusted else inner_split(y[tr], groups[tr], df.appointment_date.iloc[tr], temporal, seed)
        for code in args.configs:
            name, base, scaler, smote = CONFIGS[code]
            X, cat, cont = bases[base]
            exclude = ({"appointment_year", "periodo"} if temporal else set()) | (CLIMA if args.sem_clima else set())
            cat, cont = [c for c in cat if c not in exclude], [c for c in cont if c not in exclude]
            row = run_config(name, X.iloc[tr], y[tr], X.iloc[te], y[te], cat, cont, scaler, smote,
                             scoring=scoring, seed=seed, inner_split=split, n_bins=args.n_bins,
                             sampling_strategy=args.sampling_strategy, k_neighbors=args.k_neighbors,
                             repair_derived=not args.sem_reparo_sinteticos,
                             correct_prior=not args.sem_correcao_priori,
                             ess_grid=(1.0,) if args.sem_ajuste_ess else (1.0, 10.0, 100.0),
                             n_bins_grid=args.n_bins_grid if adjusted else None,
                             sampling_grid=args.sampling_grid if adjusted else None,
                             coherent_history=adjusted)
            print(f"  {name}: AP={row['pr_auc']:.4f} AUC={row['roc_auc']:.4f} ESS={row['ess']:g}", flush=True)
            yield row

    for stage in etapas:
        rows, started = [], time.time()
        filename = f"h3_{'validacao_cruzada' if stage == 'cv' else 'validacao_temporal'}{'_piloto' if pilot else ''}.csv"
        manifest["stages"][stage] = {"complete": False, "file": filename}
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        if stage == "cv":
            for i, (r, f, tr, te) in enumerate(repeated_group_folds(y, groups)):
                if args.max_folds is not None and i >= args.max_folds:
                    break
                print(f"CV repetição {r}, fold {f}", flush=True)
                rows.extend({"rep": r, "fold": f, **row} for row in evaluate(tr, te, seed=42+r))
                pd.DataFrame(rows).to_csv(saida / filename, index=False)
        else:
            tr = np.where(df.appointment_date.dt.year <= 2020)[0]
            te = np.where(df.appointment_date.dt.year >= 2021)[0]
            for scoring in ["aic-d", "bic-d"]:
                rows.extend({"criterio": scoring[:3].upper(), **row}
                            for row in evaluate(tr, te, temporal=True, scoring=scoring))
                pd.DataFrame(rows).to_csv(saida / filename, index=False)
        manifest["stages"][stage].update(complete=not pilot, rows=len(rows), seconds=time.time()-started)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
