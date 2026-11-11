"""Audita sobreposição externa/interna antes e depois do agrupamento de consultas."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from noshow.data import RAW_PATH, add_patient_proxy, load_raw
from noshow.evaluation import patient_groups, record_groups, repeated_group_folds


def main():
    raw = pd.read_csv(RAW_PATH)
    df = add_patient_proxy(load_raw())
    y = df.no_show.to_numpy()
    signatures = record_groups(raw)
    legacy = patient_groups(df.patient_id)
    corrected = patient_groups(df.patient_id, records=raw)
    rows = []
    for protocol, groups in [("anterior", legacy), ("sem_sobreposicao", corrected)]:
        for r, f, tr, te in repeated_group_folds(y, groups):
            _, _, it, iv = next(repeated_group_folds(y[tr], groups[tr], n_repeats=1, seed=42+r))
            outer = int(np.isin(signatures[te], signatures[tr]).sum())
            inner = int(np.isin(signatures[tr[iv]], signatures[tr[it]]).sum())
            known_train = set(df.patient_id.iloc[tr].dropna())
            known_test = set(df.patient_id.iloc[te].dropna())
            assert not known_train & known_test
            assert len(tr) + len(te) == len(df)
            if protocol == "sem_sobreposicao":
                assert outer == inner == 0
            rows.append(dict(protocolo=protocol, rep=r, fold=f, n_treino=len(tr), n_teste=len(te),
                             linhas_iguais_teste_treino=outer, linhas_iguais_validacao_treino_interno=inner,
                             pacientes_identificados_compartilhados=0))
    audit = pd.DataFrame(rows)
    root = RAW_PATH.parents[2] / "resultados/artigo_sem_sobreposicao"
    root.mkdir(parents=True, exist_ok=True)
    audit.to_csv(root / "auditoria_divisoes.csv", index=False)
    summary = dict(linhas=len(raw), grupos_anteriores=len(set(legacy)), grupos_corrigidos=len(set(corrected)),
                   assinatura="todos os campos originais exceto no_show e no_show_reason; inclui data/hora",
                   linhas_removidas=0, folds_externos=15, holdouts_internos=15,
                   sobreposicao_corrigida_externa=0, sobreposicao_corrigida_interna=0,
                   pacientes_identificados_compartilhados=0)
    (root / "auditoria_divisoes.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(audit.groupby("protocolo")[["linhas_iguais_teste_treino", "linhas_iguais_validacao_treino_interno"]]
          .agg(["min", "max"]).to_string())


if __name__ == "__main__":
    main()
