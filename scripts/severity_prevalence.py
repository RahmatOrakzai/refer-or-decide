"""Addendum C1-C2 (post hoc): severity-stratified discrimination and prevalence-adjusted predictive values.

Re-uses the out-of-fold predictions already written by run_baselines.py; fits nothing new.
  (1) AUC of each dementia severity band (MMSE 24-30, 20-23, 10-19, <10; CDR 0.5-1, >=2) against ALL
      development controls, with a participant-level bootstrap CI (2000 resamples, as in run_baselines.py).
  (2) Sensitivity/specificity at the 0.5 threshold and at the 90%-sensitivity threshold, converted to
      PPV/NPV across prevalences 2-49% by Bayes' rule, for all cases and for mild cases (MMSE >= 24).

Usage: python severity_prevalence.py <derived_dir> <results_dir> <out_dir>
"""
import os, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

der, res, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
T = "T-adj: roberta-base, pause tokens, adjusted"
F = "F-adj: features, demographics+interviewer removed"
m = pd.read_csv(f"{der}/pitt_master_recordings.csv", dtype={"media_id": str, "participant_id": str})
oof = pd.read_csv(f"{res}/baseline_oof_predictions.csv", dtype={"media_id": str})
d = oof.merge(m[["media_id", "participant_id", "label", "mmse", "cdr"]], on="media_id")
ctl = d[d.label == 0]


def boot(dem, col, B=2000, seed=0):
    sub = pd.concat([dem, ctl])
    rng = np.random.default_rng(seed)
    pid = sub.participant_id.unique()
    idx = {g: np.where(sub.participant_id.values == g)[0] for g in pid}
    v = []
    for _ in range(B):
        s = np.concatenate([idx[g] for g in rng.choice(pid, len(pid))])
        if len(np.unique(sub.label.values[s])) == 2:
            v.append(roc_auc_score(sub.label.values[s], sub[col].values[s]))
    return roc_auc_score(sub.label, sub[col]), np.percentile(v, 2.5), np.percentile(v, 97.5)


bands = [("MMSE 24-30 (mild/none)", lambda x: x.mmse >= 24), ("MMSE 20-23", lambda x: (x.mmse >= 20) & (x.mmse < 24)),
         ("MMSE 10-19", lambda x: (x.mmse >= 10) & (x.mmse < 20)), ("MMSE <10", lambda x: x.mmse < 10),
         ("CDR 0.5-1", lambda x: x.cdr <= 1), ("CDR >=2", lambda x: x.cdr >= 2)]
rows = []
for nm, fn in bands:
    dem = d[(d.label == 1) & fn(d)]
    if len(dem) < 5:
        continue
    for col, lab in [(T, "T-adj"), (F, "F-adj")]:
        a, lo, hi = boot(dem, col)
        rows.append(dict(stratum=nm, model=lab, n_dementia=len(dem), AUC=round(a, 3),
                         lo=round(lo, 3), hi=round(hi, 3), mean_p=round(dem[col].mean(), 3)))
sev = pd.DataFrame(rows)
sev.to_csv(f"{out}/severity_gradient_Tadj_Fadj.csv", index=False)

pT, pC = d[d.label == 1][T].values, ctl[T].values
pMild = d[(d.label == 1) & (d.mmse >= 24)][T].values
thr90 = np.percentile(pT, 10)
pv = []
for tname, thr in [("threshold 0.5", 0.5), (f"90% sens. threshold ({thr90:.2f})", thr90)]:
    for pop, dd in [("all dementia", pT), ("mild only (MMSE>=24)", pMild)]:
        se, sp = (dd >= thr).mean(), (pC < thr).mean()
        for prev in [0.02, 0.05, 0.10, 0.20, 0.49]:
            ppv = se * prev / (se * prev + (1 - sp) * (1 - prev))
            npv = sp * (1 - prev) / (sp * (1 - prev) + (1 - se) * prev)
            pv.append(dict(operating_point=tname, population=pop, sens=round(se, 3), spec=round(sp, 3),
                           prevalence=prev, PPV=round(ppv, 3), NPV=round(npv, 3)))
pd.DataFrame(pv).to_csv(f"{out}/prevalence_adjusted_predictive_values.csv", index=False)
print(sev.to_string(index=False))
print(pd.DataFrame(pv).to_string(index=False))
