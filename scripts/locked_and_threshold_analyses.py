"""Reviewer points 2.6, 2.8 and §4: locked-test AUC intervals, the one-visit vs all-visit gap, and predictive
values of the certified rule's actual thresholds (not the 0.5 threshold).
Usage: python locked_and_threshold_analyses.py <locked_predictions_csv> <master_csv> <baseline_oof_csv> <lambda>
"""
import sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
lp, master, oof, lam = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
L = pd.read_csv(lp, dtype={"media_id": str, "participant_id": str})
M = pd.read_csv(master, dtype={"media_id": str, "participant_id": str})[["media_id", "visit", "mmse"]]
L = L.merge(M, on="media_id")
rng = np.random.default_rng(0)

def boot(df, B=2000):
    pid = df.participant_id.unique(); ix = {g: np.where(df.participant_id.values == g)[0] for g in pid}; v = []
    for _ in range(B):
        s = np.concatenate([ix[g] for g in rng.choice(pid, len(pid))])
        if len(np.unique(df.label.values[s])) == 2:
            v.append(roc_auc_score(df.label.values[s], df.p.values[s]))
    return roc_auc_score(df.label, df.p), np.percentile(v, 2.5), np.percentile(v, 97.5)

one = L[L.in_one_visit_set]
print("locked one-visit AUC %.3f [%.3f, %.3f]" % boot(one), "n", len(one))
print("locked all-visit AUC %.3f [%.3f, %.3f]" % boot(L), "n", len(L), "participants", L.participant_id.nunique())
rest = L[~L.in_one_visit_set]
print("recordings not in the one-visit set: n", len(rest), "AUC %.3f" % roc_auc_score(rest.label, rest.p) if rest.label.nunique() == 2 else "")
first = L.loc[L.groupby("participant_id").visit.idxmin()]
print("locked first-visit AUC %.3f [%.3f, %.3f]" % boot(first))
w = 1 / L.groupby("participant_id").media_id.transform("count")
print("locked participant-weighted all-visit AUC %.3f" % roc_auc_score(L.label, L.p, sample_weight=w))
# within-participant label consistency and later-visit behaviour
L["late"] = L.visit > L.groupby("participant_id").visit.transform("min")
for nm, g in L.groupby("late"):
    print("later visits" if nm else "first visits", "n", len(g), "dementia share %.2f" % g.label.mean(),
          "mean p dementia %.2f controls %.2f" % (g.p[g.label == 1].mean(), g.p[g.label == 0].mean()))
# cross-pairs: AUC is lower when dementia later visits score low or control repeat visits score high
dem_later = L[(L.label == 1) & L.late]; ctl = L[L.label == 0]
print("dementia later-visit recordings with p<0.5: %d of %d" % ((dem_later.p < 0.5).sum(), len(dem_later)))

# predictive values of the certified rule's thresholds on development OOF predictions
O = pd.read_csv(oof, dtype={"media_id": str}).merge(pd.read_csv(master, dtype={"media_id": str, "participant_id": str})[["media_id", "participant_id", "visit", "label"]], on="media_id")
T = "T-adj: roberta-base, pause tokens, adjusted"
for unit, D in (("recording", O), ("participant (first visit)", O.loc[O.groupby("participant_id").visit.idxmin()])):
    p, y = D[T].values, D.label.values
    pos, neg = p >= lam, p <= 1 - lam
    a1, a0 = pos[y == 1].mean(), pos[y == 0].mean()          # P(auto-positive | y)
    n1, n0 = neg[y == 1].mean(), neg[y == 0].mean()          # P(auto-negative | y)
    print(f"[{unit}] lambda={lam}: dementia auto-flagged {a1:.3f}, cleared {n1:.3f}, referred {1-a1-n1:.3f}; "
          f"controls flagged {a0:.3f}, cleared {n0:.3f}, referred {1-a0-n0:.3f}")
    for pi in (0.05, 0.10, 0.49):
        ppv = pi * a1 / (pi * a1 + (1 - pi) * a0) if (a1 + a0) > 0 else np.nan
        npv = (1 - pi) * n0 / ((1 - pi) * n0 + pi * n1) if (n1 + n0) > 0 else np.nan
        print(f"   prevalence {pi:.2f}: PPV of automatic positives {ppv:.3f}, NPV of automatic negatives {npv:.3f}")
