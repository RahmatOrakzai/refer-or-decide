"""Step 2 of the protocol: discrimination baselines and the shortcut ledger on the development set.

10 repeats x participant-grouped 5-fold CV (folds from derived/protocol_folds.csv). Inner grouped 3-fold CV picks C.
Writes per-recording out-of-fold predictions (averaged over repeats) and a results table.

Usage: python run_baselines.py <derived_dir> <out_dir>
"""
import os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss

der, out_dir = sys.argv[1:3]
os.makedirs(out_dir, exist_ok=True)
ids = {"media_id": str, "participant_id": str}
m = pd.read_csv(f"{der}/pitt_master_recordings.csv", dtype=ids)
f = pd.read_csv(f"{der}/chat_features.csv", dtype={"media_id": str})
folds = pd.read_csv(f"{der}/protocol_folds.csv", dtype=ids)
sp = pd.read_csv(f"{der}/pitt_splits.csv", dtype=ids)[["media_id", "first_visit_only", "age_matched"]]
d = folds.merge(m.drop(columns=["participant_id", "label", "visit"]), on="media_id").merge(f, on="media_id").merge(sp, on="media_id")
d = d[(d.transcript_found == True) & (~d.locked_test)].reset_index(drop=True)
d["male"] = (d.sex == "male").astype(float)
y = d.label.values
groups = d.participant_id.values

DEMO = ["age_at_visit", "education_years", "male"]
INV = ["inv_turns", "inv_words", "inv_turns_after_start", "inv_word_share", "inv_prompts_anything_else", "inv_encouragement"]
PAR = ["par_words", "par_ttr", "par_mlu_words", "par_filled_pauses", "par_fragments", "par_repetitions",
       "par_retracings", "par_unintelligible", "par_info_units", "par_info_unit_tokens", "par_words_per_min",
       "par_pauses_gt_0_5s", "par_pauses_gt_2s", "par_mean_pause_s", "par_phonation_ratio", "par_span_s"]

X = {"F": d[PAR].values.astype(float), "DEMO": d[DEMO].values.astype(float),
     "INV": d[INV].values.astype(float), "LEN": d[["par_span_s"]].values.astype(float)}
for enc in ["bert-base-uncased", "roberta-base"]:
    z = np.load(f"{der}/emb_{enc}.npz", allow_pickle=True)
    idx = pd.Series(range(len(z["media_id"])), index=z["media_id"])
    for col, tag in [("text_pause", "pause"), ("text_plain", "plain")]:
        X[f"T[{enc},{tag}]"] = z[col][idx[d.media_id].values]
Z = d[DEMO + INV].values.astype(float)
Cs = [0.01, 0.1, 1, 10]


def fit_predict(Xtr, ytr, gtr, Xte, reduce=False):
    """Inner grouped CV over C; returns test probabilities."""
    def pipe(C):
        steps = [SimpleImputer(strategy="median"), StandardScaler()]
        if reduce:
            steps.append(PCA(n_components=min(64, len(Xtr) - 1), random_state=0))
        return make_pipeline(*steps, LogisticRegression(C=C, max_iter=3000))
    best, bestC = -1, 1
    for C in Cs:
        sc = []
        for a, b in StratifiedGroupKFold(3, shuffle=True, random_state=0).split(Xtr, ytr, gtr):
            sc.append(roc_auc_score(ytr[b], pipe(C).fit(Xtr[a], ytr[a]).predict_proba(Xtr[b])[:, 1]))
        if np.mean(sc) > best:
            best, bestC = np.mean(sc), C
    return pipe(bestC).fit(Xtr, ytr).predict_proba(Xte)[:, 1]


def residualise(Xa, tr):
    """Regress each column on demographics + interviewer covariates using training controls only."""
    Xa = SimpleImputer(strategy="median").fit(Xa[tr]).transform(Xa)
    Zi = SimpleImputer(strategy="median").fit(Z[tr]).transform(Z)
    ctl = tr[y[tr] == 0]
    lr = LinearRegression().fit(Zi[ctl], Xa[ctl])
    return Xa - lr.predict(Zi)


REPEAT_AUCS = {}


def run(name, key, adj=False, leaky=False, reps=10):
    P = np.zeros((reps, len(d)))
    Xa = X[key]
    for r in range(reps):
        if leaky:
            splits = list(StratifiedKFold(5, shuffle=True, random_state=r).split(Xa, y))
        else:
            fc = d[f"fold_r{r}"].values
            splits = [(np.where(fc != k)[0], np.where(fc == k)[0]) for k in range(5)]
        for tr, te in splits:
            Xu = residualise(Xa, tr) if adj else Xa
            P[r, te] = fit_predict(Xu[tr], y[tr], groups[tr], Xu[te], reduce=Xa.shape[1] > 100)
        print(f"  {name} repeat {r}: AUC {roc_auc_score(y, P[r]):.3f}", flush=True)
    REPEAT_AUCS[name] = [roc_auc_score(y, P[r]) for r in range(reps)]
    return P


def boot_auc(p, sel=None, B=2000, seed=0):
    sel = np.ones(len(d), bool) if sel is None else sel
    dd = pd.DataFrame({"g": groups[sel], "y": y[sel], "p": p[sel]})
    pid = dd.g.unique(); byg = {g: x for g, x in dd.groupby("g")}
    rng = np.random.default_rng(seed); vals = []
    for _ in range(B):
        s = pd.concat([byg[g] for g in rng.choice(pid, len(pid))])
        if s.y.nunique() == 2:
            vals.append(roc_auc_score(s.y, s.p))
    return roc_auc_score(dd.y, dd.p), np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def ece(p, yy, bins=10):
    e = 0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        s = (p >= lo) & (p < lo + 1 / bins)
        if s.any():
            e += s.mean() * abs(p[s].mean() - yy[s].mean())
    return e


runs = [("Demographics only (negative control)", "DEMO", False, False),
        ("Interviewer only (negative control)", "INV", False, False),
        ("Speech length only (negative control)", "LEN", False, False),
        ("F: participant features", "F", False, False),
        ("F: participant features, leaky recording-level split", "F", False, True),
        ("F-adj: features, demographics+interviewer removed", "F", True, False)]
for enc in ["bert-base-uncased", "roberta-base"]:
    runs += [(f"T: {enc}, pause tokens", f"T[{enc},pause]", False, False),
             (f"T: {enc}, no pause tokens", f"T[{enc},plain]", False, False),
             (f"T-adj: {enc}, pause tokens, adjusted", f"T[{enc},pause]", True, False)]

preds = {}
for name, key, adj, leaky in runs:
    print(name, flush=True)
    preds[name] = run(name, key, adj, leaky).mean(0) if not leaky else run(name, key, adj, leaky)

# late fusion: average of logits of F and the best T (by mean AUC), plain and adjusted
def logit(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))
tnames = [n for n in preds if n.startswith("T: ") and "pause" in n and "no pause" not in n]
best_t = max(tnames, key=lambda n: roc_auc_score(y, preds[n]))
preds["F+T fusion (" + best_t[3:] + ")"] = 1 / (1 + np.exp(-(logit(preds["F: participant features"]) + logit(preds[best_t])) / 2))
best_ta = best_t.replace("T: ", "T-adj: ").replace(", pause tokens", ", pause tokens, adjusted")
preds["F-adj+T-adj fusion"] = 1 / (1 + np.exp(-(logit(preds["F-adj: features, demographics+interviewer removed"]) + logit(preds[best_ta])) / 2))

rep_rows = []
first = d.first_visit_only.astype(str).eq("True").values
matched = d.age_matched.astype(str).eq("True").values
age = d.age_at_visit.values; edu = d.education_years.values
rows = []
for name, P in preds.items():
    p = P.mean(0) if P.ndim == 2 else P
    rep_aucs = [roc_auc_score(y, q) for q in (P if P.ndim == 2 else [P])]
    a, lo, hi = boot_auc(p)
    row = {"model": name, "AUC (all visits)": f"{a:.3f} [{lo:.3f}, {hi:.3f}]",
           "AUC first visit": f"{roc_auc_score(y[first], p[first]):.3f}",
           "AUC age/edu-matched": f"{roc_auc_score(y[matched], p[matched]):.3f}",
           "AUC age>=70": f"{roc_auc_score(y[age >= 70], p[age >= 70]):.3f}",
           "AUC age<70": f"{roc_auc_score(y[age < 70], p[age < 70]):.3f}",
           "AUC edu<=12": f"{roc_auc_score(y[edu <= 12], p[edu <= 12]):.3f}",
           "AUC edu>12": f"{roc_auc_score(y[edu > 12], p[edu > 12]):.3f}",
           "Brier": f"{brier_score_loss(y, p):.3f}", "ECE": f"{ece(p, y):.3f}"}
    rows.append(row)
    rep_rows.append({"model": name, **{f"rep{k}": v for k, v in enumerate(REPEAT_AUCS.get(name, rep_aucs))}})
R = pd.DataFrame(rows)
R.to_csv(f"{out_dir}/baselines.csv", index=False)
pd.DataFrame(rep_rows).to_csv(f"{out_dir}/baseline_repeat_aucs.csv", index=False)
pd.DataFrame({"media_id": d.media_id, **{k: (v.mean(0) if v.ndim == 2 else v) for k, v in preds.items()}}).to_csv(f"{out_dir}/baseline_oof_predictions.csv", index=False)

# paired participant-bootstrap AUC differences
def paired(n1, n2, B=2000):
    p1 = preds[n1] if preds[n1].ndim == 1 else preds[n1].mean(0)
    p2 = preds[n2] if preds[n2].ndim == 1 else preds[n2].mean(0)
    pid = np.unique(groups); idx = {g: np.where(groups == g)[0] for g in pid}
    rng = np.random.default_rng(1); diffs = []
    for _ in range(B):
        s = np.concatenate([idx[g] for g in rng.choice(pid, len(pid))])
        if len(np.unique(y[s])) == 2:
            diffs.append(roc_auc_score(y[s], p1[s]) - roc_auc_score(y[s], p2[s]))
    dlt = roc_auc_score(y, p1) - roc_auc_score(y, p2)
    return f"{dlt:+.3f} [{np.percentile(diffs, 2.5):+.3f}, {np.percentile(diffs, 97.5):+.3f}]"

comps = [("F: participant features", "F-adj: features, demographics+interviewer removed"),
         ("F: participant features, leaky recording-level split", "F: participant features"),
         (best_t, best_ta),
         (best_t, best_t.replace("pause tokens", "no pause tokens")),
         ("F+T fusion (" + best_t[3:] + ")", "F: participant features"),
         ("F-adj+T-adj fusion", "F-adj: features, demographics+interviewer removed")]
C = pd.DataFrame([{"comparison": f"{a}  minus  {b}", "AUC difference [95% CI]": paired(a, b)} for a, b in comps])
C.to_csv(f"{out_dir}/baseline_comparisons.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)
print(R.to_string(index=False)); print(C.to_string(index=False))
