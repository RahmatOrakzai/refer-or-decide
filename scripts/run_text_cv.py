"""Step-2 protocol (10 x participant-grouped 5-fold CV on development, inner C choice) for any extra input:
an embedding (npz:key -> T and T-adj with PCA(64)) or a CSV of features (csv:col1,col2,...).

Usage: python run_text_cv.py <derived_dir> <out_csv> <name> <npz:key | csv:file:cols>
Appends one row per model to out_csv and writes <out_csv minus .csv>_<name>_oof.csv.
"""
import os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

der, out, name, spec = sys.argv[1:5]
ids = {"media_id": str, "participant_id": str}
folds = pd.read_csv(f"{der}/protocol_folds.csv", dtype=ids)
m = pd.read_csv(f"{der}/pitt_master_recordings.csv", dtype=ids).drop(columns=["participant_id", "label", "visit"])
f = pd.read_csv(f"{der}/chat_features.csv", dtype={"media_id": str})
d = folds.merge(m, on="media_id").merge(f, on="media_id")
d = d[~d.locked_test].reset_index(drop=True)
if spec.startswith("csv:"):
    _, fn, cols = spec.split(":", 2)
    x = pd.read_csv(fn if fn.startswith("/") else f"{der}/{fn}", dtype={"media_id": str})
    d = d.merge(x, on="media_id", how="inner").reset_index(drop=True)
    X = d[cols.split(",")].values.astype(float)
else:
    fn, key = spec.split(":")
    z = np.load(f"{der}/{fn}", allow_pickle=True)
    idx = pd.Series(range(len(z["media_id"])), index=z["media_id"])
    d = d[d.media_id.isin(idx.index)].reset_index(drop=True)
    X = z[key][idx[d.media_id].values]
d["male"] = (d.sex == "male").astype(float)
y, groups = d.label.values, d.participant_id.values
Z = d[["age_at_visit", "education_years", "male", "inv_turns", "inv_words", "inv_turns_after_start", "inv_word_share",
       "inv_prompts_anything_else", "inv_encouragement"]].values.astype(float)
if os.environ.get("COVARS") == "demo":
    Z = Z[:, :3]   # age, education, sex only
reduce = X.shape[1] > 100


def pipe(C):
    s = [SimpleImputer(strategy="median"), StandardScaler()] + ([PCA(64, random_state=0)] if reduce else [])
    return make_pipeline(*s, LogisticRegression(C=C, max_iter=3000))


def fit_predict(Xtr, ytr, gtr, Xte):
    best, bestC = -1, 1
    for C in [0.01, 0.1, 1, 10]:
        sc = [roc_auc_score(ytr[b], pipe(C).fit(Xtr[a], ytr[a]).predict_proba(Xtr[b])[:, 1])
              for a, b in StratifiedGroupKFold(3, shuffle=True, random_state=0).split(Xtr, ytr, gtr)]
        if np.mean(sc) > best:
            best, bestC = np.mean(sc), C
    return pipe(bestC).fit(Xtr, ytr).predict_proba(Xte)[:, 1]


def resid(tr):
    Xa = SimpleImputer(strategy="median").fit(X[tr]).transform(X)
    Zi = SimpleImputer(strategy="median").fit(Z[tr]).transform(Z)
    ctl = tr[y[tr] == 0]
    return Xa - LinearRegression().fit(Zi[ctl], Xa[ctl]).predict(Zi)


def boot(p, B=2000):
    pid = np.unique(groups); ix = {g: np.where(groups == g)[0] for g in pid}; rng = np.random.default_rng(0); v = []
    for _ in range(B):
        s = np.concatenate([ix[g] for g in rng.choice(pid, len(pid))])
        if len(np.unique(y[s])) == 2:
            v.append(roc_auc_score(y[s], p[s]))
    return f"{roc_auc_score(y, p):.3f} [{np.percentile(v, 2.5):.3f}, {np.percentile(v, 97.5):.3f}]"


rows, oof = [], {"media_id": d.media_id}
models = [("plain", False)] + ([("adj", True)] if not spec.startswith("csv:") else [])
for tag, adj in models:
    P = np.zeros((10, len(d)))
    for r in range(10):
        fc = d[f"fold_r{r}"].values
        for k in range(5):
            tr, te = np.where(fc != k)[0], np.where(fc == k)[0]
            Xu = resid(tr) if adj else X
            P[r, te] = fit_predict(Xu[tr], y[tr], groups[tr], Xu[te])
        print(name, tag, "repeat", r, round(roc_auc_score(y, P[r]), 3), flush=True)
    p = P.mean(0); oof[f"{name}_{tag}"] = p
    rows.append({"input": name, "model": "T-adj" if adj else ("T" if not spec.startswith("csv:") else "features"),
                 "n_recordings": len(d), "AUC [95% CI]": boot(p)})
pd.DataFrame(rows).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
pd.DataFrame(oof).to_csv(out.replace(".csv", f"_{name}_oof.csv"), index=False)
print(pd.DataFrame(rows).to_string(index=False))
