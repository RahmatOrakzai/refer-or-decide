"""Validate regenerated embeddings: reproduce the T-adj development AUC with run_baselines.py's exact
procedure (per-fold residualisation on training controls, inner grouped CV for C, PCA(64), 10 repeats),
and compare against the saved out-of-fold predictions.

Usage: python validate_embeddings.py <derived_dir> <results_dir>
"""
import sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

der, res = sys.argv[1:3]
ids = {"media_id": str, "participant_id": str}
m = pd.read_csv(f"{der}/pitt_master_recordings.csv", dtype=ids)
f = pd.read_csv(f"{der}/chat_features.csv", dtype={"media_id": str})
folds = pd.read_csv(f"{der}/protocol_folds.csv", dtype=ids)
sp = pd.read_csv(f"{der}/pitt_splits.csv", dtype=ids)[["media_id", "first_visit_only", "age_matched"]]
d = folds.merge(m.drop(columns=["participant_id", "label", "visit"]), on="media_id").merge(f, on="media_id").merge(sp, on="media_id")
d = d[(d.transcript_found == True) & (~d.locked_test)].reset_index(drop=True)
d["male"] = (d.sex == "male").astype(float)
y = d.label.values; groups = d.participant_id.values
DEMO = ["age_at_visit", "education_years", "male"]
INV = ["inv_turns", "inv_words", "inv_turns_after_start", "inv_word_share", "inv_prompts_anything_else", "inv_encouragement"]
Z = d[DEMO + INV].values.astype(float)
z = np.load(f"{der}/emb_roberta-base.npz", allow_pickle=True)
idx = pd.Series(range(len(z["media_id"])), index=z["media_id"])
X = z["text_pause"][idx[d.media_id].values]
Cs = [0.01, 0.1, 1, 10]


def fit_predict(Xtr, ytr, gtr, Xte, reduce=False):
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
    Xa = SimpleImputer(strategy="median").fit(Xa[tr]).transform(Xa)
    Zi = SimpleImputer(strategy="median").fit(Z[tr]).transform(Z)
    ctl = tr[y[tr] == 0]
    return Xa - LinearRegression().fit(Zi[ctl], Xa[ctl]).predict(Zi)


P = np.zeros((10, len(d)))
for r in range(10):
    fc = d[f"fold_r{r}"].values
    for k in range(5):
        tr, te = np.where(fc != k)[0], np.where(fc == k)[0]
        Xu = residualise(X, tr)
        P[r, te] = fit_predict(Xu[tr], y[tr], groups[tr], Xu[te], reduce=True)
    print(f"  repeat {r}: AUC {roc_auc_score(y, P[r]):.4f}", flush=True)
pd.DataFrame({"repeat": range(10), "auc": [roc_auc_score(y, q) for q in P]}).to_csv(
    f"{res}/baseline_repeat_aucs_Tadj.csv", index=False)
p_new = P.mean(0)
auc_new = roc_auc_score(y, p_new)
oof = pd.read_csv(f"{res}/baseline_oof_predictions.csv", dtype={"media_id": str})
col = "T-adj: roberta-base, pause tokens, adjusted"
p_pub = d[["media_id"]].merge(oof[["media_id", col]], on="media_id")[col].values
print(f"\\nPublished T-adj AUC (from saved OOF): {roc_auc_score(y, p_pub):.4f}")
print(f"Reproduced T-adj AUC (new embeddings): {auc_new:.4f}")
print(f"Pearson r between prediction vectors : {np.corrcoef(p_new, p_pub)[0,1]:.4f}")
print(f"Mean |difference| in predictions     : {np.abs(p_new - p_pub).mean():.4f}")
print(f"Per-repeat AUC mean {np.mean([roc_auc_score(y, q) for q in P]):.4f} (vs averaged {auc_new:.4f})")
