"""Class-specific Learn-then-Test risk control for the referral layer, with severity reporting.

Extends the project's run_referral.py in two ways:
  (1) Two thresholds certified for two DIFFERENT budgets: the miss rate among automatic NEGATIVE
      decisions (false-omission rate, FOR) at alpha_fn, and the false-discovery rate among automatic
      POSITIVE decisions (FDR) at alpha_fp. Each is a separate Bonferroni LTT over the same fixed
      lambda grid; delta is split over the two families (delta/(2*|grid|)), so the pair is valid at delta.
  (2) Realised risk and automatic rate are also reported within MMSE severity bands of the test set.

Everything else (participant-level resplits, one visit per participant in calibration and test,
residualisation on training controls, inner grouped CV for C) follows run_referral.py exactly.

Usage: python run_referral_classwise.py <derived_dir> <out_dir> <F-adj|T-adj> [--emb=npz:key]
"""
import os, sys
import numpy as np, pandas as pd
from scipy.stats import binom
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

der, out_dir, MODEL = sys.argv[1:4]
OPT = dict(a[2:].split("=", 1) for a in sys.argv if a.startswith("--") and "=" in a)
EMB = OPT.get("emb", "emb_roberta-base.npz:text_pause")
os.makedirs(out_dir, exist_ok=True)
ids = {"media_id": str, "participant_id": str}
m = pd.read_csv(f"{der}/pitt_master_recordings.csv", dtype=ids)
f = pd.read_csv(f"{der}/chat_features.csv", dtype={"media_id": str})
folds = pd.read_csv(f"{der}/protocol_folds.csv", dtype=ids)[["media_id", "locked_test"]]
d = m.merge(f, on="media_id").merge(folds, on="media_id")
d = d[d.transcript_found == True].reset_index(drop=True)
d["male"] = (d.sex == "male").astype(float)
y = d.label.values
DEMO = ["age_at_visit", "education_years", "male"]
INV = ["inv_turns", "inv_words", "inv_turns_after_start", "inv_word_share", "inv_prompts_anything_else", "inv_encouragement"]
PAR = ["par_words", "par_ttr", "par_mlu_words", "par_filled_pauses", "par_fragments", "par_repetitions",
       "par_retracings", "par_unintelligible", "par_info_units", "par_info_unit_tokens", "par_words_per_min",
       "par_pauses_gt_0_5s", "par_pauses_gt_2s", "par_mean_pause_s", "par_phonation_ratio", "par_span_s"]
XF, Z = d[PAR].values.astype(float), d[DEMO + INV].values.astype(float)
if MODEL.startswith("T"):
    zt = np.load(f"{der}/{EMB.split(':')[0]}", allow_pickle=True)
    XF = zt[EMB.split(":")[1]][pd.Series(range(len(zt["media_id"])), index=zt["media_id"])[d.media_id].values]
mmse = d.mmse.values
SEV = {"MMSE >=24": mmse >= 24, "MMSE 20-23": (mmse >= 20) & (mmse < 24), "MMSE <20": mmse < 20}
GRID = np.round(np.arange(0.95, 0.5, -0.05), 2)
DELTA = 0.10
ALPHA_FN = [0.05, 0.10, 0.20]      # budgets for the miss rate among automatic negatives
ALPHA_FP = [0.20, 0.30]            # budgets for the false-discovery rate among automatic positives


def train_scores(tr):
    Xa = SimpleImputer(strategy="median").fit(XF[tr]).transform(XF)
    if MODEL in ("F-adj", "T-adj"):
        Zi = SimpleImputer(strategy="median").fit(Z[tr]).transform(Z)
        ctl = tr[y[tr] == 0]
        Xa = Xa - LinearRegression().fit(Zi[ctl], Xa[ctl]).predict(Zi)
    def pipe(C):
        steps = [StandardScaler()] + ([PCA(64, random_state=0)] if Xa.shape[1] > 100 else [])
        return make_pipeline(*steps, LogisticRegression(C=C, max_iter=3000))
    best, bestC, g = -1, 1, d.participant_id.values
    for C in [0.01, 0.1, 1, 10]:
        sc = [roc_auc_score(y[tr][b], pipe(C).fit(Xa[tr][a], y[tr][a]).predict_proba(Xa[tr][b])[:, 1])
              for a, b in StratifiedGroupKFold(3, shuffle=True, random_state=0).split(Xa[tr], y[tr], g[tr])]
        if np.mean(sc) > best:
            best, bestC = np.mean(sc), C
    return pipe(bestC).fit(Xa[tr], y[tr]).predict_proba(Xa)[:, 1]


def certify(p_cal, y_cal, alpha, side, n_families=2):
    """Lowest grid threshold whose one-sided binomial test rejects 'risk > alpha' at delta/(n_families*|grid|).

    side='neg': automatic negatives are p <= 1-lam; risk = P(y=1 | decided control)  (false-omission rate)
    side='pos': automatic positives are p >= lam;   risk = P(y=0 | decided dementia) (false-discovery rate)
    """
    lam_ok = None
    for lam in GRID:
        s = (p_cal <= 1 - lam) if side == "neg" else (p_cal >= lam)
        n = int(s.sum())
        k = int((y_cal[s] == 1).sum()) if side == "neg" else int((y_cal[s] == 0).sum())
        if n and binom.cdf(k, n, alpha) <= DELTA / (n_families * len(GRID)):
            lam_ok = lam
    return lam_ok


rows = []
pids = d.participant_id.unique()
for rep in range(200):
    rng = np.random.default_rng(rep)
    perm = rng.permutation(pids); n = len(perm)
    trp, calp = set(perm[: n // 2]), set(perm[n // 2: 3 * n // 4])
    tr = np.where(d.participant_id.isin(trp))[0]
    one = d.groupby("participant_id").sample(1, random_state=rep).index.values
    cal = np.array([i for i in one if d.participant_id[i] in calp])
    te = np.array([i for i in one if d.participant_id[i] not in trp and d.participant_id[i] not in calp])
    p = train_scores(tr)
    base = {"rep": rep, "auc_test": roc_auc_score(y[te], p[te]), "n_cal": len(cal), "n_test": len(te)}
    for a_fn in ALPHA_FN:
        for a_fp in ALPHA_FP:
            lam_n = certify(p[cal], y[cal], a_fn, "neg")
            lam_p = certify(p[cal], y[cal], a_fp, "pos")
            autoneg = (p[te] <= 1 - lam_n) if lam_n is not None else np.zeros(len(te), bool)
            autopos = (p[te] >= lam_p) if lam_p is not None else np.zeros(len(te), bool)
            auto = autoneg | autopos
            r = {**base, "alpha_fn": a_fn, "alpha_fp": a_fp, "lambda_neg": lam_n, "lambda_pos": lam_p,
                 "auto_rate": auto.mean(), "auto_neg_rate": autoneg.mean(), "auto_pos_rate": autopos.mean(),
                 "FOR": (y[te][autoneg] == 1).mean() if autoneg.any() else np.nan,
                 "FDR": (y[te][autopos] == 0).mean() if autopos.any() else np.nan,
                 "sens_auto": autopos[y[te] == 1].sum() / max((y[te] == 1).sum(), 1),
                 "err_auto_pooled": ((y[te][autoneg] == 1).sum() + (y[te][autopos] == 0).sum()) / max(auto.sum(), 1)
                 if auto.any() else np.nan}
            for sn, sm in SEV.items():     # severity of the dementia cases in this test set
                dem_s = sm[te] & (y[te] == 1)
                r[f"missed[{sn}]"] = (autoneg & dem_s).sum() / max(dem_s.sum(), 1) if dem_s.any() else np.nan
                r[f"caught[{sn}]"] = (autopos & dem_s).sum() / max(dem_s.sum(), 1) if dem_s.any() else np.nan
            rows.append(r)
    if rep % 25 == 0:
        print("resplit", rep, flush=True)

R = pd.DataFrame(rows)
R.to_csv(f"{out_dir}/classwise_resplits_{MODEL}.csv", index=False)
agg = []
for (a_fn, a_fp), g in R.groupby(["alpha_fn", "alpha_fp"]):
    row = {"model": MODEL, "alpha_fn": a_fn, "alpha_fp": a_fp,
           "auto_rate": g.auto_rate.mean(), "auto_neg_rate": g.auto_neg_rate.mean(),
           "auto_pos_rate": g.auto_pos_rate.mean(),
           "FOR_mean": g.FOR.mean(), "P(FOR>alpha_fn)": (g.FOR.fillna(0) > a_fn).mean(),
           "FDR_mean": g.FDR.mean(), "P(FDR>alpha_fp)": (g.FDR.fillna(0) > a_fp).mean(),
           "never_certified_neg": g.lambda_neg.isna().mean(), "never_certified_pos": g.lambda_pos.isna().mean()}
    for c in g.columns:
        if c.startswith(("missed[", "caught[")):
            row[c] = g[c].mean()
    agg.append(row)
A = pd.DataFrame(agg)
A.to_csv(f"{out_dir}/classwise_summary_{MODEL}.csv", index=False)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40)
print(A.round(3).to_string(index=False))
