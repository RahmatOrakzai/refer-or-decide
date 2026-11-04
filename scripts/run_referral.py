"""Step 3 of the protocol: participant-level referral layer.

For each of 200 participant-level resplits (50% train / 25% calibration / 25% test, one random visit per
participant in calibration and test):
  (a) class-conditional split conformal sets (refer if not a singleton), marginal and Mondrian by education;
  (b) Learn-then-Test selective classification: decide only when confidence max(p, 1-p) >= lambda, where
      lambda is the smallest threshold whose binomial-tail p-value for "error among decided > alpha" is <= delta,
      tested in fixed sequence from the most to the least confident threshold. Run overall and per stratum.

Usage: python run_referral.py <derived_dir> <out_dir> [model: F-adj | F | T-adj | T]
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

der, out_dir = sys.argv[1:3]
MODEL = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else "F-adj"
OPT = dict(a[2:].split("=", 1) for a in sys.argv if a.startswith("--") and "=" in a)
EMB = OPT.get("emb", "emb_roberta-base.npz:text_pause")   # --emb=<npz in derived>:<key> for T models
TAG = OPT.get("tag", MODEL)                               # --tag=<name used in output files>
KDIV = int(OPT.get("kdiv", 1))                             # --kdiv=<extra Bonferroni factor, e.g. number of k options>
COVARS = OPT.get("covars", "demo+inv")                     # --covars=demo: adjust for age, education, sex only (no interviewer counts)
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
XF, Z = d[PAR].values.astype(float), d[DEMO + (INV if COVARS == "demo+inv" else [])].values.astype(float)
if MODEL.startswith("T"):  # T-adj: roberta-base with pause tokens, embedding dimensions residualised, PCA(64)
    zt = np.load(f"{der}/{EMB.split(':')[0]}", allow_pickle=True)
    XF = zt[EMB.split(":")[1]][pd.Series(range(len(zt["media_id"])), index=zt["media_id"])[d.media_id].values]
edu_hi = (d.education_years.values > 12)
age_hi = (d.age_at_visit.values >= 70)
STRATA = {"education <=12": ~edu_hi, "education >12": edu_hi, "age <70": ~age_hi, "age >=70": age_hi}
ALPHAS, DELTA = [0.05, 0.10, 0.15, 0.20], 0.10


def train_scores(tr):
    Xa = SimpleImputer(strategy="median").fit(XF[tr]).transform(XF)
    if MODEL in ("F-adj", "T-adj"):
        Zi = SimpleImputer(strategy="median").fit(Z[tr]).transform(Z)
        ctl = tr[y[tr] == 0]
        Xa = Xa - LinearRegression().fit(Zi[ctl], Xa[ctl]).predict(Zi)
    def pipe(C):
        steps = [StandardScaler()] + ([PCA(64, random_state=0)] if Xa.shape[1] > 100 else [])
        return make_pipeline(*steps, LogisticRegression(C=C, max_iter=3000))
    best, bestC = -1, 1
    g = d.participant_id.values
    for C in [0.01, 0.1, 1, 10]:
        sc = [roc_auc_score(y[tr][b], pipe(C).fit(Xa[tr][a], y[tr][a]).predict_proba(Xa[tr][b])[:, 1])
              for a, b in StratifiedGroupKFold(3, shuffle=True, random_state=0).split(Xa[tr], y[tr], g[tr])]
        if np.mean(sc) > best:
            best, bestC = np.mean(sc), C
    return pipe(bestC).fit(Xa[tr], y[tr]).predict_proba(Xa)[:, 1]


def conformal_sets(p, cal, te, alpha, mondrian=None):
    """Class-conditional (optionally per Mondrian stratum) split conformal. Returns include0, include1 for te."""
    inc0 = np.zeros(len(te), bool); inc1 = np.zeros(len(te), bool)
    cells = [np.ones(len(d), bool)] if mondrian is None else [mondrian, ~mondrian]
    for cell in cells:
        c_cal, t_sel = cal[cell[cal]], cell[te]
        for k, inc in ((0, inc0), (1, inc1)):
            s = np.sort(np.where(k == 1, 1 - p, p)[c_cal][y[c_cal] == k])  # nonconformity of class k
            n = len(s); j = int(np.ceil((n + 1) * (1 - alpha)))
            q = s[j - 1] if j <= n else np.inf
            inc[t_sel] = (np.where(k == 1, 1 - p, p)[te][t_sel] <= q)
    return inc0, inc1


GRID = np.round(np.arange(0.95, 0.5, -0.05), 2)  # fixed, data-independent confidence thresholds


def ltt_lambda(conf, err, alpha, mode="bonferroni"):
    """Lowest confidence threshold whose binomial-tail p-value for H0: risk > alpha is rejected.
    mode='fixed-sequence': test GRID from high to low, stop at the first non-rejection (FWER delta).
    mode='bonferroni': test every GRID point at delta/len(GRID), take the lowest rejected (FWER delta)."""
    lam_ok = None
    for lam in GRID:
        s = conf >= lam
        n, k = s.sum(), err[s].sum()
        pval = binom.cdf(k, n, alpha) if n else 1.0
        if mode == "bonferroni":
            if pval <= DELTA / len(GRID):
                lam_ok = lam
        elif mode == "bonferroni-joint":
            if pval <= DELTA / (len(GRID) * KDIV):
                lam_ok = lam
        elif pval <= DELTA:
            lam_ok = lam
        else:
            break
    return lam_ok


REPS = 0 if "--defs-only" in sys.argv else 200  # --defs-only: import the definitions (used by run_locked_test.py)
rows, curve = [], []
pids = d.participant_id.unique()
for rep in range(REPS):
    rng = np.random.default_rng(rep)
    perm = rng.permutation(pids); n = len(perm)
    trp, calp = set(perm[: n // 2]), set(perm[n // 2: 3 * n // 4])
    tr = np.where(d.participant_id.isin(trp))[0]
    one = d.groupby("participant_id").sample(1, random_state=rep).index.values
    cal = np.array([i for i in one if d.participant_id[i] in calp])
    te = np.array([i for i in one if d.participant_id[i] not in trp and d.participant_id[i] not in calp])
    p = train_scores(tr)
    pred = (p >= 0.5).astype(int); conf = np.maximum(p, 1 - p); err = (pred != y)
    base = {"rep": rep, "auc_test": roc_auc_score(y[te], p[te]), "n_cal": len(cal), "n_test": len(te)}
    for lam in GRID:  # oracle view: realised test risk at each fixed threshold (not used for any decision)
        s_ = conf[te] >= lam
        curve.append({"rep": rep, "lambda": lam, "auto_rate": s_.mean(), "err_auto": err[te][s_].mean() if s_.any() else np.nan})
    for alpha in ALPHAS:
        # (a) conformal
        for mname, mond in (("conformal", None), ("conformal-mondrian-edu", edu_hi)):
            i0, i1 = conformal_sets(p, cal, te, alpha, mond)
            cov = np.where(y[te] == 1, i1, i0); single = i0 ^ i1
            r = {**base, "method": mname, "alpha": alpha, "auto_rate": single.mean(),
                 "err_auto": (single & ~cov).sum() / max(single.sum(), 1),
                 "cov_dementia": cov[y[te] == 1].mean(), "cov_control": cov[y[te] == 0].mean()}
            for sn, sm in STRATA.items():
                r[f"cov[{sn}]"] = cov[sm[te]].mean()
            rows.append(r)
        # (b) LTT selective, overall and per stratum (education-specific thresholds)
        for mname, cells, mode in (("LTT-fixed-sequence", {"all": np.ones(len(d), bool)}, "fixed-sequence"),
                                   ("LTT-bonferroni", {"all": np.ones(len(d), bool)}, "bonferroni"),
                                   ("LTT-bonferroni-per-education", {"lo": ~edu_hi, "hi": edu_hi}, "bonferroni")) + \
                ((("LTT-bonferroni-joint-k", {"all": np.ones(len(d), bool)}, "bonferroni-joint"),) if KDIV > 1 else ()):
            auto = np.zeros(len(te), bool)
            for cell in cells.values():
                c_cal = cal[cell[cal]]
                lam = ltt_lambda(conf[c_cal], err[c_cal], alpha, mode)
                if lam is not None:
                    auto |= cell[te] & (conf[te] >= lam)
            e = err[te]
            r = {**base, "method": mname, "alpha": alpha, "auto_rate": auto.mean(),
                 "err_auto": e[auto].mean() if auto.any() else 0.0,
                 "sens_auto": (~e[auto & (y[te] == 1)]).mean() if (auto & (y[te] == 1)).any() else np.nan,
                 "spec_auto": (~e[auto & (y[te] == 0)]).mean() if (auto & (y[te] == 0)).any() else np.nan}
            for sn, sm in STRATA.items():
                a2 = auto & sm[te]
                r[f"auto[{sn}]"] = a2.sum() / max(sm[te].sum(), 1)
                r[f"err_auto[{sn}]"] = e[a2].mean() if a2.any() else np.nan
            rows.append(r)
    if rep % 25 == 0:
        print("resplit", rep, flush=True)

if REPS:
    R = pd.DataFrame(rows)
    R.to_csv(f"{out_dir}/referral_resplits_{TAG}.csv", index=False)
    pd.DataFrame(curve).to_csv(f"{out_dir}/referral_risk_curve_{TAG}.csv", index=False)
    agg = []
    for (meth, alpha), g in R.groupby(["method", "alpha"]):
        row = {"method": meth, "alpha": alpha, "auto_rate": g.auto_rate.mean(), "err_auto_mean": g.err_auto.mean(),
               "P(err_auto > alpha)": (g.err_auto > alpha).mean()}
        for c in g.columns:
            if c.startswith(("cov", "sens", "spec", "auto[", "err_auto[")):
                row[c] = g[c].mean()
        agg.append(row)
    A = pd.DataFrame(agg)
    A.to_csv(f"{out_dir}/referral_summary_{TAG}.csv", index=False)
    pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40)
    print(f"mean test AUC {R.drop_duplicates('rep').auc_test.mean():.3f}")
    print(A.round(3).to_string(index=False))
