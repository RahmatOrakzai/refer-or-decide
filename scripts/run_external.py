"""Step 5: external evaluation on Lu (dementia vs control) and Delaware (MCI vs control, Cookie Theft segment).

Model: RoBERTa-base mean-pooled embedding of main-tier participant text (no timings), PCA(64), logistic regression,
trained on Pitt development participants only. Pitt's locked test is not used.
  (1) Discrimination: model trained on all of Pitt development; external AUC with participant bootstrap CI.
  (2) Referral transfer: 200 random 60/40 participant splits of Pitt development (train / calibrate, one visit per
      participant in calibration); lambda certified on Pitt (LTT, Bonferroni over the grid) and applied unchanged.
  (3) Recalibration: lambda re-certified on a random half of the external participants, evaluated on the other half.
Secondary (Delaware only): T-adj with age, education, sex and 3 timing-free interviewer covariates residualised.

Usage: python run_external.py <derived_dir> <out_dir> [pitt_npz:key]
"""
import os, sys
import numpy as np, pandas as pd
from scipy.stats import binom
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

der, out = sys.argv[1:3]
PITT_EMB = sys.argv[3] if len(sys.argv) > 3 else "emb_roberta-base_pittmain.npz:text"   # npz:key of the Pitt training text
os.makedirs(out, exist_ok=True)
GRID, DELTA, ALPHAS = np.round(np.arange(0.95, 0.5, -0.05), 2), 0.10, [0.10, 0.15, 0.20]
ids = {"media_id": str, "participant_id": str}


def load(name, emb, key="text"):
    t = pd.read_csv(f"{der}/{name}", dtype=ids)
    z = np.load(f"{der}/{emb}", allow_pickle=True)
    X = z[key][pd.Series(range(len(z["media_id"])), index=z["media_id"])[t.media_id].values]
    t["age"] = pd.to_numeric(t.age.astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    t["male"] = (t.sex.astype(str).str.lower() == "male").astype(float)
    return t.reset_index(drop=True), X


P, XP = load("pitt_main_text.csv", *PITT_EMB.split(":"))
folds = pd.read_csv(f"{der}/protocol_folds.csv", dtype=ids)[["media_id", "locked_test"]]
m = pd.read_csv(f"{der}/pitt_master_recordings.csv", dtype=ids)[["media_id", "age_at_visit", "sex"]]
P = P.drop(columns=["age"]).merge(m, on="media_id").merge(folds, on="media_id").rename(columns={"age_at_visit": "age"})
P["male"] = (P.sex_y.astype(str).str.lower() == "male").astype(float)
dev = ~P.locked_test.astype(str).eq("True").values
XP, P = XP[dev], P[dev].reset_index(drop=True)
EXT = {"Lu": load("lu_text.csv", "emb_roberta-base_lu.npz"), "Delaware": load("delaware_text.csv", "emb_roberta-base_delaware.npz")}
ZC = ["age", "education_years", "male", "inv_turns", "inv_words", "inv_word_share"]


def pipe(C):
    return make_pipeline(StandardScaler(), PCA(64, random_state=0), LogisticRegression(C=C, max_iter=3000))


def fit(X, y, g):
    best, bestC = -1, 1
    for C in [0.01, 0.1, 1, 10]:
        sc = [roc_auc_score(y[b], pipe(C).fit(X[a], y[a]).predict_proba(X[b])[:, 1])
              for a, b in StratifiedGroupKFold(3, shuffle=True, random_state=0).split(X, y, g)]
        if np.mean(sc) > best:
            best, bestC = np.mean(sc), C
    return pipe(bestC).fit(X, y)


def residualiser(tr_idx):
    """Fit on Pitt training controls: embedding ~ covariates. Returns a function applied to (X, frame)."""
    imp = SimpleImputer(strategy="median").fit(P[ZC].values[tr_idx])
    ctl = tr_idx[P.label.values[tr_idx] == 0]
    lr = LinearRegression().fit(imp.transform(P[ZC].values[ctl]), XP[ctl])
    return lambda X, F: X - lr.predict(imp.transform(F[ZC].values.astype(float)))


def ltt(conf, err, alpha):
    lam_ok = None
    for lam in GRID:
        s = conf >= lam; n, k = s.sum(), err[s].sum()
        if n and binom.cdf(k, n, alpha) <= DELTA / len(GRID):
            lam_ok = lam
    return lam_ok


def boot(y, p, g, B=2000):
    pid = np.unique(g); idx = {q: np.where(g == q)[0] for q in pid}; rng = np.random.default_rng(0); v = []
    for _ in range(B):
        s = np.concatenate([idx[q] for q in rng.choice(pid, len(pid))])
        if len(np.unique(y[s])) == 2:
            v.append(roc_auc_score(y[s], p[s]))
    return f"{roc_auc_score(y, p):.3f} [{np.percentile(v, 2.5):.3f}, {np.percentile(v, 97.5):.3f}]"


def one_visit(F, seed):
    return F.groupby("participant_id").sample(1, random_state=seed).index.values


yP, gP = P.label.values, P.participant_id.values
rows = []
# (1) discrimination, model trained on all of Pitt development
allidx = np.arange(len(P))
mdl = fit(XP, yP, gP)
res = residualiser(allidx)
mdl_adj = fit(res(XP, P), yP, gP)
scores = {}
for name, (F, X) in EXT.items():
    y, g = F.label.values, F.participant_id.values
    p = mdl.predict_proba(X)[:, 1]; scores[name] = p
    ov = one_visit(F, 0)
    r = {"corpus": name, "model": "T (plain text)", "n_participants": F.participant_id.nunique(), "n_recordings": len(F),
         "AUC all recordings": boot(y, p, g), "AUC one visit per participant": boot(y[ov], p[ov], g[ov]),
         "accuracy at 0.5": f"{((p >= 0.5) == y).mean():.3f}", "mean predicted P(dementia)": f"{p.mean():.3f}",
         "median words": F.n_words.median()}
    rows.append(r)
    if name == "Delaware":
        pa = mdl_adj.predict_proba(res(X, F))[:, 1]; scores["Delaware-adj"] = pa
        rows.append({**r, "model": "T-adj (secondary)", "AUC all recordings": boot(y, pa, g),
                     "AUC one visit per participant": boot(y[ov], pa[ov], g[ov]),
                     "accuracy at 0.5": f"{((pa >= 0.5) == y).mean():.3f}", "mean predicted P(dementia)": f"{pa.mean():.3f}"})
pd.DataFrame(rows).to_csv(f"{out}/external_discrimination.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False), flush=True)

# (2) referral transfer: certify on Pitt calibration, apply unchanged
trans = []
pids = np.unique(gP)
for rep in range(200):
    rng = np.random.default_rng(rep); perm = rng.permutation(pids); calp = set(perm[: int(0.4 * len(perm))])
    tr = np.where(~np.isin(gP, list(calp)))[0]
    cal = np.array([i for i in one_visit(P, rep) if gP[i] in calp])
    mm = fit(XP[tr], yP[tr], gP[tr])
    pc = mm.predict_proba(XP[cal])[:, 1]; conf_c = np.maximum(pc, 1 - pc); err_c = (pc >= 0.5) != yP[cal]
    for name, (F, X) in EXT.items():
        ov = one_visit(F, rep); y = F.label.values[ov]
        pe = mm.predict_proba(X[ov])[:, 1]; conf_e = np.maximum(pe, 1 - pe); err_e = (pe >= 0.5) != y
        for alpha in ALPHAS:
            lam = ltt(conf_c, err_c, alpha)
            auto = conf_e >= lam if lam is not None else np.zeros(len(ov), bool)
            trans.append({"rep": rep, "corpus": name, "alpha": alpha, "lambda": lam, "auto_rate": auto.mean(),
                          "err_auto": err_e[auto].mean() if auto.any() else np.nan,
                          "sens_auto": (~err_e[auto & (y == 1)]).mean() if (auto & (y == 1)).any() else np.nan,
                          "spec_auto": (~err_e[auto & (y == 0)]).mean() if (auto & (y == 0)).any() else np.nan,
                          "auc": roc_auc_score(y, pe)})
    if rep % 25 == 0:
        print("transfer resplit", rep, flush=True)
T = pd.DataFrame(trans); T.to_csv(f"{out}/external_referral_transfer_resplits.csv", index=False)

# (3) recalibration on half of the external participants (scores from the full-development model)
recal = []
for name in ["Lu", "Delaware", "Delaware-adj"]:
    F, _ = EXT[name.split("-")[0]]; p_all = scores[name]
    for rep in range(200):
        ov = one_visit(F, rep); rng = np.random.default_rng(rep); half = rng.permutation(len(ov))
        cal, te = ov[half[: len(ov) // 2]], ov[half[len(ov) // 2:]]
        p = p_all; conf = np.maximum(p, 1 - p); err = (p >= 0.5) != F.label.values
        for alpha in ALPHAS:
            lam = ltt(conf[cal], err[cal], alpha)
            auto = conf[te] >= lam if lam is not None else np.zeros(len(te), bool)
            recal.append({"rep": rep, "corpus": name, "alpha": alpha, "auto_rate": auto.mean(),
                          "err_auto": err[te][auto].mean() if auto.any() else np.nan})
Rc = pd.DataFrame(recal); Rc.to_csv(f"{out}/external_referral_recalibrated_resplits.csv", index=False)


def summ(D, keys):
    return D.groupby(keys).apply(lambda g: pd.Series({
        "auto_rate": g.auto_rate.mean(), "err_auto (when any)": g.err_auto.mean(),
        "P(err_auto > alpha)": (g.err_auto.fillna(0) > g.name[-1]).mean(),
        **({"sens_auto": g.sens_auto.mean(), "spec_auto": g.spec_auto.mean()} if "sens_auto" in g else {})}),
        include_groups=False).reset_index()


S = pd.concat([summ(T, ["corpus", "alpha"]).assign(calibration="Pitt (transferred)"),
               summ(Rc, ["corpus", "alpha"]).assign(calibration="half of external corpus")])
S.to_csv(f"{out}/external_referral_summary.csv", index=False)
pd.set_option("display.width", 250)
print(S.round(3).to_string(index=False))
