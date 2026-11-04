"""Quick feasibility audit: how much of Pitt AD-vs-control signal is demographics, interviewer behaviour,
or participant-level leakage? Grouped 5-fold CV, repeated 20 times, logistic regression on standardised features.

Usage: python shortcut_audit.py <master_csv> <features_csv> <splits_csv> <out_md>
"""
import sys
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score

master_csv, feat_csv, split_csv, out_md = sys.argv[1:5]
ids = {"media_id": str, "participant_id": str}
m = pd.read_csv(master_csv, dtype=ids)
f = pd.read_csv(feat_csv, dtype={"media_id": str})
s = pd.read_csv(split_csv, dtype=ids)
d = m.merge(f, on="media_id").merge(s[["media_id", "first_visit_only", "age_matched"]], on="media_id")
d = d[d.transcript_found == True].copy()
d["male"] = (d.sex == "male").astype(float)

DEMO = ["age_at_visit", "education_years", "male"]
INV = ["inv_turns", "inv_words", "inv_turns_after_start", "inv_word_share", "inv_prompts_anything_else", "inv_encouragement"]
PAR = ["par_words", "par_ttr", "par_mlu_words", "par_filled_pauses", "par_fragments", "par_repetitions",
       "par_retracings", "par_unintelligible", "par_info_units", "par_info_unit_tokens", "par_words_per_min",
       "par_pauses_gt_0_5s", "par_pauses_gt_2s", "par_mean_pause_s", "par_phonation_ratio", "par_span_s"]
LEN = ["par_span_s"]


def model():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def cv_auc(df, cols, grouped=True, reps=20, residualize=None):
    y, g = df.label.values, df.participant_id.values
    aucs = []
    for rep in range(reps):
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=rep) if grouped else StratifiedKFold(5, shuffle=True, random_state=rep)
        p = np.zeros(len(df))
        for tr, te in cv.split(df, y, g if grouped else None):
            X = df[cols].copy()
            if residualize:  # regress each feature on demographics using train controls only
                Z = df[residualize].fillna(df[residualize].median())
                ctl = tr[y[tr] == 0]
                for c in cols:
                    xc = X[c].fillna(X[c].iloc[tr].median())
                    lr = LinearRegression().fit(Z.iloc[ctl], xc.iloc[ctl])
                    X[c] = xc - lr.predict(Z)
            mdl = model().fit(X.iloc[tr], y[tr])
            p[te] = mdl.predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y, p))
    return np.mean(aucs), np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)


lines = ["# Shortcut audit (feasibility, not final results)", "",
         "Logistic regression, 5-fold CV repeated 20 times. Grouped = no participant in both train and test.", "",
         "| Subset | Feature set | Split | AUC (mean) | 2.5–97.5% over repeats | n recordings |", "|---|---|---|---|---|---|"]
subsets = {"all visits": d, "first visit only": d[d.first_visit_only == True],
           "age/education-matched": d[d.age_matched == True]}
sets = [("Demographics only (age, education, sex)", DEMO, None), ("Interviewer turns only", INV, None),
        ("Participant speech features", PAR, None), ("Participant + interviewer", PAR + INV, None),
        ("Participant, residualised on age/education/sex", PAR, DEMO), ("Recording length only", LEN, None)]
for sname, sd in subsets.items():
    for fname, cols, res in sets:
        a, lo, hi = cv_auc(sd, cols, residualize=res)
        lines.append(f"| {sname} | {fname} | grouped | {a:.3f} | {lo:.3f}–{hi:.3f} | {len(sd)} |")
    if sname == "all visits":
        a, lo, hi = cv_auc(sd, PAR + INV, grouped=False)
        lines.append(f"| {sname} | Participant + interviewer | random (leaky) | {a:.3f} | {lo:.3f}–{hi:.3f} | {len(sd)} |")

# Longitudinal: within-person change vs MMSE change (consecutive visits)
lines += ["", "## Within-person change (consecutive visits of the same participant)", "",
          "| Feature change | Spearman rho with MMSE change | p | n pairs |", "|---|---|---|---|"]
dd = d.sort_values(["participant_id", "visit"])
pairs = []
for pid, gdf in dd.groupby("participant_id"):
    gdf = gdf.dropna(subset=["mmse"])
    for a, b in zip(gdf.iloc[:-1].itertuples(), gdf.iloc[1:].itertuples()):
        pairs.append({"dmmse": b.mmse - a.mmse, "label": a.label, **{c: getattr(b, c) - getattr(a, c) for c in PAR + INV}})
P = pd.DataFrame(pairs)
for c in ["par_info_units", "par_words_per_min", "par_mean_pause_s", "par_phonation_ratio", "par_retracings",
          "inv_turns_after_start", "inv_word_share"]:
    x = P[[c, "dmmse"]].dropna()
    r, p = stats.spearmanr(x[c], x.dmmse)
    lines.append(f"| {c} | {r:.2f} | {p:.3g} | {len(x)} |")

# How much do interviewer prompts track severity?
lines += ["", "## Interviewer behaviour vs severity (all recordings)", ""]
for c in ["inv_turns_after_start", "inv_word_share", "inv_prompts_anything_else"]:
    x = d[[c, "mmse"]].dropna()
    r, p = stats.spearmanr(x[c], x.mmse)
    lines.append(f"- {c} vs MMSE: Spearman rho {r:.2f} (p {p:.2g}, n {len(x)})")
open(out_md, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
