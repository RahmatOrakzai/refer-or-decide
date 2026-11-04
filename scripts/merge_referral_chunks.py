"""Merge chunked run_referral.py outputs (--reps=a:b) into the standard resplit, risk-curve and summary files.
Usage: python merge_referral_chunks.py <out_dir> <tag>
"""
import sys, glob
import pandas as pd
out, tag = sys.argv[1:3]
R = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{out}/referral_resplits_{tag}_reps*.csv"))]).sort_values("rep")
C = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{out}/referral_risk_curve_{tag}_reps*.csv"))]).sort_values(["rep", "lambda"])
assert R.rep.nunique() == 200, R.rep.nunique()
R.to_csv(f"{out}/referral_resplits_{tag}.csv", index=False); C.to_csv(f"{out}/referral_risk_curve_{tag}.csv", index=False)
agg = []
for (meth, alpha), g in R.groupby(["method", "alpha"]):
    row = {"method": meth, "alpha": alpha, "auto_rate": g.auto_rate.mean(), "err_auto_mean": g.err_auto.mean(),
           "P(err_auto > alpha)": (g.err_auto > alpha).mean()}
    for c in g.columns:
        if c.startswith(("cov", "sens", "spec", "auto[", "err_auto[")):
            row[c] = g[c].mean()
    agg.append(row)
pd.DataFrame(agg).to_csv(f"{out}/referral_summary_{tag}.csv", index=False)
print(f"mean test AUC {R.drop_duplicates('rep').auc_test.mean():.3f}")
print(pd.DataFrame(agg).round(3).to_string(index=False))
