"""How many calibration participants would certify error-among-automatic-decisions <= alpha?

Uses the saved test-set risk curve (mean automatic rate a and realised error e at each fixed threshold, averaged
over the 200 resplits) and computes, for each alpha, the smallest calibration size n for which the Bonferroni LTT
test (delta/9) rejects with at least 80% power at some grid threshold, and the automatic rate it would then give.

Usage: python calibration_size.py <results_dir> <model> [<model> ...]
"""
import sys
import numpy as np, pandas as pd
from scipy.stats import binom

res, models = sys.argv[1], sys.argv[2:]
DELTA, K, POWER = 0.10, 9, 0.80
rows = []
for m in models:
    c = pd.read_csv(f"{res}/referral_risk_curve_{m}.csv").groupby("lambda")[["auto_rate", "err_auto"]].mean()
    for alpha in [0.05, 0.10, 0.15, 0.20]:
        best = None
        for lam, (a, e) in c.iterrows():
            if not e < alpha:
                continue
            for n in range(20, 5001, 5):
                nd = int(round(n * a))
                if nd < 1:
                    continue
                kmax = binom.ppf(DELTA / K, nd, alpha)  # largest k with cdf <= delta/K, approximately
                while kmax >= 0 and binom.cdf(kmax, nd, alpha) > DELTA / K:
                    kmax -= 1
                if kmax >= 0 and binom.cdf(kmax, nd, e) >= POWER:
                    if best is None or n < best[0]:
                        best = (n, lam, a, e)
                    break
        rows.append({"model": m, "alpha": alpha,
                     "calibration participants needed": best[0] if best else ">5000 / unreachable",
                     "threshold": best[1] if best else None,
                     "automatic rate": round(best[2], 3) if best else None,
                     "true error among automatic": round(best[3], 3) if best else None})
R = pd.DataFrame(rows)
R.to_csv(f"{res}/calibration_size_needed.csv", index=False)
print(R.to_string(index=False))
