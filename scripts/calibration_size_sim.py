"""Calibration-size uncertainty (reviewer point 2.9).

Extends calibration_size.py in two ways:
  (1) the number of calibration participants above each threshold is random: each simulated participant
      falls in a confidence bin (from the grid) with the bin probabilities implied by the risk curve, and
      errs with that bin's error rate; the full Bonferroni LTT (delta/9 over the grid) is then applied;
  (2) uncertainty in the risk curve is propagated by bootstrapping the 200 resplits (B curves).
n*_sim(alpha) = smallest n at which LTT certifies some threshold in >= 80% of simulated calibration sets.
Usage: python calibration_size_sim.py <results_dir> <out_csv> <model> [<model> ...]
"""
import sys
import numpy as np, pandas as pd
from scipy.stats import binom

res, out, models = sys.argv[1], sys.argv[2], sys.argv[3:]
DELTA, POWER, SIMS, B = 0.10, 0.80, 400, 100
NGRID = list(range(25, 1001, 25)) + list(range(1100, 3001, 100))
rng = np.random.default_rng(0)


def bins_from_curve(c):
    lam = np.sort(c.index.values)[::-1]                       # 0.95 ... 0.55
    a = np.maximum.accumulate(c.loc[lam, "auto_rate"].values)
    E = np.maximum.accumulate(np.nan_to_num(c.loc[lam, "err_auto"].values) * a)
    pa = np.diff(np.concatenate([[0], a])); pe = np.diff(np.concatenate([[0], E]))
    pbin = np.append(pa, 1 - a[-1]); ebin = np.where(pa > 0, np.clip(pe / np.where(pa > 0, pa, 1), 0, 1), 0)
    return lam, pbin, np.append(ebin, 0)


def kcrit(nmax, alpha):
    k = np.full(nmax + 1, -1)
    for m in range(1, nmax + 1):
        kk = int(binom.ppf(DELTA / 9, m, alpha))
        while kk >= 0 and binom.cdf(kk, m, alpha) > DELTA / 9:
            kk -= 1
        k[m] = kk
    return k


def power(lam, pbin, ebin, n, alpha, kc):
    nb = len(lam)
    cnt = rng.multinomial(n, pbin, size=SIMS)[:, :nb]                     # participants per confident bin
    errs = rng.binomial(cnt, ebin[:nb])
    m, k = np.cumsum(cnt, 1), np.cumsum(errs, 1)                          # counts above each threshold
    ok = k <= kc[m]
    ok &= m > 0
    cert = ok.any(1)
    auto = np.where(ok, m, 0).max(1) / n
    return cert.mean(), auto[cert].mean() if cert.any() else 0.0


rows = []
for mdl in models:
    R = pd.read_csv(f"{res}/referral_risk_curve_{mdl}.csv")
    reps = R.rep.unique()
    for alpha in [0.10, 0.15, 0.20]:
        kc = kcrit(max(NGRID), alpha)
        def nstar(curve):
            lam, pbin, ebin = bins_from_curve(curve)
            for n in NGRID:
                pw, au = power(lam, pbin, ebin, n, alpha, kc)
                if pw >= POWER:
                    return n, au
            return np.nan, np.nan
        n0, a0 = nstar(R.groupby("lambda")[["auto_rate", "err_auto"]].mean())
        boots = []
        for b in range(B):
            s = rng.choice(reps, len(reps))
            cb = pd.concat([R[R.rep == r] for r in s]).groupby("lambda")[["auto_rate", "err_auto"]].mean()
            boots.append(nstar(cb)[0])
        boots = np.array(boots, float)
        rows.append(dict(model=mdl, alpha=alpha, n_star_sim=n0, auto_rate_at_n_star=round(a0, 3),
                         boot_median=np.nanmedian(boots), boot_lo=np.nanpercentile(boots, 2.5) if np.isfinite(boots).any() else np.nan,
                         boot_hi=np.nanpercentile(boots, 97.5) if np.isfinite(boots).any() else np.nan,
                         share_unreachable=np.mean(~np.isfinite(boots))))
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(out, index=False)
