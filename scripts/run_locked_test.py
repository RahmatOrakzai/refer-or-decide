"""Final, one-time evaluation on the locked test set (protocol: 'touched once, at the end').

Train on 60% of development participants, calibrate the referral layer on the other 40% (one visit per
participant), then evaluate discrimination and referral once on the locked test (one visit per participant
primary; all visits secondary). Refuses to run twice unless --force is given.

Usage: python run_locked_test.py <derived_dir> <out_dir> [F-adj|F] [--force]
"""
import os, sys, runpy
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

opts = [a for a in sys.argv[1:] if a.startswith(("--emb=", "--tag=", "--covars="))]   # passed through to run_referral.py
args = [a for a in sys.argv[1:] if a != "--force" and a not in opts]
der, out_dir = args[:2]
model = args[2] if len(args) > 2 else "F-adj"
tag = next((a[6:] for a in opts if a.startswith("--tag=")), model)
flag = f"{out_dir}/LOCKED_TEST_OPENED_{tag}"
if os.path.exists(flag) and "--force" not in sys.argv:
    sys.exit(f"Locked test already opened for {model} ({flag}). Use --force only with a logged reason.")

# reuse the referral definitions without running its resplit loop
sys.argv = ["run_referral.py", der, out_dir, model, "--defs-only", *opts]
R = runpy.run_path(os.path.join(os.path.dirname(__file__), "run_referral.py"), run_name="defs")
d, y, conf_sets, ltt, train_scores = R["d"], R["y"], R["conformal_sets"], R["ltt_lambda"], R["train_scores"]
edu_hi, STRATA, ALPHAS = R["edu_hi"], R["STRATA"], R["ALPHAS"]

locked = d.locked_test.astype(str).eq("True").values
dev_p = d.participant_id[~locked].unique()
rng = np.random.default_rng(2026)
perm = rng.permutation(dev_p); calp = set(perm[: int(0.4 * len(perm))])
tr = np.where(~locked & ~d.participant_id.isin(calp))[0]
one = d.groupby("participant_id").sample(1, random_state=2026).index.values
cal = np.array([i for i in one if d.participant_id[i] in calp])
te1 = np.array([i for i in one if locked[i]])
teall = np.where(locked)[0]
p = train_scores(tr)
pred = (p >= 0.5).astype(int); conf = np.maximum(p, 1 - p); err = pred != y
rows = []
for tname, te in (("one visit per participant", te1), ("all visits", teall)):
    for alpha in ALPHAS:
        i0, i1 = conf_sets(p, cal, te, alpha, None)
        cov = np.where(y[te] == 1, i1, i0); single = i0 ^ i1
        rows.append({"test": tname, "method": "conformal", "alpha": alpha, "auc": roc_auc_score(y[te], p[te]),
                     "auto_rate": single.mean(), "err_auto": (single & ~cov).sum() / max(single.sum(), 1),
                     "cov_dementia": cov[y[te] == 1].mean(), "cov_control": cov[y[te] == 0].mean()})
        for mname, cells in (("LTT-bonferroni", {"all": np.ones(len(d), bool)}), ("LTT-bonferroni-per-education", {"lo": ~edu_hi, "hi": edu_hi})):
            auto = np.zeros(len(te), bool)
            for cell in cells.values():
                lam = ltt(conf[cal[cell[cal]]], err[cal[cell[cal]]], alpha)
                if lam is not None:
                    auto |= cell[te] & (conf[te] >= lam)
            e = err[te]
            r = {"test": tname, "method": mname, "alpha": alpha, "auc": roc_auc_score(y[te], p[te]),
                 "auto_rate": auto.mean(), "err_auto": e[auto].mean() if auto.any() else np.nan}
            for sn, sm in STRATA.items():
                a2 = auto & sm[te]
                r[f"auto[{sn}]"] = a2.sum() / max(sm[te].sum(), 1)
                r[f"err_auto[{sn}]"] = e[a2].mean() if a2.any() else np.nan
            rows.append(r)
pd.DataFrame(rows).to_csv(f"{out_dir}/locked_test_{tag}.csv", index=False)
pd.DataFrame({"media_id": d.media_id.values[teall], "participant_id": d.participant_id.values[teall], "label": y[teall], "p": p[teall], "in_one_visit_set": np.isin(teall, te1)}).to_csv(f"{out_dir}/locked_test_predictions_{tag}.csv", index=False)
lam_rows = [{"alpha": a, "lambda_bonferroni": ltt(conf[cal], err[cal], a)} for a in ALPHAS]
pd.DataFrame(lam_rows).to_csv(f"{out_dir}/locked_test_thresholds_{tag}.csv", index=False)
open(flag, "w").write("opened\n")
print(pd.DataFrame(rows).round(3).to_string(index=False))
