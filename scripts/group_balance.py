"""Group balance (Dementia vs Control) at recording and participant level, with standardized mean differences.
Usage: python group_balance.py derived/pitt_master_recordings.csv derived"""
import sys, pandas as pd, numpy as np
from scipy import stats
x = pd.read_csv(sys.argv[1], dtype={"participant_id": str}); out = sys.argv[2]
rows = []
for level, df in [("recording", x), ("participant_first_visit", x.sort_values("visit").drop_duplicates("participant_id"))]:
    for v in ["age_at_visit", "education_years", "mmse", "duration_seconds"]:
        a, b = df[df.label == 1][v].dropna(), df[df.label == 0][v].dropna()
        smd = (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)
        rows.append(dict(level=level, variable=v, dementia_mean=a.mean(), control_mean=b.mean(), smd=smd, p_value=stats.mannwhitneyu(a, b).pvalue, n_dementia=len(a), n_control=len(b)))
    ct = pd.crosstab(df.label, df.sex)
    rows.append(dict(level=level, variable="pct_female", dementia_mean=100 * ct.loc[1, "female"] / ct.loc[1].sum(), control_mean=100 * ct.loc[0, "female"] / ct.loc[0].sum(), p_value=stats.chi2_contingency(ct)[1]))
r = pd.DataFrame(rows).round(3); r.to_csv(f"{out}/group_balance.csv", index=False); print(r.to_string())
