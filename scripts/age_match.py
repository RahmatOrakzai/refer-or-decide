"""Greedy 1:1 participant matching (first visit) on age (caliper 3 y) and education (caliper 3 y).
Adds column age_matched to pitt_splits.csv. Usage: python age_match.py derived"""
import sys, pandas as pd, numpy as np
out = sys.argv[1]
x = pd.read_csv(f"{out}/pitt_master_recordings.csv", dtype={"participant_id": str})
p = x.sort_values("visit").drop_duplicates("participant_id")
ctrl, dem = p[p.label == 0].copy(), p[p.label == 1].copy()
used, pairs = set(), []
for _, c in ctrl.sort_values("age_at_visit", ascending=False).iterrows():  # oldest controls are scarcest
    cand = dem[~dem.participant_id.isin(used)]
    cand = cand[(abs(cand.age_at_visit - c.age_at_visit) <= 3) & (abs(cand.education_years - c.education_years) <= 3)]
    if len(cand):
        d = cand.iloc[(abs(cand.age_at_visit - c.age_at_visit) + 0.5 * abs(cand.education_years - c.education_years)).argsort().iloc[0]]
        used.add(d.participant_id); pairs += [c.participant_id, d.participant_id]
s = pd.read_csv(f"{out}/pitt_splits.csv", dtype={"participant_id": str})
s["age_matched"] = s.participant_id.isin(pairs); s.to_csv(f"{out}/pitt_splits.csv", index=False)
m = p[p.participant_id.isin(pairs)]
print("matched pairs:", len(pairs) // 2); print(m.groupby("label")[["age_at_visit", "education_years"]].mean().round(1))
