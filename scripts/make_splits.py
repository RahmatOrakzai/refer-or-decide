"""Participant-level splits: 20% held-out test + 5 grouped, label-stratified CV folds on the rest.
Usage: python make_splits.py derived/pitt_master_recordings.csv derived"""
import sys, pandas as pd, numpy as np
from sklearn.model_selection import StratifiedGroupKFold
x = pd.read_csv(sys.argv[1], dtype={"participant_id": str}); out = sys.argv[2]
SEED = 42
outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
tr_idx, te_idx = next(outer.split(x, x.label, x.participant_id))
x["split"] = "train"; x.loc[x.index[te_idx], "split"] = "test"
tr = x[x.split == "train"]
x["cv_fold"] = -1
for k, (_, v) in enumerate(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(tr, tr.label, tr.participant_id)):
    x.loc[tr.index[v], "cv_fold"] = k
# one-recording-per-participant view: earliest visit
x["first_visit_only"] = x.groupby("participant_id").visit.transform("min") == x.visit
assert x.groupby("participant_id").split.nunique().max() == 1
assert x[x.split == "train"].groupby("participant_id").cv_fold.nunique().max() == 1
x[["media_id", "participant_id", "visit", "source_group", "label", "split", "cv_fold", "first_visit_only"]].to_csv(f"{out}/pitt_splits.csv", index=False)
s = x.groupby(["split", "source_group"]).agg(recordings=("media_id", "size"), participants=("participant_id", "nunique"), mean_age=("age_at_visit", "mean")).round(1)
print(s); print(x[x.split == "train"].groupby(["cv_fold", "source_group"]).size().unstack())
