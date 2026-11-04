"""Write derived/protocol_folds.csv: locked test flag and 10 repeats of participant-grouped 5-fold CV on development.

Usage: python make_protocol_folds.py <pitt_splits_csv> <out_csv>
"""
import sys
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

s = pd.read_csv(sys.argv[1], dtype={"media_id": str, "participant_id": str})
s["locked_test"] = s.split == "test"
dev = s[~s.locked_test]
for r in range(10):
    col = f"fold_r{r}"
    s[col] = -1
    for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=r).split(dev, dev.label, dev.participant_id)):
        s.loc[dev.index[te], col] = k
    # no participant spans two folds
    assert s[~s.locked_test].groupby("participant_id")[col].nunique().max() == 1
assert not set(s[s.locked_test].participant_id) & set(dev.participant_id)
s[["media_id", "participant_id", "label", "visit", "locked_test"] + [f"fold_r{r}" for r in range(10)]].to_csv(sys.argv[2], index=False)
print(s.locked_test.sum(), "locked test recordings;", len(dev), "development recordings")
