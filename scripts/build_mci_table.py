"""MCI-vs-Control label table (transcript-level) from Delaware (all files) and Pitt MCI sessions.
Kept separate from the Pitt AD-vs-Control set. Audio: Delaware has none here; Pitt MCI audio exists only
as unmapped student 'streamRange' files, so it is marked unavailable.
Usage: python build_mci_table.py <delaware_extracted_dir> <pitt_ad_transcripts_dir> <pitt_master_csv> <out_csv>"""
import sys, os, glob, re, pandas as pd
dela, pitt_dir, master, out = sys.argv[1:5]
def parse(f):
    txt = open(f, encoding="utf-8", errors="ignore").read()
    pid = next((l for l in txt.splitlines() if l.startswith("@ID:") and "|PAR|" in l), "")
    p = pid.split("|") if pid else [""] * 11
    gems = re.findall(r"^@G:\s*(\S+)", txt, re.M)
    # PAR words inside the Cookie gem only
    words, in_cookie = 0, False
    for l in txt.splitlines():
        if l.startswith("@G:"): in_cookie = "cookie" in l.lower()
        elif in_cookie and l.startswith("*PAR:"):
            words += len([w for w in re.sub(r"\d+_\d+|\[.*?\]|&[-=+]\S+|[.,?!]", " ", l[5:]).split() if w.isalpha()])
    age = re.match(r"(\d+)", p[3] or "")
    mmse = p[8].strip() if len(p) > 8 else ""
    return dict(header_dx=p[5], age=int(age.group(1)) if age else None, sex=p[4],
                mmse=float(mmse) if mmse.replace(".", "").isdigit() else None,
                tasks=";".join(dict.fromkeys(gems)), has_cookie=any("cookie" in g.lower() for g in gems), cookie_par_words=words)
rows = []
for f in sorted(glob.glob(f"{dela}/**/*.cha", recursive=True)):
    mid = os.path.basename(f)[:-4]; folder = os.path.basename(os.path.dirname(f))
    r = parse(f); rows.append(dict(dataset="Delaware", media_id=mid, participant_id="D" + folder[0] + mid.split("-")[0], folder_label=folder, **r, audio_available=False))
m = pd.read_csv(master, dtype={"media_id": str})
for f in sorted(glob.glob(f"{pitt_dir}/*.cha")):
    mid = os.path.basename(f)[:-4]
    if " (" in mid or mid in set(m.media_id): continue  # skip duplicates and verified AD/Control sessions
    r = parse(f)
    if r["header_dx"] != "MCI": continue
    rows.append(dict(dataset="Pitt", media_id=mid, participant_id="P" + mid.split("-")[0], folder_label="With AD (student folder)", **r, audio_available=False))
x = pd.DataFrame(rows)
x["label"] = x.header_dx.replace({"": None}).fillna(x.folder_label).map(lambda d: "MCI" if d == "MCI" else ("Control" if d == "Control" else d))
x["label_conflict"] = (x.dataset == "Delaware") & (x.header_dx != "") & (x.header_dx != x.folder_label)
pitt_ctrl = set(m[m.source_group == "Control"].participant_id.astype(str).str.zfill(3))
x["also_in_pitt_control_set"] = (x.dataset == "Pitt") & x.participant_id.str[1:].isin(pitt_ctrl)
pitt_dem = set(m[m.source_group == "Dementia"].participant_id.astype(str).str.zfill(3))
x["also_in_pitt_dementia_set"] = (x.dataset == "Pitt") & x.participant_id.str[1:].isin(pitt_dem)
x.to_csv(out, index=False)
print(x.groupby(["dataset", "label"]).agg(files=("media_id", "size"), participants=("participant_id", "nunique"), with_cookie=("has_cookie", "sum"), mean_age=("age", "mean")).round(1))
print("label conflicts:", x.label_conflict.sum(), "| missing header dx:", (x.header_dx == "").sum())
print("Pitt MCI participants also in verified Control set:", sorted(x[x.also_in_pitt_control_set].participant_id.unique()))
print("Pitt MCI participants also in verified Dementia set:", sorted(x[x.also_in_pitt_dementia_set].participant_id.unique()))

# Participant-grouped 80/20 split; participants whose diagnosis changes across visits are flagged.
from sklearn.model_selection import StratifiedGroupKFold
x["dx_changes_across_visits"] = x.groupby("participant_id").label.transform("nunique") > 1
ok = x.label.isin(["MCI", "Control"])
y = x[ok]; strat = y.groupby("participant_id").label.transform(lambda s: "MCI" if (s == "MCI").any() else "Control")
tr, te = next(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(y, strat, y.participant_id))
x["split"] = ""; x.loc[y.index[tr], "split"] = "train"; x.loc[y.index[te], "split"] = "test"
x.to_csv(out, index=False)
print(x.groupby(["split", "label"]).size().unstack(), "\nparticipants with changing dx:", x[x.dx_changes_across_visits].participant_id.nunique())
