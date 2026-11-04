"""Join the verified Pitt audio manifest with Pitt-data.xlsx demographics/clinical scores.
Usage: python build_master.py <source_metadata_dir> <out_dir>"""
import sys, warnings, pandas as pd, numpy as np
warnings.filterwarnings("ignore")
src, out = sys.argv[1], sys.argv[2]
m = pd.read_csv(f"{src}/source_verified_audio_manifest.csv", dtype={"participant_id": str, "media_id": str})
d = pd.read_excel(f"{src}/Pitt-data.xlsx", sheet_name="data", header=2)
d = d[pd.to_numeric(d["id"], errors="coerce").notna()].copy()
d["participant_id"] = d["id"].astype(int).astype(str).str.zfill(3)
# spreadsheet visit k (1-based) == CHAT visit k-1
DATE = ["idate", "visit2da", "visit3", "visit4", "visit5"]
MMSE = ["mms", "mmse2", "mmse3", "mmse4", "mmse5"]
CDR = ["cdrfs", "cdr2", "cdr3", "cdr4", "cdr5"]
m["visit"] = m.media_id.str.split("-").str[1].astype(int)
rows = []
for _, r in m.iterrows():
    p = d[d.participant_id == r.participant_id]
    rec = {"media_id": r.media_id, "found_in_spreadsheet": len(p) == 1}
    if len(p) == 1:
        p = p.iloc[0]; v = r.visit
        rec["sex"] = {1: "male", 0: "female"}.get(p["sex"], np.nan)
        rec["education_years"] = p["educ"]
        rec["entry_age"] = p["entryage"]
        base, vd = pd.to_datetime(p["idate"], errors="coerce"), pd.to_datetime(p[DATE[v]], errors="coerce") if v < 5 else pd.NaT
        rec["age_at_visit"] = round(p["entryage"] + (vd - base).days / 365.25, 1) if pd.notna(base) and pd.notna(vd) else (p["entryage"] if v == 0 else np.nan)
        rec["mmse"] = pd.to_numeric(p[MMSE[v]], errors="coerce") if v < 5 else np.nan
        rec["cdr"] = pd.to_numeric(p[CDR[v]], errors="coerce") if v < 5 else np.nan
        rec["base_dx_code"] = p["basedx"]
    rows.append(rec)
x = m.merge(pd.DataFrame(rows), on="media_id")
for c in ["mmse", "cdr", "education_years", "age_at_visit"]:
    x.loc[x[c] < 0, c] = np.nan  # negative codes = missing
x["label"] = (x.source_group == "Dementia").astype(int)
x["duration_flag"] = np.where(x.duration_seconds < 30, "under_30s", "")
x.to_csv(f"{out}/pitt_master_recordings.csv", index=False)
print(x.shape, "not in spreadsheet:", (~x.found_in_spreadsheet).sum())

# Fallback: spreadsheet has no visit dates/MMSE for some later visits; use the CHAT @ID tier
# (eng|Pitt|PAR|age;|sex|Dx||Participant|MMSE||) where a transcript is available.
import glob, os, re
cha_dir = sys.argv[3] if len(sys.argv) > 3 else None
if cha_dir:
    ids = {}
    for f in glob.glob(f"{cha_dir}/**/*.cha", recursive=True):
        mid = os.path.basename(f)[:-4]
        for line in open(f, encoding="utf-8", errors="ignore"):
            if line.startswith("@ID:") and "|PAR|" in line:
                parts = line.split("|"); ids[mid] = parts; break
    x["age_source"] = np.where(x.age_at_visit.notna(), "spreadsheet", "")
    for i, r in x.iterrows():
        p = ids.get(r.media_id)
        if p is None: continue
        if pd.isna(r.age_at_visit):
            a = re.match(r"(\d+)", p[3]);
            if a: x.at[i, "age_at_visit"] = float(a.group(1)); x.at[i, "age_source"] = "chat_id"
        if pd.isna(r.mmse) and p[8].strip().isdigit():
            x.at[i, "mmse"] = float(p[8])
        x.at[i, "chat_dx"] = p[5]
    x.to_csv(f"{out}/pitt_master_recordings.csv", index=False)
    print("after CHAT fallback, missing age:", x.age_at_visit.isna().sum(), "missing mmse:", x.mmse.isna().sum())
