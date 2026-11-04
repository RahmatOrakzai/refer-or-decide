"""Participant-only text per recording, plain and with pause tokens from gold word timings
(Yuan et al. 2020 bins: ',' < 0.5 s, '.' 0.5-2 s, '...' > 2 s; gaps under 0.05 s get no token).

Usage: python make_par_text.py <master_csv> <ad_cha_dir> <ctl_cha_dir> <out_csv> [extra_dir ...]
"""
import os, re, sys
import pandas as pd

master_csv, ad_dir, ctl_dir, out_csv = sys.argv[1:5]
extra = sys.argv[5:]
WT = re.compile(r"(\S+?)\s+\x15?(\d+)_(\d+)\x15?")


def clean(tok):
    if tok.startswith("&-"):            # filled pause: keep as a word
        return tok[2:]
    if tok.startswith("&") or tok.startswith("xxx") or tok.startswith("["):
        return None                      # fragments, gestures, unintelligible, codes
    tok = re.sub(r"[<>\[\]]", "", tok)
    tok = re.sub(r"\(([^)]*)\)", r"\1", tok)  # sayin(g) -> saying
    tok = tok.replace("_", " ").replace("+", "")
    tok = re.sub(r"@\w+$", "", tok)
    return tok if re.search(r"[A-Za-z]", tok) else None


def par_words(path):
    txt = open(path, encoding="utf-8", errors="replace").read()
    words = []
    for b in re.split(r"\n(?=\*)", txt):
        if not b.startswith("*PAR"):
            continue
        mw = re.search(r"%wor:\s*(.*?)(?=\n%|\Z)", b, re.S)
        if mw:  # untimed tokens inherit the previous word's end time (no pause token before them)
            last = None
            toks = mw.group(1).replace("\n", " ").split()
            for i, t in enumerate(toks):
                m = re.fullmatch(r"\x15?(\d+)_(\d+)\x15?", t)
                if m:
                    continue
                nxt = re.fullmatch(r"\x15?(\d+)_(\d+)\x15?", toks[i + 1]) if i + 1 < len(toks) else None
                if nxt:
                    a, e = int(nxt.group(1)), int(nxt.group(2)); last = e
                elif last is not None:
                    a = e = last
                else:
                    continue
                words.append((t, a, e))
    words.sort(key=lambda x: x[1])
    return [(c, a, e) for w, a, e in words if (c := clean(w))]


rows = []
for r in pd.read_csv(master_csv, dtype={"media_id": str}).itertuples():
    cands = [os.path.join(ad_dir if r.source_group == "Dementia" else ctl_dir, r.media_id + ".cha")] + \
            [os.path.join(d, r.media_id + ".cha") for d in extra]
    p = next((c for c in cands if os.path.exists(c)), None)
    if p is None:
        continue
    w = par_words(p)
    out = []
    for i, (tok, a, e) in enumerate(w):
        if i:
            gap = (a - w[i - 1][2]) / 1000
            if gap >= 2:
                out.append("...")
            elif gap >= 0.5:
                out.append(".")
            elif gap >= 0.05:
                out.append(",")
        out.append(tok)
    rows.append({"media_id": r.media_id, "n_words": len(w),
                 "text_plain": " ".join(t for t, _, _ in w), "text_pause": " ".join(out)})
pd.DataFrame(rows).to_csv(out_csv, index=False)
print(len(rows), "transcripts written")
