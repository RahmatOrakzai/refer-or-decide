"""Participant text from the CHAT main tier (no word timings needed), identical preprocessing for every corpus.

Writes one row per recording: corpus, media_id, participant_id, label, age, sex, education_years, n_words, text,
inv_turns, inv_words, inv_word_share (interviewer covariates that need no timings).

Usage:
  python make_main_tier_text.py pitt <master_csv> <ad_dir> <ctl_dir> <out_csv> [extra_dir]
  python make_main_tier_text.py lu <lu_dir> <out_csv>
  python make_main_tier_text.py delaware <delaware_dir> <demo_xlsx> <out_csv>
"""
import os, re, sys, glob
import pandas as pd

CODE = re.compile(r"\[[^\]]*\]")          # [/], [+ exc], [: target], [* m] ...
TIME = re.compile(r"\x15[^\x15]*\x15")
EDU = {0: 0, 1: 8, 2: 10, 3: 12, 4: 14, 5: 13, 6: 14, 7: 16, 8: 18, 9: 19, 10: 20}


def clean_tok(tok):
    if tok.startswith("&-"):
        return tok[2:]
    if tok.startswith(("&", "xxx", "yyy", "www", "+", "0")):
        return None
    tok = re.sub(r"[<>]", "", tok)
    tok = re.sub(r"\(([^)]*)\)", r"\1", tok)
    tok = re.sub(r"@\w+$", "", tok).replace("_", " ").replace(":", "")
    return tok.lower() if re.search(r"[A-Za-z]", tok) else None


def parse(path, gem=None):
    """Return PAR words and INV word/turn counts, optionally only inside an @G gem whose name starts with gem."""
    txt = open(path, encoding="utf-8", errors="replace").read()
    inside = gem is None
    par, inv_turns, inv_words, age, sex = [], 0, 0, None, None
    for b in re.split(r"\n(?=[*@%])", txt):
        if b.startswith("@ID") and "|PAR|" in b:
            f = b.split("|"); age = f[3].split(";")[0] or None; sex = f[4] or None
        if gem is not None and b.startswith("@G:"):
            inside = b[3:].strip().lower().startswith(gem.lower())
            continue
        if gem is not None and b.startswith("@Eg"):
            inside = False
        if not inside or not b.startswith(("*PAR", "*INV")):
            continue
        line = TIME.sub(" ", CODE.sub(" ", b.split(":", 1)[1].replace("\n", " ").replace("\t", " ")))
        toks = [c for t in line.split() if (c := clean_tok(t))]
        if b.startswith("*PAR"):
            par += toks
        else:
            inv_turns += 1; inv_words += len(toks)
    return par, inv_turns, inv_words, age, sex


def row(corpus, mid, pid, label, path, gem=None, **extra):
    par, it, iw, age, sex = parse(path, gem)
    r = {"corpus": corpus, "media_id": mid, "participant_id": pid, "label": label, "age": age, "sex": sex,
         "n_words": len(par), "text": " ".join(par), "inv_turns": it, "inv_words": iw,
         "inv_word_share": iw / max(iw + len(par), 1)}
    r.update(extra)
    return r


mode = sys.argv[1]
rows = []
if mode == "pitt":
    master, ad, ctl, out = sys.argv[2:6]; extra = sys.argv[6:]
    for r in pd.read_csv(master, dtype={"media_id": str, "participant_id": str}).itertuples():
        cands = [os.path.join(ad if r.source_group == "Dementia" else ctl, r.media_id + ".cha")] + \
                [os.path.join(e, r.media_id + ".cha") for e in extra]
        p = next((c for c in cands if os.path.exists(c)), None)
        if p:
            rows.append(row("pitt", r.media_id, r.participant_id, r.label, p, education_years=r.education_years))
elif mode == "lu":
    src, out = sys.argv[2:4]
    for lab, sub in ((1, "Dementia"), (0, "Control")):
        for p in sorted(glob.glob(f"{src}/{sub}/*.cha")):
            mid = os.path.basename(p)[:-4]
            rows.append(row("lu", mid, mid, lab, p, gem=None, education_years=None))
elif mode == "delaware":
    src, xlsx, out = sys.argv[2:5]
    # Record IDs are reused across the Control and MCI sheets (23 numeric prefixes occur in both), so the
    # education lookup and the participant identifier must both be qualified by the sheet/folder.
    edu = {}
    for s in ("Control", "MCI"):
        dsheet = pd.read_excel(xlsx, sheet_name=s).rename(columns=lambda c: c.strip().upper())
        edu.update({(s, int(a), int(v)): EDU.get(e)
                    for a, v, e in zip(dsheet["RECORD ID"], dsheet["VISIT NUMBER"], dsheet["EDUCATION LEVEL CODE"])
                    if pd.notna(a) and pd.notna(v) and str(e).strip().isdigit() for e in [int(str(e).strip())]})
    for lab, sub in ((1, "MCI"), (0, "Control")):
        for p in sorted(glob.glob(f"{src}/{sub}/*.cha")):
            mid = os.path.basename(p)[:-4]; pid, visit = mid.split("-")[:2]
            rows.append(row("delaware", mid, "D" + sub[0] + pid, lab, p, gem="Cookie", visit=int(visit),
                            education_years=edu.get((sub, int(pid), int(visit)))))
else:
    sys.exit("unknown corpus")
df = pd.DataFrame(rows)
# Recordings with no participant words in the target segment carry no text to classify and were excluded
# from the published tables; the filter is made explicit here so the script reproduces its own output.
n_empty = int((df.n_words == 0).sum())
df = df[df.n_words > 0].reset_index(drop=True)
print(f"dropped {n_empty} recordings with zero participant words")
df.to_csv(out, index=False)
print(mode, len(df), "recordings;", df.participant_id.nunique(), "participants; label counts", df.label.value_counts().to_dict(),
      "; median words", df.n_words.median(), "; empty", (df.n_words == 0).sum())
