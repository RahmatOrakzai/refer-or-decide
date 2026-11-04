"""Build participant text from Whisper output (protocol Addendum B1).

oracle: ASR words whose midpoint lies inside a PAR utterance's time span in the CHAT file.
nodiar: every ASR word (fully automatic).
Also reports word error rate (oracle text vs. CHAT main-tier PAR text) and writes silence-only features.

Usage: python make_asr_text.py <asr_dir> <master_csv> <ad_cha_dir> <ctl_cha_dir> <pitt_main_text_csv> <out_csv> <silence_csv> [extra_cha_dir]
"""
import os, re, sys, json, glob
import numpy as np, pandas as pd

asr, master, ad, ctl, main_csv, out, sil_out = sys.argv[1:8]
extra = sys.argv[8:]
BUL = re.compile(r"\x15(\d+)_(\d+)\x15")


def par_spans(path):
    txt = open(path, encoding="utf-8", errors="replace").read()
    spans = []
    for b in re.split(r"\n(?=[*@%])", txt):
        if b.startswith("*PAR"):
            m = BUL.search(b)
            if m:
                spans.append((int(m.group(1)) / 1000, int(m.group(2)) / 1000))
    return spans


def norm(w):
    w = w.lower().replace("’", "'")
    w = re.sub(r"[^a-z' ]", " ", w)
    return [t for t in w.split() if t]


def wer(ref, hyp):
    d = np.arange(len(hyp) + 1)
    for i, r in enumerate(ref, 1):
        prev, d[0] = d.copy(), i
        for j, h in enumerate(hyp, 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (r != h))
    return d[len(hyp)] / max(len(ref), 1)


m = pd.read_csv(master, dtype={"media_id": str})
main = pd.read_csv(main_csv, dtype={"media_id": str}).set_index("media_id").text.fillna("")
rows, sil = [], []
for r in m.itertuples():
    jp = f"{asr}/{r.media_id}.json"
    if not os.path.exists(jp):
        continue
    j = json.load(open(jp))
    cands = [os.path.join(ad if r.source_group == "Dementia" else ctl, r.media_id + ".cha")] + [os.path.join(e, r.media_id + ".cha") for e in extra]
    cp = next((c for c in cands if os.path.exists(c)), None)
    spans = par_spans(cp) if cp else []
    allw, parw = [], []
    for w in j["words"]:
        toks = norm(w["w"]); allw += toks
        mid = (w["s"] + w["e"]) / 2
        if any(a <= mid <= b for a, b in spans):
            parw += toks
    ref = norm(main.get(r.media_id, ""))
    rows.append({"media_id": r.media_id, "text_asr_oracle": " ".join(parw), "text_asr_nodiar": " ".join(allw),
                 "n_words_oracle": len(parw), "n_words_nodiar": len(allw), "has_chat": cp is not None,
                 "wer_oracle": wer(ref, parw) if ref else np.nan, "n_ref_words": len(ref)})
    sil.append({"media_id": r.media_id, **j["silence"]})
R = pd.DataFrame(rows); R.to_csv(out, index=False); pd.DataFrame(sil).to_csv(sil_out, index=False)
ok = R.dropna(subset=["wer_oracle"])
print(len(R), "recordings; corpus WER (oracle, PAR):",
      round((ok.wer_oracle * ok.n_ref_words).sum() / ok.n_ref_words.sum(), 3), "; median per-file", round(ok.wer_oracle.median(), 3))
