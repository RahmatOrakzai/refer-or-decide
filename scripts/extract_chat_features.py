"""Per-recording CHAT features for the participant (PAR) and the interviewer (INV).

Usage: python extract_chat_features.py <master_csv> <ad_cha_dir> <control_cha_dir> <out_csv> [extra_dir ...]
"""
import re, sys, os, statistics as st
import pandas as pd

master_csv, ad_dir, ctl_dir, out_csv = sys.argv[1:5]
extra = sys.argv[5:]

# Cookie Theft information units (Croisile et al. style keyword list)
IU = {"boy", "girl", "woman", "mother", "lady", "kid", "children", "cookie", "cookies", "jar", "stool",
      "sink", "water", "overflow", "overflowing", "spilling", "dish", "dishes", "plate", "cup", "window",
      "curtain", "curtains", "cupboard", "cabinet", "kitchen", "fall", "falling", "tipping", "wash", "washing",
      "drying", "dry", "reach", "reaching", "taking", "steal", "stealing", "shh", "outside", "garden", "floor"}
FILLED = {"&-uh", "&-um", "&-er", "&-hm", "&-mm"}
WORD_T = re.compile(r"(\S+?)\s+\x15?(\d+)_(\d+)\x15?")


def find(mid, group):
    cands = [os.path.join(ad_dir if group == "Dementia" else ctl_dir, mid + ".cha")] + \
            [os.path.join(d, mid + ".cha") for d in extra]
    return next((c for c in cands if os.path.exists(c)), None)


def parse(path):
    lines = open(path, encoding="utf-8", errors="replace").read().split("\n")
    # join continuation lines (tab-indented)
    tiers, cur = [], None
    for ln in lines:
        if ln.startswith("\t") and cur is not None:
            cur[1] += " " + ln.strip()
        elif ln.startswith("*") or ln.startswith("%"):
            cur = [ln.split(":", 1)[0], ln.split(":", 1)[1].strip() if ":" in ln else ""]
            tiers.append(cur)
    utts = []  # (speaker, text, start, end, words[(w,s,e)])
    for i, (tag, txt) in enumerate(tiers):
        if not tag.startswith("*"):
            continue
        spk = tag[1:]
        m = re.search(r"\x15?(\d+)_(\d+)\x15?\s*$", txt)
        s, e = (int(m.group(1)), int(m.group(2))) if m else (None, None)
        words = []
        for t2, x2 in tiers[i + 1:]:
            if t2.startswith("*"):
                break
            if t2 == "%wor":
                words = [(w, int(a), int(b)) for w, a, b in WORD_T.findall(x2)]
        utts.append((spk, txt, s, e, words))
    return utts


def tokens(txt):
    txt = re.sub(r"\x15?\d+_\d+\x15?", " ", txt)
    txt = re.sub(r"\[[^\]]*\]", " ", txt)
    return [t for t in re.split(r"\s+", txt) if t and t not in {".", "?", "!", "+...", "+/.", "+//.", "<", ">"}]


rows = []
master = pd.read_csv(master_csv, dtype={"media_id": str, "participant_id": str})
for _, r in master.iterrows():
    p = find(r.media_id, r.source_group)
    if p is None:
        rows.append({"media_id": r.media_id, "transcript_found": False}); continue
    utts = parse(p)
    par = [u for u in utts if u[0] == "PAR"]
    inv = [u for u in utts if u[0] == "INV"]
    f = {"media_id": r.media_id, "transcript_found": True}
    ptoks = [t for u in par for t in tokens(u[1])]
    lex = [t.lower().strip("<>,") for t in ptoks if not t.startswith(("&", "xxx", "(", "+", "[")) and re.match(r"[A-Za-z']", t)]
    f["par_utts"] = len(par)
    f["par_words"] = len(lex)
    f["par_ttr"] = len(set(lex)) / len(lex) if lex else None
    f["par_mlu_words"] = len(lex) / len(par) if par else None
    f["par_filled_pauses"] = sum(t in FILLED for t in ptoks)
    f["par_fragments"] = sum(t.startswith("&+") for t in ptoks)
    f["par_repetitions"] = sum(u[1].count("[/]") for u in par)
    f["par_retracings"] = sum(u[1].count("[//]") for u in par)
    f["par_unintelligible"] = sum(t.startswith("xxx") for t in ptoks)
    f["par_info_units"] = len({w.rstrip("s") if w not in IU else w for w in lex if w in IU})
    f["par_info_unit_tokens"] = sum(w in IU for w in lex)
    # timing from %wor
    pw = sorted([w for u in par for w in u[4]], key=lambda x: x[1])
    if pw:
        gaps = [b[1] - a[2] for a, b in zip(pw, pw[1:]) if b[1] - a[2] > 0]
        span = (pw[-1][2] - pw[0][1]) / 1000
        f["par_span_s"] = span
        f["par_speech_s"] = sum(e - s for _, s, e in pw) / 1000
        f["par_words_per_min"] = len(lex) / (span / 60) if span > 0 else None
        f["par_pauses_gt_0_5s"] = sum(g > 500 for g in gaps)
        f["par_pauses_gt_2s"] = sum(g > 2000 for g in gaps)
        f["par_pause_rate_per_min"] = f["par_pauses_gt_0_5s"] / (span / 60) if span > 0 else None
        f["par_mean_pause_s"] = st.mean([g for g in gaps if g > 250]) / 1000 if any(g > 250 for g in gaps) else 0
        f["par_phonation_ratio"] = f["par_speech_s"] / span if span > 0 else None
    # interviewer behaviour
    itoks = [t for u in inv for t in tokens(u[1])]
    ilex = [t for t in itoks if not t.startswith("&") and re.match(r"[A-Za-z']", t)]
    f["inv_turns"] = len(inv)
    f["inv_words"] = len(ilex)
    first_par = next((i for i, u in enumerate(utts) if u[0] == "PAR" and len(tokens(u[1])) > 2), None)
    f["inv_turns_after_start"] = sum(1 for i, u in enumerate(utts) if u[0] == "INV" and first_par is not None and i > first_par)
    f["inv_word_share"] = len(ilex) / (len(ilex) + len(lex)) if (ilex or lex) else None
    joined = " ".join(u[1].lower() for u in inv)
    f["inv_prompts_anything_else"] = len(re.findall(r"anything else|what else|anything more|is that all|any more", joined))
    f["inv_encouragement"] = len(re.findall(r"\bokay\b|\bgood\b|\bmhm\b|\balright\b|\bfine\b", joined))
    # response latency: INV utterance end -> next PAR start
    lats = [b[2] - a[3] for a, b in zip(utts, utts[1:])
            if a[0] == "INV" and b[0] == "PAR" and a[3] is not None and b[2] is not None and b[2] >= a[3]]
    f["par_mean_response_latency_s"] = st.mean(lats) / 1000 if lats else None
    rows.append(f)

out = pd.DataFrame(rows)
out.to_csv(out_csv, index=False)
print(out.shape, "found:", out.transcript_found.sum())
