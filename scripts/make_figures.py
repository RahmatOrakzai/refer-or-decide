"""Paper figures (protocol Addendum B5). Reads only saved result files.

Usage: python make_figures.py <results_dir> <fig_dir>
"""
import sys, os
import numpy as np, pandas as pd
from scipy.stats import binom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

res, fig = sys.argv[1:3]
os.makedirs(fig, exist_ok=True)
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "ink": "#2b2b2a", "muted": "#8a8984", "grid": "#e4e3de"}
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": C["muted"], "axes.labelcolor": C["ink"],
                     "xtick.color": C["muted"], "ytick.color": C["muted"], "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.5,
                     "lines.linewidth": 1.6, "legend.frameon": False, "savefig.bbox": "tight", "pdf.fonttype": 42})
W = 3.45  # IEEE single column, inches


def save(name):
    plt.savefig(f"{fig}/{name}.pdf"); plt.savefig(f"{fig}/{name}.png", dpi=200); plt.close()


# Fig 1: shortcut ledger
b = pd.read_csv(f"{res}/baselines.csv")
keep = [("Demographics only (negative control)", "Demographics only"), ("Interviewer only (negative control)", "Interviewer only"),
        ("Speech length only (negative control)", "Speech length only"), ("F: participant features", "F: speech features"),
        ("F-adj: features, demographics+interviewer removed", "F-adj: features, shortcuts removed"),
        ("T: roberta-base, pause tokens", "T: RoBERTa text"), ("T-adj: roberta-base, pause tokens, adjusted", "T-adj: text, shortcuts removed")]
rows = []
for k, lab in keep:
    r = b[b.model == k].iloc[0]
    a, lo, hi = [float(v) for v in r["AUC (all visits)"].replace("[", "").replace("]", "").replace(",", "").split()]
    rows.append((lab, a, lo, hi, float(r["AUC age/edu-matched"])))
fig1, ax = plt.subplots(figsize=(W, 2.4))
for i, (lab, a, lo, hi, mt) in enumerate(rows[::-1]):
    col = C["muted"] if i >= 4 else C["blue"]
    ax.plot([lo, hi], [i, i], color=col, lw=1.2)
    ax.plot(a, i, "o", color=col, ms=5, mec="white", mew=1)
    ax.plot(mt, i, "D", color=C["orange"], ms=4, mec="white", mew=0.8)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]])
ax.axvline(0.5, color=C["muted"], lw=0.8, ls=":")
ax.set_xlim(0.45, 1.0); ax.set_xlabel("AUC (participant-grouped CV, 95% CI)")
ax.plot([], [], "o", color=C["ink"], label="All development recordings"); ax.plot([], [], "D", color=C["orange"], label="Age/education-matched subset")
ax.legend(loc="upper center", bbox_to_anchor=(0.35, -0.22), ncol=2, fontsize=6.5); ax.grid(axis="y", visible=False)
save("fig1_shortcut_ledger")

# Fig 2: (a) realised risk vs automatic-decision rate for T-adj; (b) how often each method breaks its budget
fig2, axs = plt.subplots(1, 2, figsize=(2 * W, 2.4))
ax = axs[0]
cu = pd.read_csv(f"{res}/referral_risk_curve_T-adj.csv").groupby("lambda")[["auto_rate", "err_auto"]].mean().sort_values("auto_rate")
ax.plot(cu.auto_rate, cu.err_auto, color=C["muted"], lw=1.2, label="Every threshold (realised)")
R = pd.read_csv(f"{res}/referral_resplits_T-adj.csv")
for meth, col, mk, lab in (("conformal", C["orange"], "s", "Conformal"), ("LTT-bonferroni", C["blue"], "o", "LTT (Bonferroni)"),
                           ("LTT-fixed-sequence", C["aqua"], "^", "LTT (fixed sequence)")):
    g = R[R.method == meth].groupby("alpha").agg(a=("auto_rate", "mean"), e=("err_auto", lambda s: s[s.notna()].mean()))
    g = g[g.a > 0.005]
    ax.plot(g.a, g.e, mk, color=col, ms=6, mec="white", mew=1, label=lab, ls="none")
    if meth == "conformal":
        for al, r in g.iterrows():
            ax.annotate(f"α={al:.2f}", (r.a, r.e), xytext=(-30, 4), textcoords="offset points", fontsize=5.5, color=C["ink"])
ax.set_xlim(0, 1); ax.set_xlabel("Automatic-decision rate"); ax.set_ylabel("Error among automatic decisions")
ax.set_title("(a) T-adj: error vs. cases decided", fontsize=8, color=C["ink"]); ax.legend(fontsize=6, loc="upper left")
ax = axs[1]
alphas = [0.05, 0.10, 0.15, 0.20]; wbar = 0.2
for j, (meth, col, lab) in enumerate((("conformal", C["orange"], "Conformal"), ("LTT-bonferroni", C["blue"], "LTT (Bonferroni)"),
                                      ("LTT-fixed-sequence", C["aqua"], "LTT (fixed sequence)"))):
    v = [(R[(R.method == meth) & (R.alpha == al)].err_auto > al).mean() for al in alphas]
    ax.bar(np.arange(4) + (j - 1) * (wbar + 0.03), v, wbar, color=col, label=lab, edgecolor="white", linewidth=0.5)
ax.axhline(0.10, color=C["ink"], lw=0.8, ls="--"); ax.text(3.45, 0.115, "δ = 0.10", ha="right", fontsize=6, color=C["ink"])
ax.set_xticks(range(4)); ax.set_xticklabels([f"α={a:.2f}" for a in alphas]); ax.grid(axis="x", visible=False)
ax.set_ylabel("Resplits with error > α"); ax.set_ylim(0, 0.8)
ax.set_title("(b) T-adj: how often the budget is broken", fontsize=8, color=C["ink"]); ax.legend(fontsize=6, loc="upper right")
save("fig2_risk_vs_automatic_rate")

# Fig 3: calibration participants needed, alpha grid
DELTA, K = 0.10, 9
fig3, ax = plt.subplots(figsize=(W, 2.3))
for model, col, lab in (("F-adj", C["orange"], "F-adj"), ("F", C["aqua"], "F"), ("T-adj", C["blue"], "T-adj")):
    cu = pd.read_csv(f"{res}/referral_risk_curve_{model}.csv").groupby("lambda")[["auto_rate", "err_auto"]].mean()
    xs, ys = [], []
    for alpha in np.round(np.arange(0.06, 0.251, 0.01), 2):
        best = None
        for lam, (a, e) in cu.iterrows():
            if not e < alpha:
                continue
            for n in range(20, 5001, 5):
                nd = int(round(n * a))
                if nd < 1:
                    continue
                k = int(binom.ppf(DELTA / K, nd, alpha))
                while k >= 0 and binom.cdf(k, nd, alpha) > DELTA / K:
                    k -= 1
                if k >= 0 and binom.cdf(k, nd, e) >= 0.8:
                    best = n if best is None else min(best, n); break
        if best:
            xs.append(alpha); ys.append(best)
    ax.plot(xs, ys, color=col, label=lab)
    ax.annotate(lab, (xs[-1], ys[-1]), xytext=(3, 0), textcoords="offset points", fontsize=6.5, color=C["ink"], va="center")
ax.axhline(246, color=C["muted"], lw=0.8, ls="--"); ax.text(0.245, 262, "all of Pitt (246)", ha="right", fontsize=6, color=C["muted"])
ax.set_yscale("log"); ax.set_xlabel("Error budget α"); ax.set_ylabel("Calibration participants needed")
ax.set_xlim(0.055, 0.27)
save("fig3_calibration_size")

# Fig 4: minimum speech, two panels sharing x
ms = pd.read_csv(f"{res}/min_speech/min_speech_summary.csv")
x = [10, 20, 30, 50, 75, 100, 125]
labels = ["10", "20", "30", "50", "75", "100", "all"]
fig4, axs = plt.subplots(2, 1, figsize=(W, 3.2), sharex=True)
axs[0].plot(x, ms.AUC, "o-", color=C["blue"], ms=4); axs[0].set_ylabel("AUC")
for col, c, lab in (("auto (per-k cert.) a=0.2", C["blue"], "Bonferroni, per length"),
                    ("auto (fixed seq.) a=0.2", C["aqua"], "Fixed sequence, per length"),
                    ("auto (joint over k) a=0.2", C["orange"], "Bonferroni, joint over lengths")):
    axs[1].plot(x, ms[col], "o-", color=c, ms=4, label=lab)
axs[1].set_ylabel("Certified automatic rate\n(α = 0.20)"); axs[1].set_xlabel("Participant words used")
axs[1].set_xticks(x); axs[1].set_xticklabels(labels); axs[1].legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.45, -0.32), ncol=2)
save("fig4_minimum_speech")
print("figures written to", fig)

# Fig 5: where the model holds up - AUC by evaluation setting, manual vs Whisper training text
import re as _re
def _auc(s): return [float(v) for v in _re.findall(r"[0-9.]+", s)][:3]
asr = pd.read_csv(f"{res}/asr/asr_cv.csv")
e_man = pd.read_csv(f"{res}/external/external_discrimination.csv"); e_asr = pd.read_csv(f"{res}/external_asr/external_discrimination.csv")
lk_man = pd.read_csv(f"{res}/locked_test_T-adj.csv"); lk_asr = pd.read_csv(f"{res}/asr/locked_test_Tadj-asr-oracle.csv")
lk = lambda L: L[L.test == "one visit per participant"].auc.iloc[0]
rows = [("Pitt CV\n(T-adj)", _auc(asr[(asr.input == "manual_main") & (asr.model == "T-adj")]["AUC [95% CI]"].iloc[0]),
         _auc(asr[(asr.input == "asr_oracle") & (asr.model == "T-adj")]["AUC [95% CI]"].iloc[0])),
        ("Pitt locked\ntest (T-adj)", [lk(lk_man)] * 3, [lk(lk_asr)] * 3),
        ("Lu\n(dementia)", _auc(e_man[(e_man.corpus == "Lu")]["AUC all recordings"].iloc[0]), _auc(e_asr[(e_asr.corpus == "Lu")]["AUC all recordings"].iloc[0])),
        ("Delaware\n(MCI)", _auc(e_man[(e_man.corpus == "Delaware") & (e_man.model == "T (plain text)")]["AUC all recordings"].iloc[0]),
         _auc(e_asr[(e_asr.corpus == "Delaware") & (e_asr.model == "T (plain text)")]["AUC all recordings"].iloc[0]))]
fig5, ax = plt.subplots(figsize=(W, 2.3))
for i, (lab, m_, a_) in enumerate(rows):
    for dx, v, col in ((-0.12, m_, C["blue"]), (0.12, a_, C["orange"])):
        ax.plot([i + dx] * 2, [v[1], v[2]], color=col, lw=1.2)
        ax.plot(i + dx, v[0], "o", color=col, ms=5, mec="white", mew=1)
ax.plot([], [], "o", color=C["blue"], label="Trained on manual transcripts"); ax.plot([], [], "o", color=C["orange"], label="Trained on Whisper transcripts")
ax.axhline(0.5, color=C["muted"], lw=0.8, ls=":"); ax.set_ylim(0.45, 1.0)
ax.set_xticks(range(len(rows))); ax.set_xticklabels([r[0] for r in rows], fontsize=6.5); ax.grid(axis="x", visible=False)
ax.set_ylabel("AUC (95% CI where available)"); ax.legend(fontsize=6, loc="lower left")
save("fig5_transfer")
print("fig5 written")
