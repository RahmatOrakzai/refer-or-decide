"""Automated audio QC for Pitt WAVs. Appends one row per file to <out_csv>.
Checks: sha256 vs manifest, duration, channels, peak, clipping, RMS level, silence share, rough SNR, DC offset.
Usage: python audio_qc.py <manifest_csv> <out_csv> <wav> [<wav> ...]"""
import sys, os, hashlib, numpy as np, pandas as pd, soundfile as sf
man = pd.read_csv(sys.argv[1], dtype=str).set_index("media_id"); out = sys.argv[2]
rows = []
for f in sys.argv[3:]:
    mid = os.path.splitext(os.path.basename(f))[0]
    h = hashlib.sha256(open(f, "rb").read()).hexdigest()
    y, sr = sf.read(f, always_2d=True); ch = y.shape[1]; y = y.mean(axis=1)
    frame = int(0.025 * sr); n = len(y) // frame
    e = 10 * np.log10(np.mean(y[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
    rows.append(dict(media_id=mid, sha256_matches_manifest=(h == man.loc[mid, "sha256"]) if mid in man.index else None,
        sample_rate=sr, channels=ch, duration_s=round(len(y) / sr, 2),
        peak_dbfs=round(20 * np.log10(np.abs(y).max() + 1e-12), 2),
        clipped_pct=round(100 * np.mean(np.abs(y) >= 0.999), 4),
        rms_dbfs=round(10 * np.log10(np.mean(y ** 2) + 1e-12), 2),
        silence_pct=round(100 * np.mean(e < -50), 1),
        snr_est_db=round(np.percentile(e, 95) - np.percentile(e, 10), 1),
        dc_offset=round(float(y.mean()), 5)))
df = pd.DataFrame(rows); df.to_csv(out, mode="a", header=not os.path.exists(out), index=False)
print(len(df), "files;", df[["rms_dbfs", "clipped_pct", "snr_est_db"]].describe().loc[["min", "50%", "max"]].to_string())
