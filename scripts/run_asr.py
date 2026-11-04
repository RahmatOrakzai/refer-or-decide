"""Whisper ASR (faster-whisper, small.en, int8, CPU) with word timestamps for every Pitt recording,
plus silence-only audio statistics (protocol Addendum B1, B4). Skips recordings already done.

Usage: python run_asr.py <audio_root with Control/ and Dementia/> <out_dir> [model]
"""
import os, sys, json, glob, subprocess, time
import numpy as np
from faster_whisper import WhisperModel

root, out = sys.argv[1:3]
name = sys.argv[3] if len(sys.argv) > 3 else "small.en"
os.makedirs(out, exist_ok=True)
model = WhisperModel(name, device="cpu", compute_type="int8", cpu_threads=4)


def load16k(path, norm=False):
    af = ["-af", "speechnorm=e=50:r=0.0001:l=1"] if norm else []
    raw = subprocess.run(["ffmpeg", "-v", "quiet", "-i", path, *af, "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768


def silence_stats(x, sr=16000):
    f = 400; hop = 160
    n = 1 + (len(x) - f) // hop
    fr = np.lib.stride_tricks.as_strided(x, (n, f), (x.strides[0] * hop, x.strides[0]))
    e = 10 * np.log10((fr ** 2).mean(1) + 1e-12)
    sil = e < e.max() - 40
    spec = np.abs(np.fft.rfft(fr[sil] * np.hanning(f), axis=1)) if sil.any() else np.zeros((1, f // 2 + 1))
    freqs = np.fft.rfftfreq(f, 1 / sr)
    cent = (spec * freqs).sum(1) / (spec.sum(1) + 1e-12)
    return {"sil_frac": float(sil.mean()), "sil_energy_mean": float(e[sil].mean()) if sil.any() else 0.0,
            "sil_energy_sd": float(e[sil].std()) if sil.any() else 0.0, "sil_centroid_mean": float(cent.mean()),
            "sil_centroid_sd": float(cent.std())}


files = sorted(glob.glob(f"{root}/*/*.wav"))
for p in files:
    mid = os.path.basename(p)[:-4]
    dst = f"{out}/{mid}.json"
    if os.path.exists(dst):
        continue
    t0 = time.time()
    x = load16k(p)                   # raw level, for the silence statistics
    xn = load16k(p, norm=True)       # speech-normalised, for ASR (quiet participants are otherwise skipped)
    segs, info = model.transcribe(xn, language="en", beam_size=5, word_timestamps=True, vad_filter=False,
                                  condition_on_previous_text=False, no_speech_threshold=None,
                                  log_prob_threshold=None, compression_ratio_threshold=None)
    words = [{"w": w.word.strip(), "s": round(w.start, 3), "e": round(w.end, 3), "p": round(w.probability, 3)}
             for s in segs for w in (s.words or [])]
    json.dump({"media_id": mid, "duration": len(x) / 16000, "words": words, "silence": silence_stats(x)},
              open(dst + ".tmp", "w"))
    os.replace(dst + ".tmp", dst)
    print(f"{mid} {len(x)/16000:.0f}s audio, {len(words)} words, {time.time()-t0:.1f}s", flush=True)
