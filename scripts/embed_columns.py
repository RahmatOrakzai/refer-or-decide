"""Frozen-encoder embeddings (mean-pooled last layer) for several text columns of a CSV.

Usage: python embed_columns.py <csv> <out_npz> <model_name> <col> [<col> ...]
"""
import sys
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModel

src, out, name, cols = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
torch.set_num_threads(4)
tok = AutoTokenizer.from_pretrained(name); mdl = AutoModel.from_pretrained(name).eval()
t = pd.read_csv(src, dtype={"media_id": str}).fillna("")
res = {"media_id": t.media_id.values}
for col in cols:
    embs = []
    with torch.no_grad():
        for i in range(0, len(t), 8):
            b = tok(list(t[col].iloc[i:i + 8]), padding=True, truncation=True, max_length=512, return_tensors="pt")
            h = mdl(**b).last_hidden_state; m = b["attention_mask"].unsqueeze(-1).float()
            embs.append(((h * m).sum(1) / m.sum(1)).numpy())
    res[col] = np.vstack(embs); print(col, res[col].shape, flush=True)
np.savez_compressed(out, **res)
