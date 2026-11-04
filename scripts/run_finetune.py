"""Protocol Addendum B2: fine-tuned RoBERTa-base on manual participant text (main tier), one repeat of the
participant-grouped 5-fold CV (fold_r0) on development. 6 epochs, lr 2e-5, batch 8, wd 0.01, max 256 tokens.

Usage: python run_finetune.py <derived_dir> <out_csv> [model]
"""
import sys, time
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score

der, out = sys.argv[1:3]
name = sys.argv[3] if len(sys.argv) > 3 else "roberta-base"
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(4)
ids = {"media_id": str, "participant_id": str}
t = pd.read_csv(f"{der}/pitt_main_text.csv", dtype=ids)
f = pd.read_csv(f"{der}/protocol_folds.csv", dtype=ids)[["media_id", "locked_test", "fold_r0"]]
d = t.merge(f, on="media_id"); d = d[~d.locked_test.astype(str).eq("True")].reset_index(drop=True)
tok = AutoTokenizer.from_pretrained(name)
enc = tok(list(d.text.fillna("")), truncation=True, max_length=256, padding="max_length", return_tensors="pt")
y = torch.tensor(d.label.values)
P = np.zeros(len(d))
for k in range(5):
    tr, te = np.where(d.fold_r0 != k)[0], np.where(d.fold_r0 == k)[0]
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    dl = DataLoader(tr.tolist(), batch_size=8, shuffle=True, generator=torch.Generator().manual_seed(k))
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * 6 * len(dl)), 6 * len(dl))
    t0 = time.time()
    for ep in range(6):
        model.train()
        for b in dl:
            b = torch.tensor(b)
            loss = model(input_ids=enc.input_ids[b], attention_mask=enc.attention_mask[b], labels=y[b]).loss
            loss.backward(); opt.step(); sch.step(); opt.zero_grad()
        print(f"fold {k} epoch {ep} loss {loss.item():.3f} {time.time()-t0:.0f}s", flush=True)
    model.eval()
    with torch.no_grad():
        for i in range(0, len(te), 16):
            b = te[i:i + 16]
            P[b] = torch.softmax(model(input_ids=enc.input_ids[b], attention_mask=enc.attention_mask[b]).logits, -1)[:, 1].numpy()
    print(f"fold {k} AUC {roc_auc_score(y[te], P[te]):.3f}", flush=True)
pd.DataFrame({"media_id": d.media_id, "label": d.label, "p_finetuned": P}).to_csv(out, index=False)
print("overall AUC", roc_auc_score(d.label, P))
