#!/usr/bin/env python3
"""
Control arm `frozen`: the same prediction head trained on FROZEN ESM-2 representations (no LoRA adapters).

Answers "what does low-rank adaptation of the language model add over using its pretrained embeddings?".
The sequence representation [h_wt, h_mut, h_mut - h_wt] is computed once with the unmodified ESM-2 and
cached; the fusion MLP + pathogenicity head (identical to HEARS-LM) are then trained on it.
The number of epochs is chosen on the development set from EPOCH_GRID (the head alone converges more
slowly than the adapted model); everything else follows the main configuration.
Outputs have the same layout as train.py (runs/frozen/seed*/preds_<partition>.csv, result.json).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hearslm import DATA, DEFAULT_CONFIG, ROOT, VariantSet, load_reference_sequences, lr_lambda, set_seed  # noqa: E402

EPOCH_GRID = [5, 10, 20, 40]
PARTS = ['fit', 'dev', 'test_db', 'test_time', 'test_gene']


@torch.no_grad()
def embed(backbone_name, df, device, batch=64):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(backbone_name)
    bb = AutoModel.from_pretrained(backbone_name, add_pooling_layer=False).to(device).eval()
    vs = VariantSet(df, load_reference_sequences(), tok, DEFAULT_CONFIG['window'])
    out = []
    for i in range(0, len(vs), batch):
        idx = torch.arange(i, min(i + batch, len(vs)))
        B = len(idx)
        ids = torch.cat([vs.ids_wt[idx], vs.ids_mut[idx]]).to(device)
        mask = torch.cat([vs.mask_wt[idx], vs.mask_mut[idx]]).to(device)
        h = bb(input_ids=ids, attention_mask=mask).last_hidden_state
        hp = h[torch.arange(2 * B, device=device), vs.tokpos[idx].repeat(2).to(device)]
        out.append(torch.cat([hp[:B], hp[B:], hp[B:] - hp[:B]], dim=-1).float().cpu())
    del bb
    return torch.cat(out).numpy()


class Head(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.fusion = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, 128), nn.ReLU())
        self.out = nn.Linear(128, 1)

    def forward(self, x):
        return self.out(self.fusion(x)).squeeze(-1)


def train_head(Xtr, ytr, epochs, seed, cfg):
    set_seed(seed)
    model = Head(Xtr.shape[1], cfg['head_dropout'])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr_head'], weight_decay=cfg['weight_decay'])
    steps = int(np.ceil(len(Xtr) / cfg['batch_size'])) * epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, steps, int(cfg['warmup_frac'] * steps)))
    gen = torch.Generator().manual_seed(seed)
    X, y = torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.float32)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(X), generator=gen)
        for i in range(0, len(X), cfg['batch_size']):
            idx = perm[i:i + cfg['batch_size']]
            loss = F.binary_cross_entropy_with_logits(model(X[idx]), y[idx])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
            opt.step(); sched.step()
    return model.eval()


def main():
    backbone = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG['backbone']
    arm = sys.argv[2] if len(sys.argv) > 2 else 'frozen'
    n_seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    cfg = dict(DEFAULT_CONFIG); cfg.update({'backbone': backbone, 'lora_r': 0, 'beta_ddg': 0.0, 'arm': arm})
    v = pd.read_csv(DATA / 'variants_v8.csv', low_memory=False)
    v = v[v.partition.isin(PARTS)].reset_index(drop=True)
    cache = ROOT / f'runs/_cache/frozen_{backbone.split("/")[-1]}.npy'
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        E = np.load(cache)
    else:
        t0 = time.time()
        E = embed(backbone, v, cfg['device'])
        np.save(cache, E)
        pd.Series(v.variant_id).to_csv(cache.with_suffix('.ids.csv'), index=False)
        print(f'embedded {len(v)} variants in {(time.time() - t0) / 60:.1f} min')
    assert len(E) == len(v)
    idx = {p: np.flatnonzero((v.partition == p).to_numpy()) for p in PARTS}
    Xtr, ytr = E[idx['fit']], v.y.to_numpy()[idx['fit']]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Z = ((E - mu) / sd).astype(np.float32)

    # choose the number of epochs on the development set (seed 0)
    dev_curve = {}
    for ep in EPOCH_GRID:
        m = train_head(Z[idx['fit']], ytr, ep, 0, cfg)
        with torch.no_grad():
            p = torch.sigmoid(m(torch.tensor(Z[idx['dev']]))).numpy()
        dev_curve[ep] = float(roc_auc_score(v.y.to_numpy()[idx['dev']], p))
    best_ep = max(dev_curve, key=dev_curve.get)
    print('dev AUROC by epochs:', dev_curve, '-> epochs =', best_ep)
    cfg['epochs'] = best_ep
    for seed in range(n_seeds):
        out = ROOT / 'runs' / arm / f'seed{seed}'
        out.mkdir(parents=True, exist_ok=True)
        m = train_head(Z[idx['fit']], ytr, best_ep, seed, cfg)
        final = {}
        for part in PARTS[1:]:
            with torch.no_grad():
                logit = m(torch.tensor(Z[idx[part]])).numpy()
            pr = pd.DataFrame({'variant_id': v.variant_id.to_numpy()[idx[part]], 'y': v.y.to_numpy()[idx[part]],
                               'logit': logit, 'prob': 1 / (1 + np.exp(-logit))})
            pr.to_csv(out / f'preds_{part}.csv', index=False)
            final[part] = {'n': len(pr), 'auroc': float(roc_auc_score(pr.y, pr.prob))}
        c = dict(cfg); c['seed'] = seed
        json.dump({'config': c, 'n_train': int(len(ytr)), 'n_features': 0, 'dev_curve_epochs': dev_curve,
                   'final': final}, open(out / 'result.json', 'w'), indent=1)
        print(f'seed {seed}: dev AUROC {final["dev"]["auroc"]:.4f}')


if __name__ == '__main__':
    main()
