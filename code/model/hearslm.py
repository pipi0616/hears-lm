#!/usr/bin/env python3
"""
HEARS-LM (revision v8): model, data handling and training loop.

Model (identical in structure to the released two-head model; nothing that the model predicts is an input):
  1. Sequence encoder: ESM-2 with LoRA adapters (rank r) on the query and value projections of every
     self-attention layer; all original ESM-2 weights are frozen.
  2. For a missense variant at residue i of the reference protein, the wild-type and the mutant sequence
     windows (residues i-W .. i+W, W = 32) are both encoded; the final-layer hidden states at residue i,
     h_wt and h_mut (dimension H), form the sequence representation [h_wt, h_mut, h_mut - h_wt] (3H).
     Optionally ('use_lm_logits') the pretrained masked-language-model head is applied to h_wt, giving the
     log-probabilities of the 20 amino acids at residue i in the wild-type context; these, the log-likelihood
     ratio of the substitution and one-hot codes of the two amino acids are appended (the head is frozen).
  3. Optional tabular descriptors x (standardised with training-set statistics; missing -> 0 = training mean)
     are encoded by a 2-layer MLP (F -> 128 -> 64) and concatenated to the sequence representation.
  4. Fusion MLP (-> 256 -> 128, ReLU, dropout) followed by two linear heads:
        pathogenicity logit (binary cross-entropy) and, optionally, a ddG regression head
        (mean-squared error on standardised DDGun ddG; only variants with a ddG target contribute).
  Loss = L_path + beta * L_ddG.

Everything that defines a run is in the config dict; it is saved next to the predictions.
"""
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

ROOT = Path(__file__).resolve().parents[2]            # revision_2026-10/
DATA = ROOT / 'data'

DEFAULT_CONFIG = {
    'backbone': 'facebook/esm2_t30_150M_UR50D',
    'lora_r': 16, 'lora_alpha': 32, 'lora_dropout': 0.1, 'lora_targets': ['query', 'value'],
    'window': 32,
    'feature_groups': [],          # e.g. ['cons', 'struct', 'hear']; [] = sequence only
    'use_lm_logits': False,        # add the language model's own substitution log-probabilities (see HearsLM)
    'use_sequence': True,          # False = descriptors-only model (no language model)
    'beta_ddg': 0.3,               # 0 disables the ddG head
    'epochs': 3, 'batch_size': 8,
    'lr_lora': 1e-4, 'lr_head': 1e-3, 'weight_decay': 0.01,
    'warmup_frac': 0.06, 'grad_clip': 1.0, 'head_dropout': 0.3,
    'seed': 0,
    'device': 'mps' if torch.backends.mps.is_available() else 'cpu',
}

AA = 'ACDEFGHIKLMNPQRSTVWY'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ----------------------------------------------------------------------------------------------- data
def load_reference_sequences():
    ref = pd.read_csv(DATA / 'sources/reference_proteins.csv')
    return dict(zip(ref.gene, ref.sequence))


def feature_columns(groups):
    spec = json.load(open(DATA / 'features/feature_groups.json'))
    cols = []
    for g in groups:
        cols += spec[g]
    return list(dict.fromkeys(cols))


def load_feature_table(groups, path=None):
    """Return (DataFrame indexed by variant_id, ordered list of columns) for the requested groups.
    path: feature table to read (default: the labelled-variant table data/features/features_v8.csv)."""
    if not groups:
        return None, []
    cols = feature_columns(groups)
    path = path or DATA / 'features/features_v8.csv'
    ft = pd.read_csv(path, usecols=['variant_id'] + cols).set_index('variant_id')
    return ft, cols


def load_ddg_table():
    p = DATA / 'annotations/ddgun_v8.csv'
    if not p.exists():
        return {}
    d = pd.read_csv(p, usecols=['variant_id', 'ddg'])
    return dict(zip(d.variant_id, d.ddg))


class VariantSet:
    """Pre-tokenised variants. Holds CPU tensors; batches are moved to the device by the caller."""

    def __init__(self, df, seqs, tokenizer, window, feat_table=None, feat_cols=None, ddg=None):
        self.df = df.reset_index(drop=True)
        W = window
        wt_seqs, mut_seqs, tokpos, aa_wt, aa_mut = [], [], [], [], []
        for gene, pos, ref, alt in zip(self.df.gene, self.df.ref_pos, self.df.ref1, self.df.alt1):
            s = seqs[gene]
            pos = int(pos)
            assert s[pos - 1] == ref, f'reference mismatch {gene}:{ref}{pos}{alt}'
            start, end = max(0, pos - 1 - W), min(len(s), pos + W)
            wt = s[start:end]
            rel = pos - 1 - start
            mut = wt[:rel] + alt + wt[rel + 1:]
            wt_seqs.append(wt); mut_seqs.append(mut); tokpos.append(rel + 1)      # +1 for <cls>
            aa_wt.append(AA.index(ref)); aa_mut.append(AA.index(alt))
        L = 2 * W + 3
        enc_wt = tokenizer(wt_seqs, return_tensors='pt', padding='max_length', max_length=L, truncation=True)
        enc_mut = tokenizer(mut_seqs, return_tensors='pt', padding='max_length', max_length=L, truncation=True)
        self.ids_wt, self.mask_wt = enc_wt['input_ids'], enc_wt['attention_mask']
        self.ids_mut, self.mask_mut = enc_mut['input_ids'], enc_mut['attention_mask']
        self.tokpos = torch.tensor(tokpos, dtype=torch.long)
        self.aa_wt = torch.tensor(aa_wt, dtype=torch.long)
        self.aa_mut = torch.tensor(aa_mut, dtype=torch.long)
        self.aa_token_ids = torch.tensor(tokenizer.convert_tokens_to_ids(list(AA)), dtype=torch.long)
        ycol = 'y' if 'y' in self.df else None
        y = self.df[ycol].to_numpy(dtype=np.float32) if ycol else np.full(len(self.df), np.nan, np.float32)
        self.y = torch.tensor(y)
        if feat_cols:
            x = feat_table.reindex(self.df.variant_id)[feat_cols].to_numpy(dtype=np.float32)
        else:
            x = np.zeros((len(self.df), 0), np.float32)
        self.x_raw = x
        self.x = None
        d = np.array([ddg.get(v, np.nan) if ddg else np.nan for v in self.df.variant_id], dtype=np.float32)
        self.ddg_raw = d
        self.ddg = None
        self.has_ddg = torch.tensor(~np.isnan(d))

    def __len__(self):
        return len(self.df)

    def standardise(self, stats):
        if self.x_raw.shape[1]:
            x = (self.x_raw - stats['x_mean']) / stats['x_std']
            x = np.where(np.isnan(x), 0.0, x)                    # missing -> training mean
            self.x = torch.tensor(np.clip(x, -8, 8), dtype=torch.float32)
        else:
            self.x = torch.zeros((len(self.df), 0))
        d = (self.ddg_raw - stats['ddg_mean']) / stats['ddg_std']
        self.ddg = torch.tensor(np.where(np.isnan(d), 0.0, d), dtype=torch.float32)

    def batch(self, idx, device):
        return {'ids_wt': self.ids_wt[idx].to(device), 'mask_wt': self.mask_wt[idx].to(device),
                'ids_mut': self.ids_mut[idx].to(device), 'mask_mut': self.mask_mut[idx].to(device),
                'tokpos': self.tokpos[idx].to(device), 'x': self.x[idx].to(device),
                'aa_wt': self.aa_wt[idx].to(device), 'aa_mut': self.aa_mut[idx].to(device),
                'aa_token_ids': self.aa_token_ids.to(device),
                'y': self.y[idx].to(device), 'ddg': self.ddg[idx].to(device),
                'has_ddg': self.has_ddg[idx].to(device)}


def fit_stats(train_set):
    stats = {}
    if train_set.x_raw.shape[1]:
        m = np.nanmean(train_set.x_raw, axis=0)
        s = np.nanstd(train_set.x_raw, axis=0)
        s = np.where((s < 1e-8) | np.isnan(s), 1.0, s)
        m = np.where(np.isnan(m), 0.0, m)
        stats['x_mean'], stats['x_std'] = m.astype(np.float32), s.astype(np.float32)
    else:
        stats['x_mean'], stats['x_std'] = np.zeros(0, np.float32), np.ones(0, np.float32)
    d = train_set.ddg_raw[~np.isnan(train_set.ddg_raw)]
    stats['ddg_mean'] = float(d.mean()) if len(d) else 0.0
    stats['ddg_std'] = float(d.std()) if len(d) and d.std() > 0 else 1.0
    return stats


# ---------------------------------------------------------------------------------------------- model
class HearsLM(nn.Module):
    def __init__(self, cfg, n_feat):
        super().__init__()
        self.cfg = cfg
        self.use_seq = cfg['use_sequence']
        self.n_feat = n_feat
        dim = 0
        if self.use_seq:
            from peft import LoraConfig, get_peft_model
            from transformers import AutoModel, AutoModelForMaskedLM
            self.use_lm = bool(cfg.get('use_lm_logits', False))
            if self.use_lm:
                mlm = AutoModelForMaskedLM.from_pretrained(cfg['backbone'])
                bb, self.lm_head = mlm.esm, mlm.lm_head          # pretrained masked-LM head, kept frozen
                for p in self.lm_head.parameters():
                    p.requires_grad_(False)
            else:
                bb = AutoModel.from_pretrained(cfg['backbone'], add_pooling_layer=False)
            if cfg['lora_r'] > 0:
                bb = get_peft_model(bb, LoraConfig(r=cfg['lora_r'], lora_alpha=cfg['lora_alpha'],
                                                   target_modules=cfg['lora_targets'],
                                                   lora_dropout=cfg['lora_dropout'], bias='none'))
            else:
                for p in bb.parameters():
                    p.requires_grad_(False)
            self.backbone = bb
            dim += 3 * bb.config.hidden_size
            if self.use_lm:
                dim += 20 + 1 + 20 + 20     # log p(aa | wt context), LLR(mut vs wt), one-hot wt, one-hot mut
        if n_feat:
            self.feat_encoder = nn.Sequential(nn.Linear(n_feat, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 64))
            dim += 64
        assert dim > 0, 'model needs a sequence encoder and/or descriptors'
        self.fusion = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Dropout(cfg['head_dropout']),
                                    nn.Linear(256, 128), nn.ReLU())
        self.head_path = nn.Linear(128, 1)
        self.head_ddg = nn.Linear(128, 1) if cfg['beta_ddg'] > 0 else None

    def forward(self, b):
        parts = []
        if self.use_seq:
            B = b['ids_wt'].shape[0]
            ids = torch.cat([b['ids_wt'], b['ids_mut']])
            mask = torch.cat([b['mask_wt'], b['mask_mut']])
            h = self.backbone(input_ids=ids, attention_mask=mask).last_hidden_state
            hp = h[torch.arange(2 * B, device=h.device), b['tokpos'].repeat(2)]
            h_wt, h_mut = hp[:B], hp[B:]
            parts += [h_wt, h_mut, h_mut - h_wt]
            if self.use_lm:
                logp = torch.log_softmax(self.lm_head(h_wt)[:, b['aa_token_ids']], dim=-1)      # (B, 20)
                llr = logp.gather(1, b['aa_mut'][:, None]) - logp.gather(1, b['aa_wt'][:, None])
                parts += [logp, llr, F.one_hot(b['aa_wt'], 20).float(), F.one_hot(b['aa_mut'], 20).float()]
        if self.n_feat:
            parts.append(self.feat_encoder(b['x']))
        z = self.fusion(torch.cat(parts, dim=-1))
        out = {'logit': self.head_path(z).squeeze(-1)}
        if self.head_ddg is not None:
            out['ddg'] = self.head_ddg(z).squeeze(-1)
        return out

    def trainable_state(self):
        keep = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if k in keep}


def n_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ------------------------------------------------------------------------------------------- training
@torch.no_grad()
def predict(model, vset, device, batch_size=64):
    model.eval()
    logits, ddgs = [], []
    for i in range(0, len(vset), batch_size):
        idx = torch.arange(i, min(i + batch_size, len(vset)))
        out = model(vset.batch(idx, device))
        logits.append(out['logit'].float().cpu().numpy())
        if 'ddg' in out:
            ddgs.append(out['ddg'].float().cpu().numpy())
    model.train()
    logit = np.concatenate(logits) if logits else np.zeros(0)
    res = vset.df[['variant_id']].copy()
    if 'y' in vset.df:
        res['y'] = vset.df['y'].to_numpy()
    res['logit'] = logit
    res['prob'] = 1.0 / (1.0 + np.exp(-logit))
    if ddgs:
        res['ddg_pred_z'] = np.concatenate(ddgs)
    res['ddg_target'] = vset.ddg_raw
    return res


def quick_metrics(pred, stats=None):
    m = {'n': int(len(pred))}
    if 'y' in pred and pred.y.notna().all() and pred.y.nunique() == 2:
        m['auroc'] = float(roc_auc_score(pred.y, pred.prob))
        m['auprc'] = float(average_precision_score(pred.y, pred.prob))
    if 'ddg_pred_z' in pred:
        k = pred.ddg_target.notna()
        if k.sum() > 10:
            m['ddg_spearman'] = float(spearmanr(pred.ddg_pred_z[k], pred.ddg_target[k]).statistic)
            m['n_ddg'] = int(k.sum())
    return m


def lr_lambda(step, total, warmup):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))


def train_model(cfg, train_df, eval_dfs, out_dir, log=print):
    """Train one model. eval_dfs: dict name -> DataFrame (predictions are written for each)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg['seed'])
    device = cfg['device']
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg['backbone'])
    seqs = load_reference_sequences()
    feat_table, feat_cols = load_feature_table(cfg['feature_groups'])
    ddg = load_ddg_table() if cfg['beta_ddg'] > 0 else {}

    def make(df):
        return VariantSet(df, seqs, tok, cfg['window'], feat_table, feat_cols, ddg)

    tr = make(train_df)
    stats = fit_stats(tr)
    tr.standardise(stats)
    evals = {}
    for name, df in eval_dfs.items():
        evals[name] = make(df)
        evals[name].standardise(stats)

    model = HearsLM(cfg, len(feat_cols)).to(device)
    lora_params = [p for n, p in model.named_parameters() if p.requires_grad and 'lora_' in n]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and 'lora_' not in n]
    groups = [{'params': head_params, 'lr': cfg['lr_head']}]
    if lora_params:
        groups.append({'params': lora_params, 'lr': cfg['lr_lora']})
    opt = torch.optim.AdamW(groups, weight_decay=cfg['weight_decay'])
    steps_per_epoch = math.ceil(len(tr) / cfg['batch_size'])
    total = steps_per_epoch * cfg['epochs']
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, total, int(cfg['warmup_frac'] * total)))
    log(f'train n={len(tr)} (P={int((tr.y == 1).sum())}) features={len(feat_cols)} ddg_targets={int(tr.has_ddg.sum())} '
        f'trainable={n_trainable(model):,} steps={total}')

    gen = torch.Generator().manual_seed(cfg['seed'])
    history, t0 = [], time.time()
    model.train()
    for epoch in range(cfg['epochs']):
        perm = torch.randperm(len(tr), generator=gen)
        losses, lp, ld = [], [], []
        for s in range(steps_per_epoch):
            idx = perm[s * cfg['batch_size']:(s + 1) * cfg['batch_size']]
            b = tr.batch(idx, device)
            out = model(b)
            loss_p = F.binary_cross_entropy_with_logits(out['logit'], b['y'])
            loss = loss_p
            if 'ddg' in out and b['has_ddg'].any():
                m = b['has_ddg']
                loss_d = F.mse_loss(out['ddg'][m], b['ddg'][m])
                loss = loss + cfg['beta_ddg'] * loss_d
                ld.append(loss_d.item())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg['grad_clip'])
            opt.step(); sched.step()
            losses.append(loss.item()); lp.append(loss_p.item())
            if (s + 1) % 200 == 0:
                el = time.time() - t0
                log(f'  ep{epoch} step {s + 1}/{steps_per_epoch} loss={np.mean(losses[-200:]):.4f} elapsed={el / 60:.1f}min')
        rec = {'epoch': epoch, 'loss': float(np.mean(losses)), 'loss_path': float(np.mean(lp)),
               'loss_ddg': float(np.mean(ld)) if ld else None, 'time_min': (time.time() - t0) / 60}
        if 'dev' in evals:                       # only development data are looked at during training
            rec['dev'] = quick_metrics(predict(model, evals['dev'], device))
        if 'devgene' in evals:
            rec['devgene'] = quick_metrics(predict(model, evals['devgene'], device))
        history.append(rec)
        log(f'epoch {epoch}: {json.dumps(rec)}')
        json.dump(history, open(out_dir / 'train_log.json', 'w'), indent=1)

    final = {}
    for name, vs in evals.items():
        pred = predict(model, vs, device)
        if 'ddg_pred_z' in pred:
            pred['ddg_pred'] = pred['ddg_pred_z'] * stats['ddg_std'] + stats['ddg_mean']
        pred.to_csv(out_dir / f'preds_{name}.csv', index=False)
        final[name] = quick_metrics(pred)
    torch.save({'config': cfg, 'feat_cols': feat_cols,
                'stats': {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in stats.items()},
                'state': model.trainable_state()}, out_dir / 'model_trainable.pt')
    json.dump({'config': cfg, 'n_train': len(tr), 'n_features': len(feat_cols), 'feat_cols': feat_cols,
               'n_trainable': n_trainable(model), 'final': final, 'history': history,
               'runtime_min': (time.time() - t0) / 60}, open(out_dir / 'result.json', 'w'), indent=1)
    log(f'done in {(time.time() - t0) / 60:.1f} min: {json.dumps(final)}')
    del model
    if device == 'mps':
        torch.mps.empty_cache()
    return final


def load_trained(run_dir, device=None):
    """Rebuild a trained model from model_trainable.pt (LoRA + head weights) for inference."""
    ck = torch.load(Path(run_dir) / 'model_trainable.pt', map_location='cpu', weights_only=False)
    cfg = ck['config']
    if device:
        cfg['device'] = device
    model = HearsLM(cfg, len(ck['feat_cols']))
    missing, unexpected = model.load_state_dict(ck['state'], strict=False)
    assert not unexpected, unexpected
    assert all(('lora_' not in k) and k.startswith(('backbone', 'lm_head')) for k in missing), 'trainable weights missing'
    stats = {k: (np.array(v, dtype=np.float32) if isinstance(v, list) else v) for k, v in ck['stats'].items()}
    return model.to(cfg['device']).eval(), cfg, ck['feat_cols'], stats
