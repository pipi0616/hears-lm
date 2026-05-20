#!/usr/bin/env python3
"""
v4 ESM-2 t30 (150M params, hidden=640) + LoRA r=16, same v1 architecture, HHL-specific.

Strategy: keep HHL-specialized training; just upgrade PLM capacity.
  - ESM-2 t12 (35M, hidden=480) → ESM-2 t30 (150M, hidden=640)
  - LoRA r=8 → r=16 (more LoRA capacity to leverage larger PLM)
  - Same 100 features, same 36K HHL training, same loss
  - Batch size 4 → 2 (memory headroom)
  - Expected: AUC 0.821 → 0.84+ on val_db
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '2')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import time, json, warnings, re
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')
torch.set_num_threads(2)

ROOT = Path('/Users/pipi/Projects/QAFI_Paper')
DATA = ROOT / 'hearing_v2/data'
SEQ_FILE = ROOT / 'hearing/data/features/gene_sequences.csv'
DDG_FILE = ROOT / 'hearing/data/external/tollefson2023_ddg_parsed.csv'
CKPT_DIR = ROOT / 'hearing_v2/checkpoints'
LOG_DIR  = ROOT / 'hearing_v2/logs'

CONFIG = {
    'model_name': 'facebook/esm2_t30_150M_UR50D',  # ← UPGRADED
    'lora_r': 16, 'lora_alpha': 32, 'lora_dropout': 0.1,
    'lora_targets': ['query', 'value'],
    'batch_size': 2,           # smaller batch for memory
    'seq_window': 32,
    'lr': 5e-5,                 # slightly lower for bigger model
    'epochs': 3,
    'device': 'mps' if torch.backends.mps.is_available() else 'cpu',
    'pos_weight': 5.93,
    'beta_ddg': 0.3,
    'focal_gamma': 2.0,
    'eval_subset': 800,
    'log_every': 200,
    'save_every_epoch': True,
}
print(f'Config: {CONFIG}')


class HHLVariantDataset(Dataset):
    def __init__(self, df, sequences_df, ddg_df, tokenizer, config, feat_cols):
        self.df = df.reset_index(drop=True)
        self.seq_map = dict(zip(sequences_df.gene, sequences_df.sequence))
        self.ddg_map = {(r.gene, r.protein_change): r.ddg_fold for _, r in ddg_df.iterrows()}
        self.tokenizer = tokenizer
        self.window = config['seq_window']
        self.feat_cols = feat_cols
        ok = self.df.gene.isin(self.seq_map.keys())
        self.df = self.df[ok].reset_index(drop=True)

    def __len__(self): return len(self.df)

    @staticmethod
    def parse_pc(pc):
        AA = {'Ala':'A','Arg':'R','Asn':'N','Asp':'D','Cys':'C','Gln':'Q','Glu':'E','Gly':'G',
              'His':'H','Ile':'I','Leu':'L','Lys':'K','Met':'M','Phe':'F','Pro':'P','Ser':'S',
              'Thr':'T','Trp':'W','Tyr':'Y','Val':'V','Ter':'*'}
        m = re.match(r'^([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})$', str(pc))
        if m:
            return AA.get(m.group(1)), int(m.group(2)), AA.get(m.group(3))
        return None, None, None

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = self.seq_map[row.gene]
        orig, pos, mut = self.parse_pc(row.protein_change)
        if orig is None or pos is None or pos > len(seq):
            pos = min(int(row.aa_pos) if pd.notna(row.aa_pos) else 1, len(seq))
            orig = mut = 'X'
        seq_mut = seq[:pos-1] + (mut or 'X') + seq[pos:]
        start = max(0, pos - self.window - 1)
        end = min(len(seq), pos + self.window)
        seq_wt_win = seq[start:end]
        seq_mut_win = seq_mut[start:end]
        rel_pos = pos - 1 - start
        tok_wt = self.tokenizer(seq_wt_win, return_tensors='pt', padding='max_length',
                                max_length=2*self.window+3, truncation=True)
        tok_mut = self.tokenizer(seq_mut_win, return_tensors='pt', padding='max_length',
                                 max_length=2*self.window+3, truncation=True)
        y = float(row.y) if pd.notna(row.y) else 0.0
        ddg = self.ddg_map.get((row.gene, row.protein_change), np.nan)
        has_ddg = 0 if np.isnan(ddg) else 1
        ddg_val = 0.0 if np.isnan(ddg) else float(ddg)
        aux = torch.tensor([float(row[c]) if c in row.index and pd.notna(row[c]) else 0.0
                            for c in self.feat_cols], dtype=torch.float32)
        return {
            'input_ids_wt': tok_wt['input_ids'].squeeze(0),
            'attention_mask_wt': tok_wt['attention_mask'].squeeze(0),
            'input_ids_mut': tok_mut['input_ids'].squeeze(0),
            'attention_mask_mut': tok_mut['attention_mask'].squeeze(0),
            'rel_pos': rel_pos + 1,
            'aux_features': aux,
            'y': torch.tensor(y, dtype=torch.float32),
            'ddg': torch.tensor(ddg_val, dtype=torch.float32),
            'has_ddg': torch.tensor(has_ddg, dtype=torch.float32),
        }


class HHLVariantModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        from transformers import AutoModel
        from peft import get_peft_model, LoraConfig
        self.backbone = AutoModel.from_pretrained(config['model_name'])
        peft_config = LoraConfig(
            r=config['lora_r'], lora_alpha=config['lora_alpha'],
            target_modules=config['lora_targets'],
            lora_dropout=config['lora_dropout'], bias='none')
        self.backbone = get_peft_model(self.backbone, peft_config)
        self.backbone.print_trainable_parameters()
        H = self.backbone.config.hidden_size  # 640 for t30
        n_aux = config.get('n_aux', 100)
        self.aux_encoder = nn.Sequential(
            nn.Linear(n_aux, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64))
        head_in_dim = 3*H + 64
        self.fusion = nn.Sequential(
            nn.Linear(head_in_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU())
        self.head_A = nn.Linear(128, 1)
        self.head_D = nn.Linear(128, 1)

    def forward(self, batch):
        out_wt = self.backbone(input_ids=batch['input_ids_wt'],
                               attention_mask=batch['attention_mask_wt'])
        out_mut = self.backbone(input_ids=batch['input_ids_mut'],
                                attention_mask=batch['attention_mask_mut'])
        idx = batch['rel_pos'].unsqueeze(-1).unsqueeze(-1).expand(
            -1, 1, out_wt.last_hidden_state.size(-1))
        emb_wt = out_wt.last_hidden_state.gather(1, idx).squeeze(1)
        emb_mut = out_mut.last_hidden_state.gather(1, idx).squeeze(1)
        emb_diff = emb_wt - emb_mut
        aux_emb = self.aux_encoder(batch['aux_features'])
        fused = torch.cat([emb_wt, emb_mut, emb_diff, aux_emb], dim=-1)
        shared = self.fusion(fused)
        return {'logit_A': self.head_A(shared).squeeze(-1),
                'pred_D': self.head_D(shared).squeeze(-1)}


def focal_bce(logit, target, pos_weight, gamma):
    p = torch.sigmoid(logit)
    ce = F.binary_cross_entropy_with_logits(
        logit, target, pos_weight=torch.tensor(pos_weight, device=logit.device), reduction='none')
    pt = torch.where(target == 1, p, 1 - p)
    return ((1 - pt) ** gamma * ce).mean()


def masked_mse(pred, target, mask):
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    return F.mse_loss(pred[mask.bool()], target[mask.bool()])


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    probs, ys, ddp, ddt, ddm = [], [], [], [], []
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        out = model(batch)
        probs.append(torch.sigmoid(out['logit_A']).cpu().numpy())
        ys.append(batch['y'].cpu().numpy())
        ddp.append(out['pred_D'].cpu().numpy())
        ddt.append(batch['ddg'].cpu().numpy())
        ddm.append(batch['has_ddg'].cpu().numpy())
    probs = np.concatenate(probs); ys = np.concatenate(ys)
    auc = float(roc_auc_score(ys, probs)) if len(set(ys)) > 1 else float('nan')
    ap = float(average_precision_score(ys, probs)) if len(set(ys)) > 1 else float('nan')
    dp = np.concatenate(ddp); dt = np.concatenate(ddt); dm = np.concatenate(ddm).astype(bool)
    rho = float(spearmanr(dp[dm], dt[dm]).statistic) if dm.sum() > 10 else float('nan')
    model.train()
    return {'auc': auc, 'ap': ap, 'ddg_spearman': rho, 'n': len(ys), 'n_ddg': int(dm.sum())}


def main():
    print('\n[1] Load data...')
    train = pd.read_csv(DATA/'splits/train_with_features.csv', low_memory=False)
    val_time = pd.read_csv(DATA/'splits/val_time_with_features.csv', low_memory=False)
    val_gene = pd.read_csv(DATA/'splits/val_gene_with_features.csv', low_memory=False)
    seq = pd.read_csv(SEQ_FILE)
    ddg = pd.read_csv(DDG_FILE)

    META = {'gene','protein_change','aa_pos','y','clinical_significance','review_status',
            'VariationID','variation_id','year','chromosome','position','ref','alt',
            'hgvs','transcript','impact','consequence','assertion','submitter','condition',
            'source','split_origin','set','category','classification','last_evaluated','snapshot'}

    # EXCLUDE other tools' pathogenicity predictor scores (no stacking — HHL-specific PLM only)
    # Keep: HHL-niche (cell-type/HPO/CGD/IMPC/constraint), conservation, structural ΔΔG (Tollefson FoldX = biophysics, not pathogenicity),
    #       AlphaFold pLDDT, gnomAD AF, n_sources.
    # Exclude: baseline_* (dbNSFP predictors), am_*, esm_llr (zero-shot prediction), revel/cadd from val_db.
    PREDICTOR_EXCLUDE = set()
    for c in train.columns:
        if c.startswith('baseline_'): PREDICTOR_EXCLUDE.add(c)
        if c.startswith('am_'): PREDICTOR_EXCLUDE.add(c)
        if c in ('esm_llr','has_esm_llr','revel','cadd_phred'): PREDICTOR_EXCLUDE.add(c)
    print(f'  Excluded {len(PREDICTOR_EXCLUDE)} predictor-score features (no stacking)')

    FEAT_COLS = [c for c in train.columns
                 if c not in META
                 and c not in PREDICTOR_EXCLUDE
                 and pd.api.types.is_numeric_dtype(train[c])
                 and not train[c].isna().all()]
    FEAT_COLS = [c for c in FEAT_COLS if c in val_time.columns and c in val_gene.columns]
    CONFIG['n_aux'] = len(FEAT_COLS)
    print(f'  train={len(train)} val_time={len(val_time)} val_gene={len(val_gene)}  features={len(FEAT_COLS)}')

    if len(val_time) > CONFIG['eval_subset']:
        val_time = val_time.sample(n=CONFIG['eval_subset'], random_state=42).reset_index(drop=True)
    if len(val_gene) > CONFIG['eval_subset']:
        val_gene = val_gene.sample(n=CONFIG['eval_subset'], random_state=42).reset_index(drop=True)

    print(f'\n[2] Tokenizer + ESM-2 t30 (150M)...')
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    model = HHLVariantModel(CONFIG).to(CONFIG['device'])

    print(f'\n[3] Datasets...')
    ds_train = HHLVariantDataset(train, seq, ddg, tok, CONFIG, FEAT_COLS)
    ds_vt = HHLVariantDataset(val_time, seq, ddg, tok, CONFIG, FEAT_COLS)
    ds_vg = HHLVariantDataset(val_gene, seq, ddg, tok, CONFIG, FEAT_COLS)
    L_train = DataLoader(ds_train, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)
    L_vt = DataLoader(ds_vt, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)
    L_vg = DataLoader(ds_vg, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                   lr=CONFIG['lr'], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs']*len(L_train))

    metrics = []
    print(f'\n[4] Training {CONFIG["epochs"]} epochs ({len(L_train)} steps/epoch)...')
    t0 = time.time()
    for epoch in range(CONFIG['epochs']):
        ep_losses = []; ep_t0 = time.time()
        for step, batch in enumerate(L_train):
            batch = {k: v.to(CONFIG['device']) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            out = model(batch)
            lA = focal_bce(out['logit_A'], batch['y'], CONFIG['pos_weight'], CONFIG['focal_gamma'])
            lD = masked_mse(out['pred_D'], batch['ddg'], batch['has_ddg']) * CONFIG['beta_ddg']
            loss = lA + lD
            loss.backward()
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            ep_losses.append(loss.item())
            if (step+1) % CONFIG['log_every'] == 0:
                el = time.time() - ep_t0
                spd = (step+1)/el
                eta = (len(L_train)-step-1)/spd
                print(f'  ep{epoch} step {step+1}/{len(L_train)}  loss={np.mean(ep_losses[-200:]):.4f}  '
                      f'spd={spd:.2f}b/s eta={eta/60:.0f}min', flush=True)

        print(f'\n  Epoch {epoch} done. avg_loss={np.mean(ep_losses):.4f}  time={time.time()-ep_t0:.0f}s', flush=True)
        print('  Eval val_time + val_gene...', flush=True)
        m_vt = evaluate(model, L_vt, CONFIG['device'])
        m_vg = evaluate(model, L_vg, CONFIG['device'])
        print(f'  val_time  AUC={m_vt["auc"]:.3f}  AP={m_vt["ap"]:.3f}  ddg_rho={m_vt["ddg_spearman"]:.3f}')
        print(f'  val_gene  AUC={m_vg["auc"]:.3f}  AP={m_vg["ap"]:.3f}  ddg_rho={m_vg["ddg_spearman"]:.3f}')
        metrics.append({'epoch': epoch, 'train_loss': float(np.mean(ep_losses)),
                        'val_time': m_vt, 'val_gene': m_vg, 'time_s': time.time()-ep_t0})
        json.dump(metrics, open(LOG_DIR/'train_local_t30_metrics.json','w'), indent=2)
        ckpt = CKPT_DIR / f'local_t30_ep{epoch}.pt'
        torch.save({'state_dict': model.state_dict(), 'config': CONFIG, 'feat_cols': FEAT_COLS,
                    'epoch': epoch, 'metrics': m_vt}, ckpt)
        print(f'  ckpt → {ckpt.name}')

    print(f'\n[DONE] total: {(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
