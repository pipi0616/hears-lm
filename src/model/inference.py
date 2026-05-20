#!/usr/bin/env python3
"""Eval t30 ckpt on val_db + val_gene + val_func_mave."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import torch, warnings, time, sys
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')
torch.set_num_threads(2)

sys.path.insert(0, str(Path(__file__).parent))
from train_local_t30 import HHLVariantModel, HHLVariantDataset

ROOT = Path('/Users/pipi/Projects/QAFI_Paper')
DATA = ROOT / 'hearing_v2/data'
CKPT = ROOT / 'hearing_v2/checkpoints/local_t30_ep2.pt'
SEQ_FILE = ROOT / 'hearing/data/features/gene_sequences.csv'
DDG_FILE = ROOT / 'hearing/data/external/tollefson2023_ddg_parsed.csv'
EVAL_DIR = ROOT / 'hearing_v2/eval_results'

ck = torch.load(CKPT, map_location='cpu', weights_only=False)
CONFIG = ck['config']
FEAT = ck['feat_cols']
device = CONFIG.get('device', 'mps')
print(f'Loaded t30, n_aux={len(FEAT)}, device={device}')
CONFIG['n_aux'] = len(FEAT)

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(CONFIG['model_name'])
model = HHLVariantModel(CONFIG).to(device)
model.load_state_dict(ck['state_dict'])
model.eval()

seq = pd.read_csv(SEQ_FILE)
ddg = pd.read_csv(DDG_FILE)


@torch.no_grad()
def predict(df, name):
    for c in FEAT:
        if c not in df.columns: df[c] = 0.0
    ds = HHLVariantDataset(df, seq, ddg, tok, CONFIG, FEAT)
    L = DataLoader(ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)
    print(f'  {name}: {len(ds)}/{len(df)}')
    probs, ddgs = [], []
    t0 = time.time()
    for batch in L:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}
        out = model(batch)
        probs.append(torch.sigmoid(out['logit_A']).cpu().numpy())
        ddgs.append(out['pred_D'].cpu().numpy())
    probs = np.concatenate(probs); ddgs = np.concatenate(ddgs)
    print(f'    inference {time.time()-t0:.0f}s')
    return ds.df.assign(pred_prob=probs, pred_ddg=ddgs)


# val_db (the key benchmark vs REVEL)
print('\n=== val_db ===')
db = pd.read_csv(DATA/'splits/val_db_with_features.csv', low_memory=False)
print(f'  Loaded {len(db)}, pos_rate={db.y.mean():.3f}')
db_pred = predict(db, 'val_db')
y = db_pred.y
auc = roc_auc_score(y, db_pred.pred_prob)
ap = average_precision_score(y, db_pred.pred_prob)
print(f'\n  t30 val_db AUC: {auc:.3f}  AP: {ap:.3f}  n={len(db_pred)}')
print(f'  REFERENCE:')
print(f'    REVEL:           0.872 (Tier A 0.866)')
print(f'    Ours_v1 (t12):   0.821')
print(f'    AlphaMissense:   0.814 (full file)')
print(f'    CADD:            0.808')

# Save
OUT = EVAL_DIR / 'val_db_predictions_t30_ep2.csv'
db_pred[['gene','protein_change','y','pred_prob','pred_ddg']].to_csv(OUT, index=False)
print(f'\n✅ Saved → {OUT}')

# val_gene
print('\n=== val_gene ===')
vg = pd.read_csv(DATA/'splits/val_gene_with_features.csv', low_memory=False)
vg_pred = predict(vg, 'val_gene')
auc = roc_auc_score(vg_pred.y, vg_pred.pred_prob)
print(f'\n  t30 val_gene AUC: {auc:.3f}  (t12 v1: 0.929; ep2 in-train sub: 0.947)')
vg_pred[['gene','protein_change','y','pred_prob','pred_ddg']].to_csv(EVAL_DIR/'val_gene_predictions_t30_ep2.csv', index=False)

# val_func_mave
print('\n=== val_func_mave ===')
mv = pd.read_csv(DATA/'splits/val_func_mave_with_features.csv', low_memory=False)
mv['y'] = mv['func_label'].fillna(0)
if 'aa_pos' not in mv.columns: mv['aa_pos'] = 1
mv_pred = predict(mv, 'val_func_mave')
m = mv_pred.func_label.notna()
auc = roc_auc_score(mv_pred.loc[m,'func_label'], mv_pred.loc[m,'pred_prob'])
print(f'\n  t30 val_func_mave binary AUC: {auc:.3f}  (t12 v1: 0.599)')
m2 = mv_pred.mean_score_z.notna()
rho = spearmanr(mv_pred.loc[m2,'pred_prob'], mv_pred.loc[m2,'mean_score_z']).statistic
print(f'  t30 vs MAVE z continuous Spearman: {rho:+.3f}  (t12: -0.099)')

# Per-gene Spearman
print(f'  Per-gene Spearman:')
rho_per = []
for g in mv_pred.gene.unique():
    sub = mv_pred[(mv_pred.gene==g) & mv_pred.mean_score_z.notna()]
    if len(sub) > 20:
        r = spearmanr(sub.pred_prob, sub.mean_score_z).statistic
        rho_per.append(r)
        print(f'    {g:10}  n={len(sub):5d}  ρ={r:+.3f}')
print(f'\n  Median per-gene ρ: {np.median(rho_per):+.3f}')
mv_pred[['gene','protein_change','func_label','mean_score_z','pred_prob','pred_ddg']].to_csv(EVAL_DIR/'val_func_mave_predictions_t30_ep2.csv', index=False)
