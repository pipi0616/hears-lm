#!/usr/bin/env python3
"""Generate model predictions for val_time + val_gene (full set) from ckpts."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import torch, warnings, time, sys
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
warnings.filterwarnings('ignore')
torch.set_num_threads(2)

ROOT = Path('/Users/pipi/Projects/QAFI_Paper')
DATA = ROOT / 'hearing_v2/data'
SEQ_FILE = ROOT / 'hearing/data/features/gene_sequences.csv'
DDG_FILE = ROOT / 'hearing/data/external/tollefson2023_ddg_parsed.csv'
EVAL = ROOT / 'hearing_v2/eval_results'

sys.path.insert(0, str(Path(__file__).parent))

def predict_from_ckpt(ckpt_path, model_class_name, version_tag):
    print(f'\n=== {ckpt_path.name} ===')
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    CONFIG = ck['config']
    device = CONFIG.get('device', 'mps')

    if model_class_name == 'v1':
        from train_local_full import HHLVariantModel, HHLVariantDataset
        FEAT = ck['feat_cols']
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(CONFIG['model_name'])
        model = HHLVariantModel(CONFIG).to(device)
    elif model_class_name == 'v2':
        from train_local_v2 import HHLVariantModelV2, HHLVariantDataset
        VAR_FEATS = ck['var_feats']; GENE_FEATS = ck['gene_feats']
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(CONFIG['model_name'])
        model = HHLVariantModelV2(CONFIG, len(VAR_FEATS), len(GENE_FEATS)).to(device)
    elif model_class_name == 'v3':
        from train_local_v3 import HHLVariantModelV3, HHLVariantDataset
        VAR_FEATS = ck['var_feats']; GENE_FEATS = ck['gene_feats']
        GMAP = ck['gene_id_map']
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(CONFIG['model_name'])
        model = HHLVariantModelV3(CONFIG, len(VAR_FEATS), len(GENE_FEATS), n_genes=len(GMAP)).to(device)
    else:
        raise ValueError(model_class_name)

    model.load_state_dict(ck['state_dict'])
    model.eval()

    seq = pd.read_csv(SEQ_FILE)
    ddg = pd.read_csv(DDG_FILE)
    seq_map = dict(zip(seq.gene, seq.sequence))

    for sname in ['val_time', 'val_gene']:
        df = pd.read_csv(DATA/f'splits/{sname}_with_features.csv', low_memory=False)
        if model_class_name == 'v1':
            for c in FEAT:
                if c not in df.columns: df[c] = 0.0
            ds = HHLVariantDataset(df, seq, ddg, tok, CONFIG, FEAT)
        elif model_class_name == 'v2':
            for c in VAR_FEATS + GENE_FEATS:
                if c not in df.columns: df[c] = 0.0
            ds = HHLVariantDataset(df, seq_map, ddg, tok, CONFIG, VAR_FEATS, GENE_FEATS)
        elif model_class_name == 'v3':
            for c in VAR_FEATS + GENE_FEATS:
                if c not in df.columns: df[c] = 0.0
            ds = HHLVariantDataset(df, seq_map, ddg, tok, CONFIG, VAR_FEATS, GENE_FEATS, gene_id_map=GMAP)
        L = DataLoader(ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)
        probs, ddgs = [], []
        t0 = time.time()
        with torch.no_grad():
            for batch in L:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}
                if model_class_name == 'v1':
                    out = model(batch)
                elif model_class_name == 'v2':
                    out = model(batch, gene_dropout=False)
                else:
                    out = model(batch, gene_dropout=False, grl_lambda=0.0)
                probs.append(torch.sigmoid(out['logit_A']).cpu().numpy())
                ddgs.append(out['pred_D'].cpu().numpy())
        probs = np.concatenate(probs); ddgs = np.concatenate(ddgs)
        pred_df = ds.df.assign(pred_prob=probs, pred_ddg=ddgs)
        pred_df[['gene','protein_change','y','pred_prob','pred_ddg']].to_csv(
            EVAL/f'{sname}_predictions_{version_tag}_ep2.csv', index=False)
        print(f'  {sname}  inference {time.time()-t0:.0f}s  n={len(pred_df)}  saved')


for tag, model_cls in [('','v1'), ('_v2','v2'), ('_v3','v3')]:
    ckpt_name = f'local{tag}_full_ep2.pt' if tag == '' else f'local{tag[1:]}_ep2.pt'
    if tag == '':
        ckpt = ROOT / f'hearing_v2/checkpoints/local_full_ep2.pt'
        vtag = 'ep2'
    else:
        ckpt = ROOT / f'hearing_v2/checkpoints/local{tag}_ep2.pt'
        vtag = f'{tag[1:]}_ep2'
    if ckpt.exists():
        predict_from_ckpt(ckpt, model_cls, vtag)
    else:
        print(f'  Missing: {ckpt}')

print('\n✅ Done')
