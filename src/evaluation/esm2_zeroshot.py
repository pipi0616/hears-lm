#!/usr/bin/env python3
"""
ESM-2 zero-shot pseudo-likelihood baseline (no training, no LoRA).

Score = log P(WT aa | masked context) - log P(mut aa | masked context)
Higher score → mutation is unlikely under language model → more pathogenic-like.

This is the canonical "ESM-1v / ESM-2 zero-shot" baseline (Meir & Rao 2021).
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import re
from pathlib import Path
import time
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
torch.set_num_threads(2)

ROOT = Path('/Users/pipi/Projects/QAFI_Paper')
DATA = ROOT / 'hearing_v2/data'
SEQ_FILE = ROOT / 'hearing/data/features/gene_sequences.csv'
OUT = ROOT / 'hearing_v2/eval_results'
OUT.mkdir(parents=True, exist_ok=True)

MODEL_NAME = 'facebook/esm2_t12_35M_UR50D'  # same backbone we fine-tuned
WINDOW = 511  # max ESM context; 1024 - special tokens

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'Device: {device}')

print(f'Loading ESM-2...')
from transformers import AutoModelForMaskedLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(device).eval()
MASK_ID = tok.mask_token_id

seq_df = pd.read_csv(SEQ_FILE)
SEQ = dict(zip(seq_df.gene, seq_df.sequence))

AA3 = {'Ala':'A','Arg':'R','Asn':'N','Asp':'D','Cys':'C','Gln':'Q','Glu':'E','Gly':'G',
       'His':'H','Ile':'I','Leu':'L','Lys':'K','Met':'M','Phe':'F','Pro':'P','Ser':'S',
       'Thr':'T','Trp':'W','Tyr':'Y','Val':'V','Ter':'*'}

def parse_pc(pc):
    m = re.match(r'^([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})$', str(pc))
    if not m:
        return None, None, None
    return AA3.get(m.group(1)), int(m.group(2)), AA3.get(m.group(3))


@torch.no_grad()
def score_variants(df, name, batch_size=8):
    """Compute ESM-2 zero-shot LLR for each variant. Returns df with esm_llr column."""
    scores = []
    n_skip = 0
    t0 = time.time()
    # Group by gene → reuse sequence per gene
    rows = list(df[['gene','protein_change']].itertuples(index=True))
    print(f'  Scoring {len(rows)} variants ({df.gene.nunique()} genes)...')

    # Process one variant at a time (different sequences per gene; batching across genes is messy)
    for i, r in enumerate(rows):
        if i % 500 == 0 and i > 0:
            el = time.time()-t0
            spd = i/el
            eta = (len(rows)-i)/spd
            print(f'    {i}/{len(rows)}  speed={spd:.1f}/s  eta={eta/60:.1f}min', flush=True)

        gene = r.gene
        pc = r.protein_change
        if gene not in SEQ:
            scores.append((r.Index, np.nan)); n_skip += 1; continue
        seq = SEQ[gene]
        orig, pos, mut = parse_pc(pc)
        if orig is None or pos is None or pos > len(seq) or mut is None:
            scores.append((r.Index, np.nan)); n_skip += 1; continue
        if seq[pos-1] != orig:
            # sequence mismatch — skip
            scores.append((r.Index, np.nan)); n_skip += 1; continue

        # Build masked sequence window
        # ESM-2 max length ~1024 tokens — for HHL proteins many are <1024. Window to be safe.
        if len(seq) > WINDOW:
            half = WINDOW // 2
            start = max(0, pos - 1 - half)
            end = min(len(seq), start + WINDOW)
            start = max(0, end - WINDOW)
            local_pos = pos - 1 - start  # 0-indexed in window
            seq_window = seq[start:end]
        else:
            seq_window = seq
            local_pos = pos - 1

        # Tokenize, mask position
        enc = tok(seq_window, return_tensors='pt').to(device)
        # Token positions: [CLS] + seq + [SEP], so target token at index local_pos+1
        target_tok_idx = local_pos + 1
        input_ids = enc['input_ids'].clone()
        input_ids[0, target_tok_idx] = MASK_ID

        out = model(input_ids=input_ids, attention_mask=enc['attention_mask'])
        logits = out.logits[0, target_tok_idx, :]   # (vocab,)
        log_probs = F.log_softmax(logits, dim=-1)
        wt_tok = tok.get_vocab().get(orig)
        mt_tok = tok.get_vocab().get(mut)
        if wt_tok is None or mt_tok is None:
            scores.append((r.Index, np.nan)); n_skip += 1; continue
        llr = (log_probs[wt_tok] - log_probs[mt_tok]).item()
        scores.append((r.Index, llr))

    el = time.time()-t0
    print(f'  Done: {len(rows)-n_skip} scored / {n_skip} skipped in {el/60:.1f}min')
    res = pd.Series({i: s for i, s in scores})
    df = df.copy()
    df['esm_llr'] = res
    df.to_csv(OUT/f'esm_zeroshot_{name}.csv', index=False)
    return df


# === val_db ===
print(f'\n=== val_db ===')
db = pd.read_csv(DATA/'splits/val_db_with_features.csv', low_memory=False)
db = score_variants(db, 'val_db')
m = db.esm_llr.notna()
if m.sum() > 50:
    auc = roc_auc_score(db.loc[m,'y'], db.loc[m,'esm_llr'])
    if auc < 0.5: auc = 1 - auc
    print(f'\nval_db  ESM-2 zero-shot AUC = {auc:.3f}  (n={m.sum()})')
    # Compare to our fine-tuned + baselines
    print(f'  Compare:')
    print(f'    Our fine-tuned model (LoRA + aux): AUC=0.821  AP=0.923')
    print(f'    REVEL: AUC=0.872')
    print(f'    CADD: AUC=0.808')


# === val_func_mave ===
print(f'\n=== val_func_mave ===')
mv = pd.read_csv(DATA/'splits/val_func_mave_with_features.csv', low_memory=False)
mv = score_variants(mv, 'val_func_mave')

m_func = mv.esm_llr.notna() & mv.func_label.notna()
m_z = mv.esm_llr.notna() & mv.mean_score_z.notna()
if m_func.sum() > 50:
    auc = roc_auc_score(mv.loc[m_func,'func_label'], mv.loc[m_func,'esm_llr'])
    if auc < 0.5: auc = 1 - auc
    print(f'\nval_func_mave  ESM-2 zero-shot:')
    print(f'  vs func_label AUC = {auc:.3f}  (n={m_func.sum()})')
if m_z.sum() > 100:
    rho = spearmanr(mv.loc[m_z,'esm_llr'], mv.loc[m_z,'mean_score_z']).statistic
    print(f'  vs mean_score_z ρ = {rho:+.3f}  (n={m_z.sum()})  expected: negative (high LLR=pathogenic-like, low z=deleterious)')

    print(f'\n  Per-gene Spearman:')
    rows = []
    for g in mv.gene.unique():
        s = mv[(mv.gene==g) & mv.esm_llr.notna() & mv.mean_score_z.notna()]
        if len(s) > 20:
            r = spearmanr(s.esm_llr, s.mean_score_z).statistic
            rows.append((g, len(s), r))
    for g, n, r in sorted(rows, key=lambda x: x[2]):
        print(f'    {g:10}  n={n:5d}  ρ={r:+.3f}')
    if rows:
        med = np.median([r for _,_,r in rows])
        print(f'  median ρ = {med:+.3f}  (compare to fine-tuned: -0.095)')
