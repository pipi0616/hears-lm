#!/usr/bin/env python3
"""
Step 13 - Full-context amino-acid log-probabilities from the pretrained (frozen) ESM-2 150M, for every residue of
every reference protein ("wild-type marginal" scoring, Meier et al. 2021).

Why: the fine-tuned encoder of HEARS-LM sees a 65-residue window around the variant (as in the original study).
Zero-shot scoring shows that this window discards much of the information available to the language model
(masked-marginal AUROC on the development set 0.84 with full context vs 0.77 with the window). The full-context
log-probabilities of the unmodified language model are therefore provided to the model as an input group ('plm').
Wild-type marginals need one forward pass per protein window, so they can be computed for any variant
(training, test, VUS, saturation maps) at negligible cost.

Context: whole protein if <= 1022 residues; otherwise windows of 1022 residues with stride 511, and every residue
takes its log-probabilities from the window in which it lies furthest from a window edge.

Outputs
  data/annotations/esm2_wt_marginals.csv.gz : gene, ref_pos, ref_aa, lp_<A..Y> (natural-log probabilities over the
                                              20 standard amino acids, renormalised), entropy
  data/annotations/esm2_wt_marginals_summary.json
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
AA = 'ACDEFGHIKLMNPQRSTVWY'
MODEL = 'facebook/esm2_t30_150M_UR50D'
MAXLEN, STRIDE = 1022, 511


def windows(n):
    if n <= MAXLEN:
        return [(0, n)]
    starts = list(range(0, n - MAXLEN + 1, STRIDE))
    if starts[-1] != n - MAXLEN:
        starts.append(n - MAXLEN)
    return [(s, s + MAXLEN) for s in starts]


@torch.no_grad()
def main():
    from transformers import AutoTokenizer, EsmForMaskedLM
    dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = EsmForMaskedLM.from_pretrained(MODEL).to(dev).eval()
    aa_ids = torch.tensor(tok.convert_tokens_to_ids(list(AA)), device=dev)
    ref = pd.read_csv(ROOT / 'data/sources/reference_proteins.csv')
    rows, t0, n_fwd = [], time.time(), 0
    for gene, seq in zip(ref.gene, ref.sequence):
        n = len(seq)
        best_lp = np.full((n, 20), np.nan, np.float32)
        best_margin = np.full(n, -1)
        for s, e in windows(n):
            enc = tok(seq[s:e], return_tensors='pt').to(dev)
            logits = model(**enc).logits[0, 1:-1]                       # drop <cls>, <eos>
            lp = torch.log_softmax(logits[:, aa_ids].float(), dim=-1).cpu().numpy()
            n_fwd += 1
            idx = np.arange(s, e)
            margin = np.minimum(idx - s, e - 1 - idx)
            if s == 0:
                margin = np.where(idx - s < MAXLEN // 2, MAXLEN, margin)     # protein start is a true edge
            if e == n:
                margin = np.where(e - 1 - idx < MAXLEN // 2, MAXLEN, margin)
            upd = margin > best_margin[idx]
            best_lp[idx[upd]] = lp[upd]
            best_margin[idx[upd]] = margin[upd]
        ent = -(np.exp(best_lp) * best_lp).sum(1)
        df = pd.DataFrame(best_lp, columns=[f'lp_{a}' for a in AA])
        df.insert(0, 'ref_aa', list(seq))
        df.insert(0, 'ref_pos', np.arange(1, n + 1))
        df.insert(0, 'gene', gene)
        df['entropy'] = ent
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(ROOT / 'data/annotations/esm2_wt_marginals.csv.gz', index=False, float_format='%.5f')

    # validation against the masked-marginal scores of the labelled variants
    from sklearn.metrics import roc_auc_score
    v = pd.read_csv(ROOT / 'data/variants_v8.csv', usecols=['variant_id', 'gene', 'ref_pos', 'ref1', 'alt1', 'partition', 'y'])
    m = pd.read_csv(ROOT / 'data/annotations/esm2_zeroshot_v8.csv', usecols=['variant_id', 'esm2_150m_llr'])
    t = v.merge(out, on=['gene', 'ref_pos'], how='left').merge(m, on='variant_id', how='left')
    assert (t.ref_aa == t.ref1).all(), 'reference residue mismatch'
    t['wt_llr'] = [r[f'lp_{a}'] - r[f'lp_{w}'] for r, a, w in zip(t.to_dict('records'), t.alt1, t.ref1)]
    summ = {'model': MODEL, 'n_genes': int(ref.shape[0]), 'n_residues': int(len(out)), 'n_forward': n_fwd,
            'seconds': round(time.time() - t0, 1),
            'spearman_wt_vs_masked_llr': float(t[['wt_llr', 'esm2_150m_llr']].corr(method='spearman').iloc[0, 1]),
            'auroc': {}}
    for p in ['fit', 'dev', 'test_time', 'test_gene', 'test_db']:
        d = t[t.partition == p]
        summ['auroc'][p] = {'wt_marginal': float(roc_auc_score(d.y, -d.wt_llr)),
                            'masked_marginal': float(roc_auc_score(d.y, -d.esm2_150m_llr))}
    json.dump(summ, open(ROOT / 'data/annotations/esm2_wt_marginals_summary.json', 'w'), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
