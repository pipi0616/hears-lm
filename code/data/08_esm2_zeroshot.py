#!/usr/bin/env python3
"""
Step 08 - Zero-shot ESM-2 scores (masked-marginal log-likelihood ratio) for every labelled variant,
computed on the REFERENCE protein of the gene (the sequence the fine-tuned model sees).

Inputs
  data/variants_v8.csv                  variant_id, gene, ref_pos, ref1, alt1 (ref_pos = 1-based position on the
                                        reference protein), partition, y
  data/sources/reference_proteins.csv   gene, sequence (reference protein)
  Hugging Face cache                    facebook/esm2_t30_150M_UR50D (EsmForMaskedLM, float32)

Output
  data/annotations/esm2_zeroshot_v8.csv           one row per variant_id (same order as variants_v8.csv):
      variant_id, gene, ref_pos, ref1, alt1,
      esm2_150m_llr        LLR with the long context
      esm2_150m_llr_w32    LLR with the short context used by the fine-tuned model
      ctx_start, ctx_end   1-based inclusive residue range of the long context on the reference protein
      w32_start, w32_end   1-based inclusive residue range of the short context
  data/annotations/esm2_zeroshot_v8_summary.json  counts, runtime, AUROC per partition, context agreement, self-check

Rules
  * Score: the variant position is replaced by <mask>; LLR = log p(alt) - log p(ref) at that position
    (log-softmax over the model vocabulary; the difference does not depend on the normalisation set).
    More negative = less compatible with the sequence context = more damaging.
  * Long context (`esm2_150m_llr`): the full reference protein if it has <= 1022 residues; otherwise a window of
    exactly 1022 residues centred on the variant (511 residues before it, 510 after) and shifted, not shortened,
    where it would run past a terminus.
  * Short context (`esm2_150m_llr_w32`): residues [ref_pos-32, ref_pos+32] (65 residues), truncated at the protein
    termini - the same rule as the fine-tuned model's input window (seq[max(0, pos-33):min(L, pos+32)]).
  * Every context is tokenised as <cls> residues <eos> (as in ESM-2 training), also when it is a window.
  * One forward pass per unique (gene, position, context); the 33-token log-probability vector of the masked
    position is cached and reused for all substitutions at that position.
  * The reference residue of the reference protein must equal ref1 for every variant (asserted; failures listed).
  * No imputation: a variant that cannot be scored would keep NaN (none is expected).
  * Deterministic: eval mode, fixed batch composition (contexts sorted by length, gene, position).
    Floating-point noise between devices / batch compositions is quantified by a built-in self-check that
    recomputes a fixed sample of positions one by one on the CPU.

Run:  python code/data/08_esm2_zeroshot.py            (device mps, falls back to cpu if unavailable)
"""
import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, EsmForMaskedLM

ROOT = Path(__file__).resolve().parents[2]                       # the release folder
VARIANTS = ROOT / 'data/variants_v8.csv'
REFERENCE = ROOT / 'data/sources/reference_proteins.csv'
OUT_DIR = ROOT / 'data/annotations'
OUT = OUT_DIR / 'esm2_zeroshot_v8.csv'
OUT_SUMMARY = OUT_DIR / 'esm2_zeroshot_v8_summary.json'

MODEL_NAME = 'facebook/esm2_t30_150M_UR50D'
MAX_CTX = 1022          # residues; + <cls> + <eos> = 1024 tokens
W32 = 32                # residues on each side of the variant in the short context
AA20 = 'ACDEFGHIKLMNPQRSTVWY'
SELFCHECK_TOL = 0.02    # max |delta log p| allowed between batched device run and single-sequence CPU run


def long_context(length, pos):
    """0-based [start, end) of the long context for 1-based position `pos` in a protein of `length`."""
    if length <= MAX_CTX:
        return 0, length
    start = pos - 1 - MAX_CTX // 2
    start = max(0, min(start, length - MAX_CTX))
    return start, start + MAX_CTX


def short_context(length, pos):
    """0-based [start, end) of the +/-32 residue context (truncated at the termini)."""
    return max(0, pos - 1 - W32), min(length, pos + W32)


def build_tasks(variants, refseq):
    """One task per unique (context name, gene, position): (ctx, gene, pos, start0, end0)."""
    tasks = []
    for gene, pos in sorted(set(zip(variants.gene, variants.ref_pos))):
        length = len(refseq[gene])
        s, e = long_context(length, pos)
        tasks.append(('full', gene, int(pos), s, e))
        s, e = short_context(length, pos)
        tasks.append(('w32', gene, int(pos), s, e))
    for ctx, gene, pos, s, e in tasks:
        assert 0 <= s <= pos - 1 < e <= len(refseq[gene]), (ctx, gene, pos, s, e)
        assert e - s <= MAX_CTX
    return tasks


def make_batches(tasks, max_tokens, max_batch):
    """Deterministic batches of similar length: longest first, at most `max_tokens` padded tokens per batch."""
    order = sorted(tasks, key=lambda t: (-(t[4] - t[3]), t[1], t[2], t[0]))
    batches, cur = [], []
    for t in order:
        width = (cur[0][4] - cur[0][3] + 2) if cur else (t[4] - t[3] + 2)   # tokens of the longest (= first) item
        if cur and ((len(cur) + 1) * width > max_tokens or len(cur) >= max_batch):
            batches.append(cur)
            cur = []
        cur.append(t)
    if cur:
        batches.append(cur)
    return batches


class Scorer:
    def __init__(self, device):
        self.device = torch.device(device)
        self.tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = EsmForMaskedLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).eval().to(self.device)
        vocab = self.tok.get_vocab()
        self.aa_id = {a: vocab[a] for a in AA20}
        self.cls_id, self.eos_id = self.tok.cls_token_id, self.tok.eos_token_id
        self.pad_id, self.mask_id = self.tok.pad_token_id, self.tok.mask_token_id
        self.commit = getattr(self.model.config, '_commit_hash', None)

    def encode(self, window):
        return [self.cls_id] + [self.aa_id[a] for a in window] + [self.eos_id]

    @torch.inference_mode()
    def score_batch(self, batch, refseq):
        """log p(token | context with the variant position masked) for each task -> float64 array [B, vocab]."""
        ids, mask_col = [], []
        for ctx, gene, pos, s, e in batch:
            x = self.encode(refseq[gene][s:e])
            col = 1 + (pos - 1 - s)                      # +1 for <cls>
            assert x[col] == self.aa_id[refseq[gene][pos - 1]]
            x[col] = self.mask_id
            ids.append(x)
            mask_col.append(col)
        width = max(len(x) for x in ids)
        inp = torch.full((len(ids), width), self.pad_id, dtype=torch.long)
        att = torch.zeros((len(ids), width), dtype=torch.long)
        for i, x in enumerate(ids):
            inp[i, :len(x)] = torch.tensor(x)
            att[i, :len(x)] = 1
        logits = self.model(input_ids=inp.to(self.device), attention_mask=att.to(self.device)).logits
        rows = torch.arange(len(ids), device=logits.device)
        at_mask = logits[rows, torch.tensor(mask_col, device=logits.device)].float().cpu().double()
        return torch.log_softmax(at_mask, dim=-1).numpy()

    def release(self):
        self.model.to('cpu')
        del self.model
        gc.collect()
        if self.device.type == 'mps':
            torch.mps.empty_cache()


def auroc(y, score):
    y = np.asarray(y)
    return float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--device', default='mps')
    ap.add_argument('--max-tokens', type=int, default=8 * (MAX_CTX + 2), help='padded tokens per batch (8 long contexts)')
    ap.add_argument('--max-batch', type=int, default=64, help='sequences per batch (short contexts)')
    ap.add_argument('--limit', type=int, default=0, help='smoke test: score only the first N unique positions; no files written')
    ap.add_argument('--n-selfcheck', type=int, default=6, help='positions per stratum recomputed singly on the CPU')
    args = ap.parse_args()
    t_start = time.time()
    torch.manual_seed(0)

    v = pd.read_csv(VARIANTS, usecols=['variant_id', 'gene', 'ref_pos', 'ref1', 'alt1', 'partition', 'y'])
    ref = pd.read_csv(REFERENCE, usecols=['gene', 'sequence'])
    refseq = dict(zip(ref.gene, ref.sequence))
    assert v.variant_id.is_unique

    # ---- reference-residue check (must hold by construction of variants_v8.csv)
    bad = [vid for vid, g, p, r, a in zip(v.variant_id, v.gene, v.ref_pos, v.ref1, v.alt1)
           if g not in refseq or not (1 <= p <= len(refseq[g])) or refseq[g][p - 1] != r
           or r not in AA20 or a not in AA20 or r == a]
    print(f'{len(v)} variants, {v.gene.nunique()} genes; reference-residue failures: {len(bad)} {bad[:20]}', flush=True)
    assert not bad, f'{len(bad)} variants do not match the reference protein: {bad[:50]}'

    device = args.device
    if device == 'mps' and not torch.backends.mps.is_available():
        print('WARNING: mps not available, using cpu', flush=True)
        device = 'cpu'
    scorer = Scorer(device)
    probe = refseq[v.gene.iloc[0]][:50]
    assert scorer.tok(probe)['input_ids'] == scorer.encode(probe), 'manual tokenisation differs from the tokenizer'

    tasks = build_tasks(v, refseq)
    if args.limit:
        keep = set(sorted({(t[1], t[2]) for t in tasks})[:: max(1, len(tasks) // 2 // args.limit)][: args.limit])
        tasks = [t for t in tasks if (t[1], t[2]) in keep]
    n_full = sum(t[0] == 'full' for t in tasks)
    n_windowed = sum(t[0] == 'full' and len(refseq[t[1]]) > MAX_CTX for t in tasks)
    print(f'model {MODEL_NAME} (commit {scorer.commit}) on {device}; {n_full} unique positions; '
          f'{n_windowed} use a {MAX_CTX}-residue window; {len(tasks)} forward contexts', flush=True)

    cache = {}
    t_gpu = time.time()
    for ctx in ('full', 'w32'):
        batches = make_batches([t for t in tasks if t[0] == ctx], args.max_tokens, args.max_batch)
        t0, done = time.time(), 0
        for i, batch in enumerate(batches):
            logp = scorer.score_batch(batch, refseq)
            for t, lp in zip(batch, logp):
                cache[t[:3]] = lp
            done += len(batch)
            if i % 50 == 0 or i == len(batches) - 1:
                print(f'  [{ctx}] batch {i + 1}/{len(batches)}  contexts {done}  {time.time() - t0:.0f}s', flush=True)
    gpu_seconds = time.time() - t_gpu
    scorer.release()                                   # free the GPU before any CPU-side work
    print(f'forward passes done in {gpu_seconds:.0f}s; GPU released', flush=True)

    # ---- per-variant LLR from the cached per-position log-probabilities
    aa_id = scorer.aa_id
    span = {t[:3]: (t[3] + 1, t[4]) for t in tasks}
    rows = []
    for vid, g, p, r, a in zip(v.variant_id, v.gene, v.ref_pos, v.ref1, v.alt1):
        rec = {'variant_id': vid, 'gene': g, 'ref_pos': int(p), 'ref1': r, 'alt1': a}
        for ctx, col, pre in (('full', 'esm2_150m_llr', 'ctx'), ('w32', 'esm2_150m_llr_w32', 'w32')):
            lp = cache.get((ctx, g, int(p)))
            rec[col] = float(lp[aa_id[a]] - lp[aa_id[r]]) if lp is not None else np.nan
            rec[f'{pre}_start'], rec[f'{pre}_end'] = span.get((ctx, g, int(p)), (pd.NA, pd.NA))
        rows.append(rec)
    out = pd.DataFrame(rows)[['variant_id', 'gene', 'ref_pos', 'ref1', 'alt1', 'esm2_150m_llr', 'esm2_150m_llr_w32',
                              'ctx_start', 'ctx_end', 'w32_start', 'w32_end']]
    for c in ('ctx_start', 'ctx_end', 'w32_start', 'w32_end'):
        out[c] = out[c].astype('Int64')

    # ---- self-check: fixed sample of positions recomputed one by one on the CPU (no padding, other device)
    check = {'n': 0, 'max_abs_dlogp_aa20': None, 'max_abs_dllr': None}
    if args.n_selfcheck:
        rng = np.random.default_rng(0)
        full_w = sorted(t for t in tasks if t[0] == 'full' and t[4] - t[3] == MAX_CTX)
        full_s = sorted(t for t in tasks if t[0] == 'full' and t[4] - t[3] < MAX_CTX)
        w32 = sorted(t for t in tasks if t[0] == 'w32')
        sample = []
        for pool in (full_w, full_s, w32):
            if pool:
                sample += [pool[i] for i in rng.choice(len(pool), size=min(args.n_selfcheck, len(pool)), replace=False)]
        cpu = Scorer('cpu')
        aa_cols = [cpu.aa_id[a] for a in AA20]
        d_logp, d_llr = 0.0, 0.0
        for t in sample:
            lp_cpu, lp_dev = cpu.score_batch([t], refseq)[0], cache[t[:3]]
            d_logp = max(d_logp, float(np.abs(lp_cpu[aa_cols] - lp_dev[aa_cols]).max()))
            ref_col = cpu.aa_id[refseq[t[1]][t[2] - 1]]
            d_llr = max(d_llr, float(np.abs((lp_cpu[aa_cols] - lp_cpu[ref_col]) - (lp_dev[aa_cols] - lp_dev[ref_col])).max()))
        cpu.release()
        check = {'n': len(sample), 'max_abs_dlogp_aa20': d_logp, 'max_abs_dllr': d_llr}
        print(f'self-check ({len(sample)} positions, single-sequence CPU vs batched {device}): '
              f'max |d log p| = {d_logp:.2e}, max |d LLR| = {d_llr:.2e}', flush=True)
        assert d_logp < SELFCHECK_TOL, 'batched device run disagrees with single-sequence CPU run'

    # ---- report
    scored = out.esm2_150m_llr.notna() & out.esm2_150m_llr_w32.notna()
    rho = float(spearmanr(out.esm2_150m_llr[scored], out.esm2_150m_llr_w32[scored]).statistic)
    m = v[['variant_id', 'partition', 'y']].merge(out, on='variant_id', validate='one_to_one')
    per_part = {}
    for part, g in m.groupby('partition'):
        g = g[g.esm2_150m_llr.notna() & g.y.notna()]
        per_part[part] = {'n': int(len(g)), 'n_pos': int((g.y == 1).sum()),
                          'auroc_neg_llr': auroc(g.y, -g.esm2_150m_llr),
                          'auroc_neg_llr_w32': auroc(g.y, -g.esm2_150m_llr_w32)}
    summary = {'model': MODEL_NAME, 'model_commit': scorer.commit, 'device': device, 'dtype': 'float32',
               'torch': torch.__version__, 'n_variants': int(len(out)), 'n_scored': int(scored.sum()),
               'n_reference_residue_failures': len(bad), 'n_unique_positions': n_full,
               'n_positions_windowed_1022': n_windowed, 'n_forward_contexts': len(tasks),
               'forward_seconds': round(gpu_seconds, 1), 'total_seconds': round(time.time() - t_start, 1),
               'spearman_llr_vs_llr_w32': rho, 'mean_llr': float(out.esm2_150m_llr.mean()),
               'mean_llr_w32': float(out.esm2_150m_llr_w32.mean()), 'auroc_by_partition': per_part,
               'selfcheck_cpu_single_vs_batched': check}
    print(json.dumps(summary, indent=1), flush=True)
    if args.limit:
        print('smoke test (--limit): nothing written')
        return
    assert len(out) == len(v) and (out.variant_id.values == v.variant_id.values).all()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False, float_format='%.6f')
    json.dump(summary, open(OUT_SUMMARY, 'w'), indent=2)
    print(f'wrote {OUT} ({len(out)} rows) and {OUT_SUMMARY.name}; total {time.time() - t_start:.0f}s')


if __name__ == '__main__':
    main()
