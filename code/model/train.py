#!/usr/bin/env python3
"""
Train one HEARS-LM run.

Fixed split (default): train on partition `fit`, monitor `dev` only, write predictions for
dev / test_db / test_time / test_gene after the last epoch (no checkpoint selection).

Leave-gene-out cross-validation (--cv-fold k): train on all labelled variants whose gene is not in fold k
(final labels), write predictions for fold k.

Examples
  python train.py --name seq_ddg --seed 0
  python train.py --name full --features cons struct hear --seed 1
  python train.py --name seq --beta-ddg 0 --seed 0
  python train.py --name cv_seq_ddg --cv-fold 3
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hearslm import DATA, DEFAULT_CONFIG, ROOT, train_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True, help='experiment arm name (output folder)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--features', nargs='*', default=[])
    ap.add_argument('--no-sequence', action='store_true')
    ap.add_argument('--beta-ddg', type=float, default=None)
    ap.add_argument('--backbone', default=None)
    ap.add_argument('--lora-r', type=int, default=None)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--batch-size', type=int, default=None)
    ap.add_argument('--lr-lora', type=float, default=None)
    ap.add_argument('--lr-head', type=float, default=None)
    ap.add_argument('--cv-fold', type=int, default=None)
    ap.add_argument('--window', type=int, default=None)
    ap.add_argument('--lm-logits', action='store_true')
    ap.add_argument('--dev-gene-fold', type=int, default=None,
                    help='design pilots only: remove fit variants of this family fold from training and score them as devgene')
    ap.add_argument('--all-data', action='store_true',
                    help='release model: train on every variant with a final (2026-04) label, all partitions; '
                         'no evaluation data remain, so performance and calibration are those of the evaluated model')
    ap.add_argument('--exclude-genes', nargs='*', default=[], help='remove these genes from training')
    ap.add_argument('--train-fraction', type=float, default=1.0)
    ap.add_argument('--runs-dir', default=str(ROOT / 'runs'))
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()

    cfg = dict(DEFAULT_CONFIG)
    cfg.update({'seed': a.seed, 'feature_groups': a.features, 'use_sequence': not a.no_sequence})
    for k, v in [('beta_ddg', a.beta_ddg), ('backbone', a.backbone), ('lora_r', a.lora_r), ('epochs', a.epochs),
                 ('batch_size', a.batch_size), ('lr_lora', a.lr_lora), ('lr_head', a.lr_head), ('window', a.window)]:
        if v is not None:
            cfg[k] = v
    if cfg['lora_r'] and cfg['lora_r'] != 16:
        cfg['lora_alpha'] = 2 * cfg['lora_r']
    cfg['arm'] = a.name
    cfg['use_lm_logits'] = bool(a.lm_logits)
    cfg['dev_gene_fold'] = a.dev_gene_fold
    cfg['cv_fold'] = a.cv_fold
    cfg['all_data'] = bool(a.all_data)
    cfg['exclude_genes'] = a.exclude_genes

    sub = f'fold{a.cv_fold}' if a.cv_fold is not None else f'seed{a.seed}'
    out = Path(a.runs_dir) / a.name / sub
    if (out / 'result.json').exists() and not a.force:
        print(f'[skip] {out} already finished')
        return
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / 'log.txt', 'a')

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + '\n'); logf.flush()

    v = pd.read_csv(DATA / 'variants_v8.csv', low_memory=False)
    if a.all_data:
        train_df = v[v.y_final.notna()].copy()
        train_df['y'] = train_df['y_final']
        eval_dfs = {}
    elif a.cv_fold is None:
        train_df = v[v.partition == 'fit'].copy()
        if a.dev_gene_fold is not None:      # design pilot: gene-level development set carved out of fit
            eval_dfs = {'dev': v[v.partition == 'dev'].copy(),
                        'devgene': train_df[train_df.cv_fold == a.dev_gene_fold].copy()}
            train_df = train_df[train_df.cv_fold != a.dev_gene_fold].copy()
        else:
            eval_dfs = {p: v[v.partition == p].copy() for p in ['dev', 'test_db', 'test_time', 'test_gene']}
    else:
        pool = v[v.cv_fold >= 0].copy()
        pool['y'] = pool['y_final']
        train_df = pool[pool.cv_fold != a.cv_fold].copy()
        eval_dfs = {'cvtest': pool[pool.cv_fold == a.cv_fold].copy()}
    if a.exclude_genes:
        train_df = train_df[~train_df.gene.isin(a.exclude_genes)].copy()
    if a.train_fraction < 1.0:
        train_df = train_df.sample(frac=a.train_fraction, random_state=a.seed).copy()
    log(f'=== {a.name}/{sub} config: {json.dumps(cfg)}')
    train_model(cfg, train_df, eval_dfs, out, log=log)


if __name__ == '__main__':
    main()
