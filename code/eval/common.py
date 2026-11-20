#!/usr/bin/env python3
"""Shared loaders for the evaluation scripts (one place that defines how predictors are read and oriented)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data'
RUNS = ROOT / 'runs'
RESULTS = ROOT / 'results'
RESULTS.mkdir(exist_ok=True)

TESTS = ['test_time', 'test_gene', 'test_db']
PARTS = ['dev'] + TESTS

# display names for dbNSFP predictors (column sc_<key> in data/annotations/dbnsfp_v8.csv)
BASELINE_NAMES = {
    'alphamissense': 'AlphaMissense', 'revel': 'REVEL', 'metarnn': 'MetaRNN', 'bayesdel_noaf': 'BayesDel (no AF)',
    'bayesdel_addaf': 'BayesDel (with AF)', 'vest4': 'VEST4', 'primateai': 'PrimateAI', 'esm1b': 'ESM-1b',
    'polyphen2_hvar': 'PolyPhen-2', 'mutpred': 'MutPred', 'cadd_phred': 'CADD', 'eve': 'EVE', 'sift': 'SIFT',
    'sift4g': 'SIFT4G', 'clinpred': 'ClinPred', 'varity_r': 'VARITY_R', 'mpc': 'MPC', 'deogen2': 'DEOGEN2',
}


def load_variants():
    v = pd.read_csv(DATA / 'variants_v8.csv', low_memory=False)
    v['pos_key'] = v.gene + ':' + v.ref_pos.astype(str)
    return v


def load_baselines(dbnsfp=None):
    """Return DataFrame indexed by variant_id with one column per external predictor, all oriented so that
    higher = more pathogenic, plus ESM-2 zero-shot and allele-frequency columns.
    dbnsfp: annotation table to read (default: labelled variants, data/annotations/dbnsfp_v8.csv)."""
    d = pd.read_csv(dbnsfp or DATA / 'annotations/dbnsfp_v8.csv', low_memory=False).set_index('variant_id')
    rules = json.load(open(DATA / 'annotations/dbnsfp_rules.json'))['predictors']
    out = pd.DataFrame(index=d.index)
    for c in d.columns:
        if not c.startswith('sc_'):
            continue
        assert c in rules, f'no direction recorded for {c}'
        hi = rules[c]['higher_is_pathogenic']
        out[BASELINE_NAMES.get(c[3:], c[3:])] = d[c] if hi else -d[c]
    z = DATA / 'annotations/esm2_zeroshot_v8.csv'
    if z.exists():
        e = pd.read_csv(z).set_index('variant_id')
        out['ESM-2 150M zero-shot'] = -e['esm2_150m_llr'].reindex(out.index)
        if 'esm2_150m_llr_w32' in e:
            out['ESM-2 150M zero-shot (65-aa window)'] = -e['esm2_150m_llr_w32'].reindex(out.index)
    af = d[[c for c in d.columns if c.startswith('af_')]].copy()
    return out, af


def gene_prior(v):
    """Leak-free gene-level prior: Laplace-smoothed fraction of pathogenic training (fit) variants per gene."""
    fit = v[v.partition == 'fit']
    g = fit.groupby('gene').y.agg(['sum', 'size'])
    prior = (g['sum'] + 1) / (g['size'] + 2)
    overall = (fit.y.sum() + 1) / (len(fit) + 2)
    return v.gene.map(prior).fillna(overall)


def list_seeds(arm):
    d = RUNS / arm
    return sorted([p.name for p in d.glob('seed*') if (p / 'result.json').exists()]) if d.exists() else []


def load_preds(arm, part, seeds=None):
    """Wide table: variant_id, y, prob_<seed>..., prob (mean over seeds), ddg_pred (mean) if present."""
    seeds = seeds or list_seeds(arm)
    out = None
    for s in seeds:
        p = pd.read_csv(RUNS / arm / s / f'preds_{part}.csv')
        cols = {'prob': f'prob_{s}'}
        if 'ddg_pred' in p:
            cols['ddg_pred'] = f'ddg_{s}'
        p = p.rename(columns=cols)
        keep = ['variant_id', 'y', f'prob_{s}'] + ([f'ddg_{s}'] if 'ddg_pred' in cols else []) + \
               (['ddg_target'] if out is None and 'ddg_target' in p else [])
        out = p[keep] if out is None else out.merge(p[[c for c in keep if c not in ('y', 'ddg_target')]], on='variant_id')
    if out is None:
        return None
    out['prob'] = out[[f'prob_{s}' for s in seeds]].mean(axis=1)
    dc = [f'ddg_{s}' for s in seeds if f'ddg_{s}' in out]
    if dc:
        out['ddg_pred'] = out[dc].mean(axis=1)
    return out


def fmt_ci(d, k='est'):
    return f"{d[k]:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]"

# --- comparison panel ---------------------------------------------------------------------------------------------
# A dbNSFP predictor enters the common comparison panel if it scores at least MIN_PANEL_COVERAGE of the variants of
# every test partition (final model's test predictions). Predictors below that are compared with HEARS-LM pairwise on
# the variants they score. The same panel is used for the test sets and for the cross-validation.
MIN_PANEL_COVERAGE = 0.90
ZERO_SHOT = 'ESM-2 150M zero-shot'
GENE_PRIOR = 'Gene prior (training label fraction)'
AF_ONLY = 'Allele frequency only (-log10 AF)'


def panel_coverage(arm):
    base, _ = load_baselines()
    cols = [c for c in base.columns if not c.startswith('ESM-2 150M zero-shot')]
    cov = {}
    for part in TESTS:
        t = load_preds(arm, part)
        cov[part] = base[cols].reindex(t.variant_id).notna().mean()
    return pd.DataFrame(cov)


def panel(arm):
    """Predictors of the common comparison panel (coverage rule) plus the unadapted backbone, in a fixed order."""
    cov = panel_coverage(arm)
    keep = cov.index[(cov >= MIN_PANEL_COVERAGE).all(axis=1)].tolist()
    return sorted(keep) + [ZERO_SHOT]

