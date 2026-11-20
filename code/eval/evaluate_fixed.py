#!/usr/bin/env python3
"""
Evaluation of the fixed-split experiments (fit -> dev / test_time / test_gene / test_db).

Outputs (results/)
  arms_per_seed.csv        arm x seed x partition: AUROC, AUPRC, within-gene AUROC, ddG Spearman
  arms_summary.csv         arm x partition: mean, SD over seeds; seed-ensemble AUROC/AUPRC with bootstrap CI
  final_model.json         arm chosen on the development set by the pre-specified rule (02_ANALYSIS_PLAN.md)
  arm_contrasts.csv        final / reference arm vs every other arm: paired bootstrap + DeLong on ensembles,
                           Welch t-test across seeds
  benchmark_pairwise.csv   final model vs each external predictor on the variants that predictor scores
  benchmark_common.csv     all core predictors on one common variant set per test partition
  benchmark_strata.csv     final model and key baselines within strata (position-novel, review stars,
                           previous ClinVar state, inheritance, label source)

Usage: python evaluate_fixed.py --stage dev [--final LOCK]   then   python evaluate_fixed.py --stage test
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_ind

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics as M  # noqa: E402
from common import (DATA, PARTS, RESULTS, RUNS, TESTS, gene_prior, list_seeds, load_baselines, load_preds,  # noqa: E402
                    load_variants, panel)

# candidate arms for the released model, ordered from fewest to most inputs (AF arm is diagnostic only)
CANDIDATES = ['seq', 'seq_plm', 'seq_ddg', 'seq_plm_ddg', 'seq_plm_ddg_struct', 'seq_plm_ddg_cons',
              'seq_plm_ddg_hear', 'full']
# The common comparison panel is defined by a coverage rule (common.panel): dbNSFP predictors that score at least 90%
# of the variants of every test partition, plus the unadapted backbone. Predictors with lower coverage (EVE, MutPred2,
# MPC, classic SIFT, PolyPhen-2) appear in benchmark_pairwise.csv, each on the variants it scores.


def all_arms():
    return sorted([p.name for p in RUNS.iterdir() if p.is_dir() and list_seeds(p.name)]) if RUNS.exists() else []


def per_seed_table(arms, v):
    gene = v.set_index('variant_id').gene
    rows = []
    for arm in arms:
        for s in list_seeds(arm):
            for part in PARTS:
                f = RUNS / arm / s / f'preds_{part}.csv'
                if not f.exists():
                    continue
                p = pd.read_csv(f)
                r = {'arm': arm, 'seed': s, 'partition': part, 'n': len(p),
                     'auroc': M.auroc(p.y, p.prob), 'auprc': M.auprc(p.y, p.prob),
                     'auroc_within_gene': M.within_group_auroc(p.y, p.prob, gene.reindex(p.variant_id).to_numpy())}
                if 'ddg_pred' in p and p.ddg_target.notna().sum() > 20:
                    k = p.ddg_target.notna()
                    r['ddg_spearman'] = float(spearmanr(p.ddg_pred[k], p.ddg_target[k]).statistic)
                    r['n_ddg'] = int(k.sum())
                rows.append(r)
    return pd.DataFrame(rows)


def summarise(ps, arms, v, nboot, parts=PARTS):
    gene = v.set_index('variant_id').gene
    rows = []
    for arm in arms:
        for part in parts:
            d = ps[(ps.arm == arm) & (ps.partition == part)]
            if d.empty:
                continue
            e = load_preds(arm, part)
            ci = M.bootstrap_ci(e.y.to_numpy(), e.prob.to_numpy(), M.auroc, nboot)
            cp = M.bootstrap_ci(e.y.to_numpy(), e.prob.to_numpy(), M.auprc, nboot)
            rows.append({'arm': arm, 'partition': part, 'n_seeds': len(d), 'n': int(d.n.iloc[0]),
                         'auroc_mean': d.auroc.mean(), 'auroc_sd': d.auroc.std(ddof=1) if len(d) > 1 else np.nan,
                         'auprc_mean': d.auprc.mean(), 'auprc_sd': d.auprc.std(ddof=1) if len(d) > 1 else np.nan,
                         'within_gene_mean': d.auroc_within_gene.mean(),
                         'within_gene_sd': d.auroc_within_gene.std(ddof=1) if len(d) > 1 else np.nan,
                         'ens_auroc': ci['est'], 'ens_auroc_lo': ci['lo'], 'ens_auroc_hi': ci['hi'],
                         'ens_auprc': cp['est'], 'ens_auprc_lo': cp['lo'], 'ens_auprc_hi': cp['hi'],
                         'ens_within_gene': M.within_group_auroc(e.y, e.prob, gene.reindex(e.variant_id).to_numpy()),
                         'ddg_spearman_mean': d.ddg_spearman.mean() if 'ddg_spearman' in d else np.nan})
    return pd.DataFrame(rows)


def choose_final(summ):
    dev = summ[(summ.partition == 'dev') & summ.arm.isin(CANDIDATES)].set_index('arm')
    if dev.empty:
        return None
    best = dev.auroc_mean.idxmax()
    sd = dev.loc[best, 'auroc_sd']
    tol = 0.0 if np.isnan(sd) else sd
    eligible = [a for a in CANDIDATES if a in dev.index and dev.loc[a, 'auroc_mean'] >= dev.loc[best, 'auroc_mean'] - tol]
    choice = eligible[0]
    return {'final_arm': choice, 'best_dev_arm': best, 'best_dev_auroc_mean': float(dev.loc[best, 'auroc_mean']),
            'best_dev_auroc_sd': None if np.isnan(sd) else float(sd), 'eligible_within_1sd': eligible,
            'dev_table': dev[['n_seeds', 'auroc_mean', 'auroc_sd']].round(4).reset_index().to_dict('records'),
            'rule': 'highest mean dev AUROC; simplest arm within 1 SD (across seeds) of the best is selected',
            'seeds': list_seeds(choice)}


def contrasts(ref, arms, ps, nboot):
    rows = []
    for part in PARTS:
        a = load_preds(ref, part)
        if a is None:
            continue
        for other in arms:
            if other == ref:
                continue
            b = load_preds(other, part)
            if b is None:
                continue
            m = a[['variant_id', 'y', 'prob']].merge(b[['variant_id', 'prob']], on='variant_id', suffixes=('_ref', '_other'))
            pb = M.paired_bootstrap_diff(m.y.to_numpy(), m.prob_ref.to_numpy(), m.prob_other.to_numpy(), M.auroc, nboot)
            dl = M.delong_test(m.y.to_numpy(), m.prob_ref.to_numpy(), m.prob_other.to_numpy())
            x = ps[(ps.arm == ref) & (ps.partition == part)].auroc
            y = ps[(ps.arm == other) & (ps.partition == part)].auroc
            tt = ttest_ind(x, y, equal_var=False) if len(x) > 1 and len(y) > 1 else None
            rows.append({'partition': part, 'ref_arm': ref, 'other_arm': other, 'n': len(m),
                         'ens_auroc_ref': dl['auc1'], 'ens_auroc_other': dl['auc2'], 'diff': pb['diff'],
                         'diff_lo': pb['lo'], 'diff_hi': pb['hi'], 'p_delong': dl['p'],
                         'seed_mean_ref': x.mean(), 'seed_mean_other': y.mean(), 'seed_diff': x.mean() - y.mean(),
                         'p_welch_seeds': float(tt.pvalue) if tt is not None else np.nan,
                         'n_seeds_ref': len(x), 'n_seeds_other': len(y)})
    out = pd.DataFrame(rows)
    if len(out):
        out['q_delong'] = M.bh_fdr(out.p_delong.to_numpy())
    return out


def benchmark(final, v, nboot):
    base, af = load_baselines()
    CORE = panel(final)
    print('common comparison panel:', CORE)
    vi = v.set_index('variant_id')
    gp = gene_prior(v)
    gp.index = v.variant_id
    pair_rows, common_rows, strata_rows = [], [], []
    afcol = 'af_any' if 'af_any' in af else None
    for part in TESTS:
        e = load_preds(final, part)
        if e is None:
            continue
        t = e[['variant_id', 'y', 'prob']].set_index('variant_id')
        t['gene'] = vi.gene.reindex(t.index)
        ext = base.reindex(t.index).copy()
        ext['Gene prior (training label fraction)'] = gp.reindex(t.index)
        if afcol:
            ext['Allele frequency only (-log10 AF)'] = -np.log10(af[afcol].reindex(t.index).fillna(0) + 1e-7)
        y = t.y.to_numpy(); s = t.prob.to_numpy(); genes = t.gene.to_numpy()
        own = M.bootstrap_ci(y, s, M.auroc, nboot)
        pair_rows.append({'partition': part, 'method': 'HEARS-LM', 'n': len(t), 'coverage': 1.0,
                          'auroc': own['est'], 'auroc_lo': own['lo'], 'auroc_hi': own['hi'],
                          'auprc': M.auprc(y, s), 'within_gene_auroc': M.within_group_auroc(y, s, genes)})
        for name in ext.columns:
            k = ext[name].notna().to_numpy()
            if k.sum() < 30 or len(np.unique(y[k])) < 2:
                continue
            b = ext[name].to_numpy()[k]
            ci = M.bootstrap_ci(y[k], b, M.auroc, nboot)
            pb = M.paired_bootstrap_diff(y[k], s[k], b, M.auroc, nboot)
            dl = M.delong_test(y[k], s[k], b)
            pair_rows.append({'partition': part, 'method': name, 'n': int(k.sum()), 'coverage': float(k.mean()),
                              'auroc': ci['est'], 'auroc_lo': ci['lo'], 'auroc_hi': ci['hi'],
                              'auprc': M.auprc(y[k], b), 'within_gene_auroc': M.within_group_auroc(y[k], b, genes[k]),
                              'hearslm_auroc_same_variants': dl['auc1'],
                              'hearslm_within_gene_same_variants': M.within_group_auroc(y[k], s[k], genes[k]),
                              'diff_hearslm_minus_method': pb['diff'], 'diff_lo': pb['lo'], 'diff_hi': pb['hi'],
                              'p_delong': dl['p']})
        # common subset over core predictors
        core = [c for c in CORE if c in ext.columns]
        k = ext[core].notna().all(axis=1).to_numpy()
        if k.sum() >= 30 and len(np.unique(y[k])) == 2:
            for name, sc in [('HEARS-LM', s)] + [(c, ext[c].to_numpy()) for c in core] + \
                    [(c, ext[c].to_numpy()) for c in ext.columns if c.startswith(('Gene prior', 'Allele frequency'))]:
                kk = k & ~np.isnan(sc)
                ci = M.bootstrap_ci(y[kk], sc[kk], M.auroc, nboot)
                cp = M.bootstrap_ci(y[kk], sc[kk], M.auprc, nboot)
                common_rows.append({'partition': part, 'method': name, 'n': int(kk.sum()), 'n_P': int(y[kk].sum()),
                                    'auroc': ci['est'], 'auroc_lo': ci['lo'], 'auroc_hi': ci['hi'],
                                    'auprc': cp['est'], 'auprc_lo': cp['lo'], 'auprc_hi': cp['hi'],
                                    'within_gene_auroc': M.within_group_auroc(y[kk], sc[kk], genes[kk])})
        # strata
        meta = vi.reindex(t.index)
        strata = {'all': np.ones(len(t), bool),
                  'position_novel': ~meta.pos_in_fit.to_numpy(bool), 'position_seen': meta.pos_in_fit.to_numpy(bool),
                  'stars>=2': (meta.cv_stars.fillna(0) >= 2).to_numpy(), 'stars==1': (meta.cv_stars == 1).to_numpy(),
                  'prev_absent': (meta.prev_state_2025 == 'ABSENT').to_numpy(),
                  'prev_VUS_or_conflict': meta.prev_state_2025.isin(['VUS', 'CONFLICT']).to_numpy(),
                  'source_dvd_only': (meta.source == 'dvd').to_numpy(),
                  'source_clinvar': meta.source.isin(['clinvar', 'clinvar+dvd']).to_numpy(),
                  # labels that rest only on the DVD allele-frequency cut-off removed
                  'excl_maf_cutoff_only_labels': ~((meta.source == 'dvd') & (meta.dvd_basis == 'maf_cutoff')).to_numpy(),
                  # number of training variants of the gene (how well the gene is represented in training)
                  'gene_train_lt10': (meta.n_fit_gene.fillna(0) < 10).to_numpy(),
                  'gene_train_ge10': (meta.n_fit_gene.fillna(0) >= 10).to_numpy()}
        strata_methods = [c for c in CORE if c in ext.columns] + \
            [c for c in ext.columns if c.startswith(('Gene prior', 'Allele frequency'))]
        core_mask = ext[[c for c in CORE if c in ext.columns]].notna().all(axis=1).to_numpy()
        for sname, sm in strata.items():
            sm = sm & core_mask                        # identical variants for every method within the stratum
            if sm.sum() < 30 or len(np.unique(y[sm])) < 2 or min((y[sm] == 1).sum(), (y[sm] == 0).sum()) < 10:
                continue
            for name, sc in [('HEARS-LM', s)] + [(c, ext[c].to_numpy()) for c in strata_methods]:
                kk = sm & ~np.isnan(sc)
                if kk.sum() < 30 or len(np.unique(y[kk])) < 2:
                    continue
                ci = M.bootstrap_ci(y[kk], sc[kk], M.auroc, max(500, nboot // 2))
                strata_rows.append({'partition': part, 'stratum': sname, 'method': name, 'n': int(kk.sum()),
                                    'n_P': int(y[kk].sum()), 'auroc': ci['est'], 'auroc_lo': ci['lo'],
                                    'auroc_hi': ci['hi'],
                                    'within_gene_auroc': M.within_group_auroc(y[kk], sc[kk], genes[kk])})
    pw = pd.DataFrame(pair_rows)
    if len(pw) and 'p_delong' in pw:
        pw['q_delong'] = M.bh_fdr(pw.p_delong.to_numpy())
    return pw, pd.DataFrame(common_rows), pd.DataFrame(strata_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--final', default=None, help="'LOCK' (with --stage dev) writes the selection to results/final_model.json")
    ap.add_argument('--nboot', type=int, default=2000)
    ap.add_argument('--skip-benchmark', action='store_true')
    ap.add_argument('--only-benchmark', action='store_true', help='recompute only the comparison with other predictors')
    ap.add_argument('--stage', choices=['dev', 'test'], required=True,
                    help='dev: development-set summary and model selection only; test: everything (needs a locked selection)')
    a = ap.parse_args()
    v = load_variants()
    if a.only_benchmark:
        sel = json.load(open(RESULTS / 'final_model.json'))
        assert 'locked_at' in sel, 'model selection has not been locked on the development set'
        pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 300)
        pw, cm, st = benchmark(sel['final_arm'], v, a.nboot)
        pw.to_csv(RESULTS / 'benchmark_pairwise.csv', index=False)
        cm.to_csv(RESULTS / 'benchmark_common.csv', index=False)
        st.to_csv(RESULTS / 'benchmark_strata.csv', index=False)
        print(cm.round(4).to_string())
        return
    arms = [x for x in all_arms() if not x.startswith(('pilots', '_', 'cv_'))]
    ps = per_seed_table(arms, v)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 300)
    if a.stage == 'dev':
        dps = ps[ps.partition == 'dev']
        dps.to_csv(RESULTS / 'arms_per_seed_dev.csv', index=False)
        summ = summarise(dps, arms, v, a.nboot, parts=['dev'])
        summ.to_csv(RESULTS / 'arms_summary_dev.csv', index=False)
        sel = choose_final(summ)
        print(summ[['arm', 'n_seeds', 'auroc_mean', 'auroc_sd', 'auprc_mean', 'within_gene_mean', 'ens_auroc',
                    'ens_auroc_lo', 'ens_auroc_hi', 'ddg_spearman_mean']].round(4).to_string())
        print('selection (provisional until locked):', json.dumps(sel, indent=1))
        if sel is not None and a.final == 'LOCK':
            sel['locked_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
            json.dump(sel, open(RESULTS / 'final_model.json', 'w'), indent=1)
            print('LOCKED ->', RESULTS / 'final_model.json')
        return
    sel = json.load(open(RESULTS / 'final_model.json'))
    assert 'locked_at' in sel, 'model selection has not been locked on the development set'
    ps.to_csv(RESULTS / 'arms_per_seed.csv', index=False)
    summ = summarise(ps, arms, v, a.nboot)
    summ.to_csv(RESULTS / 'arms_summary.csv', index=False)
    final = sel['final_arm']
    print('final model (locked', sel['locked_at'], '):', final)
    con = contrasts(final, arms, ps, a.nboot)
    con.to_csv(RESULTS / 'arm_contrasts.csv', index=False)
    print(summ[['arm', 'partition', 'n_seeds', 'auroc_mean', 'auroc_sd', 'ens_auroc', 'ens_auroc_lo', 'ens_auroc_hi',
                'within_gene_mean', 'ddg_spearman_mean']].round(4).to_string())
    if a.skip_benchmark or not (DATA / 'annotations/dbnsfp_v8.csv').exists():
        return
    pw, cm, st = benchmark(final, v, a.nboot)
    pw.to_csv(RESULTS / 'benchmark_pairwise.csv', index=False)
    cm.to_csv(RESULTS / 'benchmark_common.csv', index=False)
    st.to_csv(RESULTS / 'benchmark_strata.csv', index=False)
    print(cm.round(4).to_string())


if __name__ == '__main__':
    main()
