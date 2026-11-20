#!/usr/bin/env python3
"""
Clinical-use analyses of the locked final model (run after `evaluate_fixed.py --stage test`).

1. ACMG/AMP evidence-strength calibration (Pejaver et al. 2022; prior 0.0441): local posterior probability of
   pathogenicity as a function of the score, one-sided 95% bootstrap bounds, score thresholds for
   PP3 / BP4 at Supporting, Moderate, Strong (Very Strong reported if reached). Calibration set: all test
   variants (test_time + test_gene + test_db), none of which was used for training or model selection.
   Same procedure for the comparison predictors on the same variants.
   -> results/calibration_thresholds.csv, results/calibration_curves.csv, results/evidence_assignment.csv
2. Reliability of the raw probabilities per test partition (Brier, ECE, binned observed vs predicted).
   -> results/reliability.csv, results/reliability_bins.csv
3. ClinGen Hearing Loss VCEP expert classifications: discrimination of expert P/LP vs B/LB on variants with an
   out-of-fold score (leave-family-out cross-validation), compared with predictors on identical variants.
   -> results/expert_panel.csv, results/expert_panel_variants.csv
4. Reclassification: (a) test variants (all test sets) that were VUS / conflicting in ClinVar 2025-01 and received a
   definitive ClinVar classification by 2026-04, labelled by it; (b) VUS->P/LP upgrades versus variants still VUS in
   2026-04. All methods on identical variants.
   ROC and precision-recall with bootstrap CIs; prevalence reported. -> results/reclassification.csv,
   results/reclassification_curves.csv
5. VUS prioritisation: calibrated evidence level for every ClinVar 2026-04 VUS / conflicting missense variant in
   genes that contributed training variants to the scoring model (score only for the other genes), stratified by
   whether a labelled P/LP missense variant exists at the same residue (PS1/PM5 territory).
   -> results/vus_catalogue.csv, results/vus_catalogue_summary.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics as M  # noqa: E402
from common import DATA, RESULTS, RUNS, TESTS, load_baselines, load_preds, load_variants  # noqa: E402

COMPARATORS = ['AlphaMissense', 'REVEL', 'BayesDel (no AF)', 'VEST4', 'MetaRNN', 'ClinPred', 'ESM-2 150M zero-shot']
STRENGTHS = ['supporting', 'moderate', 'strong', 'very_strong']
NOT_ASSIGNED = 'not assigned (gene absent from the training data of the scoring model)'
NBOOT = 2000


def final_arm():
    sel = json.load(open(RESULTS / 'final_model.json'))
    assert 'locked_at' in sel
    return sel['final_arm']


def test_predictions(arm):
    parts = []
    for p in TESTS:
        e = load_preds(arm, p)
        parts.append(e[['variant_id', 'y', 'prob']].assign(partition=p))
    return pd.concat(parts, ignore_index=True)


def oof_predictions(arm):
    fs = sorted((RUNS / f'cv_{arm}').glob('fold*/preds_cvtest.csv'))
    if not fs:
        return None
    return pd.concat([pd.read_csv(f)[['variant_id', 'y', 'prob']] for f in fs], ignore_index=True)


def assign_evidence(score, thr):
    """Strongest PP3 / BP4 level reached by each score (thresholds from calibrate())."""
    out = np.array(['indeterminate'] * len(score), dtype=object)
    for lvl in STRENGTHS:                      # increasing strength: later assignments overwrite
        t = thr.get(f'PP3_{lvl}', np.nan)
        if not np.isnan(t):
            out[score >= t] = f'PP3_{lvl}'
        t = thr.get(f'BP4_{lvl}', np.nan)
        if not np.isnan(t):
            out[score <= t] = f'BP4_{lvl}'
    return out


def calibrate(cal, ext):
    rows, curves, assign = [], [], []
    methods = [('HEARS-LM', cal.prob.to_numpy())] + [(m, ext[m].to_numpy()) for m in COMPARATORS if m in ext]
    common = np.all([~np.isnan(s) for _, s in methods], axis=0)
    y = cal.y.to_numpy()
    for name, s in methods:
        k = common
        r = M.evidence_thresholds(s[k], y[k], n_boot=NBOOT, seed=0)
        thr = r['thresholds']
        ev = assign_evidence(s[k], thr)
        for lvl in STRENGTHS:
            for side in ['PP3', 'BP4']:
                t = thr[f'{side}_{lvl}']
                hit = ev == f'{side}_{lvl}'
                rows.append({'method': name, 'evidence': f'{side}_{lvl}', 'threshold': t,
                             'required_posterior': r['required_posterior'][f'{side}_{lvl}'],
                             'n_P_assigned': int((hit & (y[k] == 1)).sum()), 'n_B_assigned': int((hit & (y[k] == 0)).sum()),
                             'frac_P_assigned': float((hit & (y[k] == 1)).sum() / (y[k] == 1).sum()),
                             'frac_B_assigned': float((hit & (y[k] == 0)).sum() / (y[k] == 0).sum())})
        curves.append(pd.DataFrame({'method': name, 'score': r['grid'], 'posterior': r['posterior'],
                                    'lower95': r['lower95'], 'upper95': r['upper95']}))
        assign.append(pd.DataFrame({'variant_id': cal.variant_id.to_numpy()[k], 'method': name, 'y': y[k],
                                    'score': s[k], 'evidence': ev}))
        # discordant evidence (pathogenic variant at BP4 or benign at PP3), the clinically harmful errors
        rows.append({'method': name, 'evidence': 'ANY_DISCORDANT',
                     'n_P_assigned': int(((y[k] == 1) & pd.Series(ev).str.startswith('BP4').to_numpy()).sum()),
                     'n_B_assigned': int(((y[k] == 0) & pd.Series(ev).str.startswith('PP3').to_numpy()).sum())})
    out = pd.DataFrame(rows)
    out['n_calibration'] = int(common.sum())
    out['n_P'] = int((y[common] == 1).sum())
    out['n_B'] = int((y[common] == 0).sum())
    return out, pd.concat(curves), pd.concat(assign)


def reliability(tp):
    rows, bins = [], []
    for p, d in tp.groupby('partition'):
        y, s = d.y.to_numpy(), d.prob.to_numpy()
        rows.append({'partition': p, 'n': len(d), 'prevalence': float(y.mean()), 'mean_prob': float(s.mean()),
                     'brier': M.brier(y, s), 'ece': M.ece(y, s)})
        edges = np.linspace(0, 1, 11)
        idx = np.clip(np.digitize(s, edges[1:-1]), 0, 9)
        for b in range(10):
            m = idx == b
            if m.sum():
                bins.append({'partition': p, 'bin': b, 'n': int(m.sum()), 'mean_prob': float(s[m].mean()),
                             'observed': float(y[m].mean())})
    return pd.DataFrame(rows), pd.DataFrame(bins)


PAIRED = []          # paired comparisons with the first score (HEARS-LM), filled by roc_pr_rows


def roc_pr_rows(tag, y, scores, nboot=NBOOT):
    """ROC/PR statistics for each score on the variants that ALL scores cover (identical variant set), and the paired
    difference of the first score (HEARS-LM) from every other one (bootstrap interval, DeLong test)."""
    rows, curves = [], []
    common = np.all([~np.isnan(s) for _, s in scores], axis=0)
    ref = scores[0][1]
    for name, s in scores[1:]:
        if common.sum() >= 20 and len(np.unique(y[common])) == 2:
            pb = M.paired_bootstrap_diff(y[common], ref[common], s[common], M.auroc, nboot)
            dl = M.delong_test(y[common], ref[common], s[common])
            PAIRED.append({'analysis': tag, 'method': name, 'diff_hearslm_minus_method': pb['diff'],
                           'diff_lo': pb['lo'], 'diff_hi': pb['hi'], 'p_delong': dl['p']})
    for name, s in scores:
        k = common
        if k.sum() < 20 or len(np.unique(y[k])) < 2:
            continue
        a = M.bootstrap_ci(y[k], s[k], M.auroc, nboot)
        b = M.bootstrap_ci(y[k], s[k], M.auprc, nboot)
        rows.append({'analysis': tag, 'method': name, 'n': int(k.sum()), 'n_pos': int(y[k].sum()),
                     'prevalence': float(y[k].mean()), 'auroc': a['est'], 'auroc_lo': a['lo'], 'auroc_hi': a['hi'],
                     'auprc': b['est'], 'auprc_lo': b['lo'], 'auprc_hi': b['hi']})
        fpr, tpr, _ = roc_curve(y[k], s[k])
        pr, rc, _ = precision_recall_curve(y[k], s[k])
        curves.append(pd.DataFrame({'analysis': tag, 'method': name, 'curve': 'roc', 'x': fpr, 'y': tpr}))
        curves.append(pd.DataFrame({'analysis': tag, 'method': name, 'curve': 'pr', 'x': rc, 'y': pr}))
    return rows, curves


def _yields(ev, y):
    ev = pd.Series(ev)
    return {'n_P': int((y == 1).sum()), 'n_B': int((y == 0).sum()),
            'P_with_PP3': float(ev[y == 1].str.startswith('PP3').mean()),
            'B_with_BP4': float(ev[y == 0].str.startswith('BP4').mean()),
            'P_with_BP4': float(ev[y == 1].str.startswith('BP4').mean()),
            'B_with_PP3': float(ev[y == 0].str.startswith('PP3').mean())}


def crossfit(tp, ext, k_folds=5):
    """Out-of-sample evidence yields (Amendment 3).
    kfold: the pooled test variants scored by every method are split into five folds stratified by partition and
      label; thresholds are derived on four folds by the same procedure and applied to the fifth, so every variant is
      evaluated with thresholds it did not help to set. Yields are reported per partition.
    transfer: thresholds derived on test_gene + test_db only and applied to test_time, which also changes the
      calibration population (genes without curated training variants; literature-curated labels)."""
    methods = [('HEARS-LM', tp.prob.to_numpy())] + [(c, ext[c].to_numpy()) for c in COMPARATORS if c in ext]
    common = np.all([~np.isnan(sc) for _, sc in methods], axis=0)
    t = tp[common].reset_index(drop=True)
    y = t.y.to_numpy().astype(int)
    rng = np.random.default_rng(0)
    fold = np.empty(len(t), int)
    for _, idx in t.groupby(['partition', 'y']).indices.items():
        idx = rng.permutation(idx)
        fold[idx] = np.arange(len(idx)) % k_folds
    rows, fold_rows = [], []
    for name, sc in methods:
        sc = sc[common]
        ev = np.empty(len(t), dtype=object)
        for f in range(k_folds):
            tr = fold != f
            r = M.evidence_thresholds(sc[tr], y[tr], n_boot=NBOOT // 2, seed=0)
            ev[~tr] = assign_evidence(sc[~tr], r['thresholds'])
            fold_rows.append({'method': name, 'fold': f, 'n_calibration': int(tr.sum()),
                              **{k2: v2 for k2, v2 in r['thresholds'].items() if 'very' not in k2}})
        for part in TESTS:
            m = (t.partition == part).to_numpy()
            rows.append({'design': 'kfold', 'method': name, 'partition': part, **_yields(ev[m], y[m])})
        tr = (t.partition != 'test_time').to_numpy()
        r = M.evidence_thresholds(sc[tr], y[tr], n_boot=NBOOT // 2, seed=0)
        evt = assign_evidence(sc[~tr], r['thresholds'])
        rows.append({'design': 'transfer (thresholds from test_gene + test_db)', 'method': name,
                     'partition': 'test_time', **_yields(evt, y[~tr]),
                     **{f'thr_{k2}': v2 for k2, v2 in r['thresholds'].items()}})
    cf = pd.DataFrame(rows)
    cf.to_csv(RESULTS / 'evidence_crossfit.csv', index=False)
    ft = pd.DataFrame(fold_rows)
    ft.to_csv(RESULTS / 'evidence_crossfit_fold_thresholds.csv', index=False)
    print('thresholds per cross-fitting fold (HEARS-LM):\n', ft[ft.method == 'HEARS-LM'].round(5).to_string())
    print('cross-fitted evidence yields:\n', cf.iloc[:, :9].round(3).to_string())
    return cf


def reclassification(tp, v, base, vp):
    """(a) test variants (all three test sets) that were VUS / conflicting in ClinVar 2025-01 and hold a definitive
    ClinVar classification (>= 1 star) in 2026-04, labelled by that ClinVar classification; (b) VUS in 2025-01 that
    became P/LP in ClinVar by 2026-04 against variants still VUS in 2026-04 (the latter scored as unlabelled)."""
    rows, curves = [], []
    t = tp.copy()
    vv = v.reindex(t.variant_id)
    t['prev'] = vv.prev_state_2025.to_numpy()
    t['y_cv26'] = vv.y_cv26.to_numpy()
    r = t[t.prev.isin(['VUS', 'CONFLICT']) & t.y_cv26.notna()]
    er = base.reindex(r.variant_id)
    rr, cc = roc_pr_rows('VUS/conflicting (2025-01) -> P/LP vs B/LB (2026-04)', r.y_cv26.to_numpy().astype(int),
                         [('HEARS-LM', r.prob.to_numpy())] + [(c, er[c].to_numpy()) for c in COMPARATORS if c in er])
    rows += rr; curves += cc
    print(f"reclassification (a): {len(r)} variants by partition {r.partition.value_counts().to_dict()}")
    if vp is not None:
        vt = pd.read_csv(DATA / 'vus_2026-04.csv', usecols=['variant_id', 'cv_state'])
        c25 = pd.read_csv(DATA / 'sources/clinvar_2025-01_canon.csv', usecols=['variant_id', 'status', 'canon_status'],
                          dtype=str)
        c25 = c25[c25.canon_status == 'ok']
        vus25 = set(c25[c25.status == 'VUS'].variant_id) - set(c25[c25.status.isin(['P', 'B', 'CONFLICT'])].variant_id)
        stable = vt[(vt.cv_state == 'VUS') & vt.variant_id.isin(vus25)].variant_id
        up = t[(t.prev == 'VUS') & (t.y_cv26 == 1)]
        print(f"reclassification (b): {len(up)} upgraded by partition {up.partition.value_counts().to_dict()}")
        sc = pd.concat([up[['variant_id', 'prob']].assign(y=1),
                        vp.reindex(stable)[['prob']].reset_index().rename(columns={'index': 'variant_id'}).assign(y=0)])
        sc = sc.dropna(subset=['prob'])
        vb, _ = load_baselines(DATA / 'annotations/dbnsfp_vus.csv')
        allb = pd.concat([base, vb[~vb.index.isin(base.index)]])
        comp = [(c, allb[c].reindex(sc.variant_id).to_numpy(dtype=float)) for c in COMPARATORS
                if c in allb and c != 'ESM-2 150M zero-shot']        # zero-shot scores exist for labelled variants only
        rr, cc = roc_pr_rows('VUS (2025-01) -> P/LP vs still VUS (2026-04)', sc.y.to_numpy(),
                             [('HEARS-LM', sc.prob.to_numpy())] + comp)
        rows += rr; curves += cc
    rc = pd.DataFrame(rows)
    rc = rc.merge(pd.DataFrame(PAIRED), on=['analysis', 'method'], how='left')
    rc.to_csv(RESULTS / 'reclassification.csv', index=False)
    if curves:
        pd.concat(curves).to_csv(RESULTS / 'reclassification_curves.csv', index=False)
    print('reclassification:\n', rc.round(4).to_string())
    return rc


def expert_panel(tp, base, arm):
    """ClinGen HL-VCEP classifications. P/LP vs B/LB discrimination uses only variants with an out-of-fold score
    (leave-gene-family-out cross-validation: the scoring model saw neither the gene nor its homologues), on the
    variants that every method scores. No evidence levels are assigned: the thresholds refer to the score scale of
    the selected model, not to that of the fold models."""
    ex = pd.read_csv(DATA / 'annotations/clingen_expert_v8.csv')
    ex = ex[ex.expert_panel == 'Hearing Loss VCEP'].copy()
    oof = oof_predictions(arm).set_index('variant_id')
    ex['hearslm_score'] = oof.prob.reindex(ex.variant_id).to_numpy()
    ex['score_source'] = np.where(ex.hearslm_score.notna(), 'out-of-fold (leave-family-out CV)', 'no out-of-fold score')
    bex = base.reindex(ex.variant_id)
    for c in COMPARATORS:
        if c in bex:
            ex[c] = bex[c].to_numpy()
    ex.to_csv(RESULTS / 'expert_panel_variants.csv', index=False)
    lab = ex[ex.y_expert.notna() & ex.hearslm_score.notna()]
    y = lab.y_expert.to_numpy().astype(int)
    scores = [('HEARS-LM', lab.hearslm_score.to_numpy())] + [(c, lab[c].to_numpy()) for c in COMPARATORS if c in lab]
    k = np.all([~np.isnan(s) for _, s in scores], axis=0)
    rows = []
    for name, s in scores:
        a = M.bootstrap_ci(y[k], s[k], M.auroc, NBOOT)
        r = {'method': name, 'n': int(k.sum()), 'n_P': int(y[k].sum()), 'auroc': a['est'], 'auroc_lo': a['lo'],
             'auroc_hi': a['hi']}
        if name != 'HEARS-LM':
            pb = M.paired_bootstrap_diff(y[k], scores[0][1][k], s[k], M.auroc, NBOOT)
            r.update({'diff_hearslm_minus_method': pb['diff'], 'diff_lo': pb['lo'], 'diff_hi': pb['hi'],
                      'p_delong': M.delong_test(y[k], scores[0][1][k], s[k])['p']})
        rows.append(r)
    ep = pd.DataFrame(rows)
    ep.to_csv(RESULTS / 'expert_panel.csv', index=False)
    print(f'expert panel: {len(ex)} HL-VCEP missense variants, {int(ex.y_expert.notna().sum())} P/LP/B/LB, '
          f'{len(lab)} of these with an out-of-fold score, {int(k.sum())} scored by every method')
    print(ep.round(4).to_string())
    return ep


def thresholds_by_population(tp, ext):
    """Sensitivity of the HEARS-LM thresholds to the calibration population (same procedure, same variant
    restriction as the main calibration): pooled test sets (reported), temporal set only, withheld-gene +
    literature-evidence sets only."""
    methods = [('HEARS-LM', tp.prob.to_numpy())] + [(c, ext[c].to_numpy()) for c in COMPARATORS if c in ext]
    common = np.all([~np.isnan(sc) for _, sc in methods], axis=0)
    t = tp[common]
    rows = []
    for label, m in [('pooled test sets (reported)', np.ones(len(t), bool)),
                     ('temporal test set only', (t.partition == 'test_time').to_numpy()),
                     ('withheld-gene and literature-evidence sets only', (t.partition != 'test_time').to_numpy())]:
        r = M.evidence_thresholds(t.prob.to_numpy()[m], t.y.to_numpy()[m], n_boot=NBOOT, seed=0)
        rows.append({'calibration_population': label, 'n': int(m.sum()), 'n_P': int(t.y.to_numpy()[m].sum()),
                     'n_B': int((t.y.to_numpy()[m] == 0).sum()),
                     **{k: v for k, v in r['thresholds'].items() if 'very' not in k}})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / 'calibration_thresholds_by_population.csv', index=False)
    print('HEARS-LM thresholds by calibration population:\n', out.round(5).to_string())
    return out


def catalogue(vp, thr, v):
    """Evidence level for every ClinVar 2026-04 VUS / conflicting missense variant of the panel genes. Evidence is
    assigned only in genes that contributed training variants to the scoring model; for the other genes (the ten
    withheld genes and genes without any training label) the score is reported without an evidence level."""
    n_train = v[v.partition == 'fit'].groupby('gene').size()
    pd.DataFrame({'gene': n_train.index, 'n_training_variants': n_train.to_numpy()}) \
        .to_csv(RESULTS / 'training_variants_per_gene.csv', index=False)
    vt = pd.read_csv(DATA / 'vus_2026-04.csv')
    n_other = int((vt.cv_state == 'OTHER').sum())
    vt = vt[vt.cv_state.isin(['VUS', 'CONFLICT'])].copy()          # uncertain or conflicting significance only
    vt = vt.merge(vp[['prob']], left_on='variant_id', right_index=True, how='left')
    vt['n_training_variants_gene'] = vt.gene.map(n_train).fillna(0).astype(int)
    vt['evidence'] = assign_evidence(vt.prob.fillna(0.5).to_numpy(), thr)
    vt.loc[vt.prob.isna(), 'evidence'] = 'not scored'
    vt.loc[vt.n_training_variants_gene == 0, 'evidence'] = NOT_ASSIGNED
    vt['residue_context'] = np.where(vt.same_residue_PLP == 1, 'P/LP missense at same residue',
                                     np.where(vt.dist_nearest_PLP <= 5, 'within 5 residues of a P/LP residue',
                                              'no P/LP missense within 5 residues'))
    vt.sort_values('prob', ascending=False).to_csv(RESULTS / 'vus_catalogue.csv', index=False)
    summ = vt.groupby(['cv_state', 'evidence', 'residue_context']).size().reset_index(name='n')
    summ.to_csv(RESULTS / 'vus_catalogue_summary.csv', index=False)
    ins = vt[vt.n_training_variants_gene > 0]
    print(f'VUS catalogue: {len(vt)} VUS/conflicting variants ({n_other} other unresolved records excluded); '
          f'{len(ins)} in {ins.gene.nunique()} genes with training variants, {len(vt) - len(ins)} in '
          f'{vt[vt.n_training_variants_gene == 0].gene.nunique()} genes without (no evidence level)')
    print(pd.crosstab(vt.evidence, vt.residue_context).to_string())
    return vt


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*', choices=['crossfit', 'reclass', 'expert', 'catalogue', 'populations'],
                    help='run only these sections (they reuse the stored thresholds of the full run)')
    a = ap.parse_args()
    arm = final_arm()
    v = load_variants().set_index('variant_id')
    base, _ = load_baselines()
    tp = test_predictions(arm)
    ext = base.reindex(tp.variant_id)
    vus_pred_file = RESULTS / f'vus_predictions_{arm}.csv'
    vp = pd.read_csv(vus_pred_file).set_index('variant_id') if vus_pred_file.exists() else None
    vv = load_variants()
    if a.only:
        cal = pd.read_csv(RESULTS / 'calibration_thresholds.csv')
        thr = {r.evidence: r.threshold for r in cal[cal.method == 'HEARS-LM'].itertuples()
               if r.evidence != 'ANY_DISCORDANT'}
        if 'crossfit' in a.only:
            crossfit(tp, ext)
        if 'populations' in a.only:
            thresholds_by_population(tp, ext)
        if 'expert' in a.only:
            expert_panel(tp, base, arm)
        if 'reclass' in a.only:
            reclassification(tp, v, base, vp)
        if 'catalogue' in a.only and vp is not None:
            catalogue(vp, thr, vv)
        return

    # 1 + 2
    cal, curves, assign = calibrate(tp, ext)
    cal.to_csv(RESULTS / 'calibration_thresholds.csv', index=False)
    curves.to_csv(RESULTS / 'calibration_curves.csv', index=False)
    assign.to_csv(RESULTS / 'evidence_assignment.csv', index=False)
    rel, bins = reliability(tp)
    rel.to_csv(RESULTS / 'reliability.csv', index=False)
    bins.to_csv(RESULTS / 'reliability_bins.csv', index=False)
    hl = cal[cal.method == 'HEARS-LM'][['evidence', 'threshold', 'n_P_assigned', 'n_B_assigned', 'frac_P_assigned',
                                         'frac_B_assigned']]
    print('calibration (HEARS-LM):\n', hl.round(4).to_string())
    print(rel.round(4).to_string())
    thr = {r.evidence: r.threshold for r in cal[cal.method == 'HEARS-LM'].itertuples() if r.evidence != 'ANY_DISCORDANT'}
    crossfit(tp, ext)
    thresholds_by_population(tp, ext)
    expert_panel(tp, base, arm)
    reclassification(tp, v, base, vp)
    if vp is not None:
        catalogue(vp, thr, vv)


if __name__ == '__main__':
    main()
