#!/usr/bin/env python3
"""
Statistics used throughout the revision. All functions take plain numpy arrays.

Conventions
  * scores are oriented so that HIGHER = more likely pathogenic before they reach these functions;
  * y is 1 (pathogenic / likely pathogenic) or 0 (benign / likely benign);
  * confidence intervals: percentile bootstrap, stratified by class (cases and controls resampled separately);
  * paired comparisons of two predictors are always made on the SAME variants;
  * multiple testing: Benjamini-Hochberg.
"""
import numpy as np
from scipy.stats import norm
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


# ------------------------------------------------------------------------------------- basic metrics
def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, s))


def auprc(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(average_precision_score(y, s))


def within_group_auroc(y, s, groups):
    """Pair-weighted AUROC restricted to pathogenic-benign pairs from the same group (gene).
    Removes any contribution of between-gene differences in label prevalence. Ties count 0.5."""
    y = np.asarray(y); s = np.asarray(s, dtype=float); groups = np.asarray(groups)
    conc, pairs = 0.0, 0.0
    for g in np.unique(groups):
        m = groups == g
        yp, sp = y[m], s[m]
        npos, nneg = int((yp == 1).sum()), int((yp == 0).sum())
        if npos == 0 or nneg == 0:
            continue
        a = roc_auc_score(yp, sp)
        conc += a * npos * nneg
        pairs += npos * nneg
    return float(conc / pairs) if pairs else np.nan


def ece(y, p, n_bins=10):
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            e += abs(y[m].mean() - p[m].mean()) * m.mean()
    return float(e)


def brier(y, p):
    return float(brier_score_loss(np.asarray(y), np.asarray(p, dtype=float)))


# ------------------------------------------------------------------------------------------ bootstrap
def _strat_indices(y, rng):
    pos = np.flatnonzero(y == 1); neg = np.flatnonzero(y == 0)
    return np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])


def bootstrap_ci(y, s, metric=auroc, n_boot=2000, seed=0, alpha=0.05):
    """Point estimate and percentile CI of metric(y, s)."""
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    rng = np.random.default_rng(seed)
    est = metric(y, s)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        i = _strat_indices(y, rng)
        vals[b] = metric(y[i], s[i])
    lo, hi = np.nanpercentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {'est': float(est), 'lo': float(lo), 'hi': float(hi)}


def paired_bootstrap_diff(y, s1, s2, metric=auroc, n_boot=2000, seed=0, alpha=0.05):
    """metric(s1) - metric(s2) on identical variants: estimate, percentile CI, two-sided bootstrap p."""
    y = np.asarray(y); s1 = np.asarray(s1, dtype=float); s2 = np.asarray(s2, dtype=float)
    rng = np.random.default_rng(seed)
    est = metric(y, s1) - metric(y, s2)
    d = np.empty(n_boot)
    for b in range(n_boot):
        i = _strat_indices(y, rng)
        d[b] = metric(y[i], s1[i]) - metric(y[i], s2[i])
    lo, hi = np.nanpercentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    p = float(min(1.0, max(p, 1.0 / n_boot)))
    return {'diff': float(est), 'lo': float(lo), 'hi': float(hi), 'p_boot': p}


def cluster_bootstrap_ci(y, s, clusters, metric=auroc, n_boot=2000, seed=0, alpha=0.05):
    """Percentile CI resampling whole clusters (genes) with replacement."""
    y = np.asarray(y); s = np.asarray(s, dtype=float); clusters = np.asarray(clusters)
    rng = np.random.default_rng(seed)
    uniq = np.unique(clusters)
    members = {c: np.flatnonzero(clusters == c) for c in uniq}
    vals = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(uniq, len(uniq))
        i = np.concatenate([members[c] for c in pick])
        vals[b] = metric(y[i], s[i])
    lo, hi = np.nanpercentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {'est': float(metric(y, s)), 'lo': float(lo), 'hi': float(hi)}


# --------------------------------------------------------------------------------------------- DeLong
def _midrank(x):
    order = np.argsort(x, kind='mergesort')
    z = x[order]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = t
    return out


def delong_test(y, s1, s2):
    """Two-sided DeLong test for the difference of two correlated AUROCs (same variants).
    Returns auc1, auc2, z, p."""
    y = np.asarray(y); s = np.vstack([np.asarray(s1, dtype=float), np.asarray(s2, dtype=float)])
    order = np.argsort(-y, kind='mergesort')
    m = int((y == 1).sum()); n = len(y) - m
    if m < 2 or n < 2:
        return {'auc1': np.nan, 'auc2': np.nan, 'z': np.nan, 'p': np.nan}
    s = s[:, order]
    tx = np.vstack([_midrank(s[r, :m]) for r in range(2)])
    ty = np.vstack([_midrank(s[r, m:]) for r in range(2)])
    tz = np.vstack([_midrank(s[r]) for r in range(2)])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        z = 0.0 if aucs[0] == aucs[1] else np.inf * np.sign(aucs[0] - aucs[1])
    else:
        z = (aucs[0] - aucs[1]) / np.sqrt(var)
    p = float(2 * norm.sf(abs(z)))
    return {'auc1': float(aucs[0]), 'auc2': float(aucs[1]), 'z': float(z), 'p': p}


# ------------------------------------------------------------------------------------------------ FDR
def bh_fdr(pvals):
    """Benjamini-Hochberg adjusted p-values (q-values); NaNs are preserved."""
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    n = len(pv)
    if n == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(ranked, 1.0)
    q[ok] = out
    return q


# ------------------------------------------------------------ ACMG/AMP evidence-strength calibration
PRIOR = 0.0441                       # prior probability of pathogenicity used by Pejaver et al. 2022
ODDS_VERY_STRONG = 1124.0            # odds of pathogenicity for Very Strong at that prior
STRENGTH_EXP = {'supporting': 1 / 8, 'moderate': 1 / 4, 'strong': 1 / 2, 'very_strong': 1.0}


def posterior_thresholds(prior=PRIOR, c=ODDS_VERY_STRONG):
    """Posterior probabilities that correspond to each evidence strength (Tavtigian framework)."""
    out = {}
    for name, ex in STRENGTH_EXP.items():
        lr = c ** ex
        out[f'PP3_{name}'] = lr * prior / (lr * prior + 1 - prior)
        out[f'BP4_{name}'] = (1 / lr) * prior / ((1 / lr) * prior + 1 - prior)
    return out


def local_posterior(scores, y, grid=None, prior=PRIOR, min_n=100, pseudo_b=0.0, pseudo_p=0.0):
    """Local posterior probability of pathogenicity as a function of the score (Pejaver et al. 2022):
    for each grid score g take the smallest symmetric window [g-e, g+e] holding >= min_n labelled variants
    and compute the prior-corrected fraction of pathogenic variants inside it.
    Pathogenic variants are re-weighted so that the weighted class ratio equals prior/(1-prior).
    pseudo_b / pseudo_p add that many benign / pathogenic pseudo-observations to every window; they are
    used to make the confidence bounds conservative where a window contains a single class (our
    substitute for the gnomAD-coverage requirement of the original method)."""
    s = np.asarray(scores, dtype=float); y = np.asarray(y)
    order = np.argsort(s)
    s, y = s[order], y[order]
    if grid is None:
        grid = np.unique(s)
    n_p, n_b = (y == 1).sum(), (y == 0).sum()
    w = prior * n_b / ((1 - prior) * n_p)            # weight of one pathogenic variant
    cp = np.concatenate([[0], np.cumsum(y == 1)])
    cb = np.concatenate([[0], np.cumsum(y == 0)])
    grid = np.asarray(grid, dtype=float)
    k = min(min_n, len(s))
    # the k scores nearest to g are contiguous in sorted order and lie within k positions of g's insertion point,
    # so the smallest half-width holding >= k variants is the k-th smallest distance within that band
    pos = np.searchsorted(s, grid)
    band = pos[:, None] + np.arange(-k, k)[None, :]
    valid = (band >= 0) & (band < len(s))
    d = np.where(valid, np.abs(s[np.clip(band, 0, len(s) - 1)] - grid[:, None]), np.inf)
    e = np.partition(d, k - 1, axis=1)[:, k - 1]
    lo = np.searchsorted(s, grid - e, side='left')
    hi = np.searchsorted(s, grid + e, side='right')
    pp, bb = cp[hi] - cp[lo] + pseudo_p, cb[hi] - cb[lo] + pseudo_b
    with np.errstate(invalid='ignore', divide='ignore'):
        post = np.where(w * pp + bb > 0, (w * pp) / (w * pp + bb), np.nan)
    return grid, post


def evidence_thresholds(scores, y, prior=PRIOR, min_n=100, n_boot=1000, seed=0, n_grid=400):
    """Score thresholds for PP3 / BP4 evidence strengths.
    A PP3 threshold is the smallest score t such that the one-sided 95% lower bootstrap bound of the local
    posterior is >= the required posterior for every grid score >= t (and symmetrically for BP4 with the
    upper bound and scores <= t). The lower bound is computed with one benign pseudo-observation per
    window and the upper bound with one pathogenic pseudo-observation (conservative). Returns thresholds (NaN when a strength is not reached), the grid,
    the point-estimate posterior and the bootstrap bounds."""
    s = np.asarray(scores, dtype=float); y = np.asarray(y)
    rng = np.random.default_rng(seed)
    grid = np.quantile(s, np.linspace(0, 1, n_grid))
    grid = np.unique(grid)
    _, post = local_posterior(s, y, grid, prior, min_n)
    lows = np.empty((n_boot, len(grid)))
    ups = np.empty((n_boot, len(grid)))
    for b in range(n_boot):
        i = _strat_indices(y, rng)
        _, lows[b] = local_posterior(s[i], y[i], grid, prior, min_n, pseudo_b=1.0)
        _, ups[b] = local_posterior(s[i], y[i], grid, prior, min_n, pseudo_p=1.0)
    lower = np.nanpercentile(lows, 5, axis=0)
    upper = np.nanpercentile(ups, 95, axis=0)
    req = posterior_thresholds(prior)
    thr = {}
    for name in STRENGTH_EXP:
        ok = lower >= req[f'PP3_{name}']
        # smallest t such that all grid points >= t satisfy the bound
        suffix_ok = np.logical_and.accumulate(ok[::-1])[::-1]
        thr[f'PP3_{name}'] = float(grid[np.argmax(suffix_ok)]) if suffix_ok.any() else np.nan
        okb = upper <= req[f'BP4_{name}']
        prefix_ok = np.logical_and.accumulate(okb)
        thr[f'BP4_{name}'] = float(grid[np.flatnonzero(prefix_ok)[-1]]) if prefix_ok.any() else np.nan
    return {'thresholds': thr, 'required_posterior': req, 'grid': grid, 'posterior': post,
            'lower95': lower, 'upper95': upper, 'prior': prior, 'min_n': min_n}
