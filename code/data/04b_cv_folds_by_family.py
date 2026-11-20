#!/usr/bin/env python3
"""
Step 04b - Homology-aware folds for the leave-gene-out cross-validation.

A gene-grouped split can still leak through paralogues (KCNQ1 <-> KCNQ4, collagens, myosins, ...).
Genes are therefore grouped into FAMILIES and whole families are assigned to one fold.

Family definition (union of two relations, single linkage):
  (a) sequence similarity between reference proteins: local alignment (BLOSUM62, gap open -11, extend -1)
      with >= 30 % identity over >= 100 aligned residue pairs that also cover >= 50 % of the shorter protein
      (a looser criterion - any 80-residue stretch at 30 % identity - chains almost all genes into one
      cluster through low-complexity and repeat regions and was therefore rejected);
  (b) the curated family table of the original study (UniProt family annotation + manual curation of
      functional complexes), `hearing/data/external/gene_paralog_families.csv` ("singleton" = no family).

Outputs
  data/gene_families_v8.csv   gene, family_id, family_genes, n_labelled, cv_fold, has_paralog_in_fit (for the
                              10 hold-out genes of the fixed split: whether a family member is a training gene)
  data/variants_v8.csv        column cv_fold is overwritten with the family-grouped folds
  data/cv_folds_summary.json
"""
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import Align
from Bio.Align import substitution_matrices

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'data'))
from paths import legacy_root  # noqa: E402

REV_ROOT = Path(__file__).resolve().parents[2]                   # the release folder
REV = REV_ROOT
MIN_IDENTITY, MIN_COVERAGE, MIN_PAIRS, K = 0.30, 0.50, 100, 5


def main():
    ref = pd.read_csv(REV / 'data/sources/reference_proteins.csv')
    seqs = dict(zip(ref.gene, ref.sequence))
    genes = sorted(seqs)
    al = Align.PairwiseAligner()
    al.mode = 'local'
    al.substitution_matrix = substitution_matrices.load('BLOSUM62')
    al.open_gap_score, al.extend_gap_score = -11, -1

    parent = {g: g for g in genes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    links = []
    cache = REV / 'data/gene_pairwise_alignment_v8.csv'
    if cache.exists():
        pw = pd.read_csv(cache)
    else:
        prow = []
        for a, b in combinations(genes, 2):
            sa, sb = seqs[a], seqs[b]
            aln = al.align(sa, sb)[0]
            n_pairs = n_id = 0
            for (a0, a1), (b0, b1) in zip(*aln.aligned):
                for k in range(a1 - a0):
                    n_pairs += 1
                    n_id += sa[a0 + k] == sb[b0 + k]
            prow.append({'gene_a': a, 'gene_b': b, 'len_a': len(sa), 'len_b': len(sb), 'aligned_pairs': n_pairs,
                         'identity': round(n_id / max(1, n_pairs), 4), 'score': float(aln.score)})
        pw = pd.DataFrame(prow)
        pw.to_csv(cache, index=False)
    pw['coverage_shorter'] = pw.aligned_pairs / pw[['len_a', 'len_b']].min(axis=1)
    for r in pw[(pw.identity >= MIN_IDENTITY) & (pw.coverage_shorter >= MIN_COVERAGE) & (pw.aligned_pairs >= MIN_PAIRS)].itertuples():
        links.append({'gene_a': r.gene_a, 'gene_b': r.gene_b, 'aligned_pairs': r.aligned_pairs,
                      'identity': r.identity, 'relation': 'sequence'})
        union(r.gene_a, r.gene_b)
    cur = pd.read_csv(legacy_root() / 'data/external/gene_paralog_families.csv')
    for fam, grp in cur[cur.family != 'singleton'].groupby('family'):
        gl = [g for g in grp.gene if g in parent]
        for g in gl[1:]:
            links.append({'gene_a': gl[0], 'gene_b': g, 'aligned_pairs': np.nan, 'identity': np.nan,
                          'relation': f'curated:{fam}'})
            union(gl[0], g)
    pd.DataFrame(links).to_csv(REV / 'data/gene_family_links_v8.csv', index=False)

    fam_of = {g: find(g) for g in genes}
    v = pd.read_csv(REV / 'data/variants_v8.csv', low_memory=False)
    pool = v[v.y_final.notna()]
    n_lab = pool.groupby('gene').size()
    fam_size = pd.Series({f: int(sum(n_lab.get(g, 0) for g in genes if fam_of[g] == f)) for f in set(fam_of.values())})
    # greedy balanced assignment of families to folds (largest first; deterministic)
    load = np.zeros(K)
    fold_of_fam = {}
    for f in fam_size.sort_values(ascending=False, kind='mergesort').index:
        k = int(np.argmin(load))
        fold_of_fam[f] = k
        load[k] += fam_size[f]
    v['cv_fold'] = np.where(v.y_final.notna(), v.gene.map(lambda g: fold_of_fam[fam_of[g]]), -1).astype(int)
    v.to_csv(REV / 'data/variants_v8.csv', index=False)

    fit_genes = set(v[v.partition == 'fit'].gene)
    members = {f: sorted(g for g in genes if fam_of[g] == f) for f in set(fam_of.values())}
    rows = [{'gene': g, 'family_id': fam_of[g], 'family_size': len(members[fam_of[g]]),
             'family_genes': '|'.join(members[fam_of[g]]), 'n_labelled': int(n_lab.get(g, 0)),
             'cv_fold': fold_of_fam[fam_of[g]],
             'has_family_member_in_fit': any((m != g) and (m in fit_genes) for m in members[fam_of[g]])} for g in genes]
    gf = pd.DataFrame(rows)
    gf.to_csv(REV / 'data/gene_families_v8.csv', index=False)

    cvd = v[v.cv_fold >= 0]
    summ = {'n_genes': len(genes), 'n_families': len(members),
            'n_multi_gene_families': int(sum(len(m) > 1 for m in members.values())),
            'multi_gene_families': {f: m for f, m in sorted(members.items(), key=lambda kv: -len(kv[1])) if len(m) > 1},
            'n_sequence_links': int(sum(1 for x in links if x['relation'] == 'sequence')),
            'folds': {int(k): {'n': int(len(g)), 'P': int((g.y_final == 1).sum()), 'B': int((g.y_final == 0).sum()),
                               'genes': int(g.gene.nunique())} for k, g in cvd.groupby('cv_fold')},
            'holdout_genes_with_family_member_in_fit': gf[gf.gene.isin(
                ['OTOF', 'CLDN14', 'STRC', 'TECTA', 'WFS1', 'TMC1', 'CDH23', 'LOXHD1', 'TBC1D24', 'ACTG1'])
                & gf.has_family_member_in_fit][['gene', 'family_genes']].to_dict('records'),
            'sequence_links': [f"{x['gene_a']}-{x['gene_b']} ({x['identity']:.2f})" for x in links if x['relation'] == 'sequence'],
            'criteria': {'min_identity': MIN_IDENTITY, 'min_coverage_of_shorter_protein': MIN_COVERAGE, 'min_aligned_pairs': MIN_PAIRS}}
    json.dump(summ, open(REV / 'data/cv_folds_summary.json', 'w'), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
