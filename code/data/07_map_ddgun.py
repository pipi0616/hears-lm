#!/usr/bin/env python3
"""
Step 07 - Map the DDGun folding free-energy predictions of Tollefson et al. 2023 (Table S4, all missense
variants of the Deafness Variation Database) onto the reference-protein numbering of this project.

Inputs
  hearing/data/external/tollefson2023_otoprotein_ddg.xlsx   one sheet; row 1 = title, row 2 = header
        Gene | HGVS Protein Change ("NP_056415.1:p.Ser401Leu") | Variant Classification | Surface Area |
        Normalized SA | Confidence Score (AlphaFold2 pLDDT) | ddG_fold | |ddG_fold|   (GRCh38 column is corrupt, unused)
  data/sources/reference_proteins.csv       gene, reference_protein_id, sequence
  data/sources/transcript_proteins.csv      RefSeq proteins already fetched in step 02 (read only)
  data/sources/transcript_to_reference.csv  accession -> gene (used only to detect renamed gene symbols)
  data/variants_v8.csv                      labelled variants (coverage and sign checks only)
  code/data/02_fetch_transcript_proteins.py efetch(), parse_fasta()            (imported)
  code/data/03_canonicalize.py              make_aligner(), position_map(), AA3TO1 (imported)
  code/data/06_structure_features.py        identical_run_lengths(), MIN_RUN   (imported)

Outputs
  data/sources/transcript_proteins_extra.csv   NP sequences fetched here (accession, kind, protein_id, length,
                                               sequence); cached - a re-run only fetches what is still missing
  data/annotations/ddgun_v8.csv                one row per mapped protein variant:
        variant_id, gene, ref_pos, ref1, alt1, ddg (signed value as given), abs_ddg, af2_confidence, sasa,
        norm_sasa, tollefson_class, np_accession, src_pos
  data/annotations/ddgun_v8_summary.json       row accounting, duplicates, coverage of labelled variants, sign checks

Rules
  1. Gene: the table's gene symbol must be one of the reference genes. Three symbols were renamed by HGNC
     after the panel was frozen (GENE_ALIASES: GSDME=DFNA5, PJVK=DFNB59, KARS1=KARS); they are mapped to the
     panel symbol and the protein is checked by sequence. All other symbols are counted and left out.
  2. Source protein: the NP accession (with version) of "HGVS Protein Change"; sequence from
     transcript_proteins.csv, else fetched from NCBI (efetch db=protein, as in step 02).
  3. Mapping onto the reference protein: global alignment source -> reference (position_map of step 03) and the
     three acceptance rules of step 03:
        (i)   the reported reference residue matches the source protein at the reported position,
        (ii)  that position aligns to a residue of the reference protein,
        (iii) the aligned reference residue is identical to the reported reference residue;
     plus, when REQUIRE_CONTEXT is True (default),
        (iv)  the position lies in an ungapped run of >= 11 residues identical in source and reference protein.
     Rule (iv) rejects single-residue chance identities inside isoform-specific segments that the global
     alignment pairs up; such a row would otherwise put a ddG on an unrelated residue of the reference protein
     and could win the duplicate rule below. A source protein that shares no such run with the reference
     protein at all is a different protein (status source_protein_unrelated; Table S4 lists PSMC2 accessions
     under SLC26A5). Set REQUIRE_CONTEXT = False for the plain step-03 behaviour.
     Status per row: ok | gene_not_in_reference_set | unparsed_protein_change | no_transcript |
     source_protein_unrelated | ref_mismatch_source (i) | not_in_reference_isoform (ii) |
     ref_mismatch_reference (iii) | unverified_alignment_context (iv).
  4. variant_id = GENE:<ref1><ref_pos><alt1>. If a variant_id occurs more than once (several nucleotide changes
     or several source isoforms) the first row in table order is kept; duplicates and their agreement are counted.
  5. Values are copied unchanged (no sign flip, no imputation). The sign convention is examined, not assumed.
"""
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'data'))
from paths import legacy_root  # noqa: E402

sys.dont_write_bytecode = True      # importing the sibling step scripts must not leave __pycache__ files behind

REV_ROOT = Path(__file__).resolve().parents[2]                   # the release folder
REV = REV_ROOT
XLSX = legacy_root() / 'data/external/tollefson2023_otoprotein_ddg.xlsx'
SRC = REV / 'data/sources'
EXTRA = SRC / 'transcript_proteins_extra.csv'
OUT_DIR = REV / 'data/annotations'
OUT = OUT_DIR / 'ddgun_v8.csv'
OUT_SUMMARY = OUT_DIR / 'ddgun_v8_summary.json'

REQUIRE_CONTEXT = True
GENE_ALIASES = {'GSDME': 'DFNA5', 'PJVK': 'DFNB59', 'KARS1': 'KARS'}   # symbol in the table -> symbol of the panel
HGVS_P = re.compile(r'^(NP_\d+\.\d+):p\.([A-Za-z]{3})(\d+)([A-Za-z]{3})$')
HYDROPHOBIC, CHARGED = set('AVILMFWC'), set('DEKR')


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # the step scripts only define functions/constants at import time
    return mod


fetch02 = _load('fetch02', '02_fetch_transcript_proteins.py')
canon = _load('canonicalize03', '03_canonicalize.py')
struct06 = _load('structure06', '06_structure_features.py')
MIN_RUN = struct06.MIN_RUN


def read_table():
    raw = pd.read_excel(XLSX, sheet_name=0, header=1)
    raw.columns = [str(c).strip() for c in raw.columns]
    g_cols = [c for c in raw.columns if c.endswith('G_fold') or c.endswith('G_fold|')]
    signed = [c for c in g_cols if '|' not in c]
    absolute = [c for c in g_cols if '|' in c]
    assert len(signed) == 1 and len(absolute) == 1, raw.columns.tolist()
    t = pd.DataFrame({'tollefson_gene': raw['Gene'].astype(str).str.strip(),
                      'hgvs_p': raw['HGVS Protein Change'].astype(str).str.strip(),
                      'tollefson_class': raw['Variant Classification'],
                      'ddg': pd.to_numeric(raw[signed[0]]), 'abs_ddg': pd.to_numeric(raw[absolute[0]]),
                      'af2_confidence': pd.to_numeric(raw['Confidence Score']),
                      'sasa': pd.to_numeric(raw['Surface Area']), 'norm_sasa': pd.to_numeric(raw['Normalized SA'])})
    p = t.hgvs_p.str.extract(HGVS_P)
    t['np_accession'], t['src_pos'] = p[0], pd.to_numeric(p[2]).astype('Int64')
    t['ref1'], t['alt1'] = p[1].map(canon.AA3TO1), p[3].map(canon.AA3TO1)
    t['parsed'] = t.np_accession.notna() & t.ref1.notna() & t.alt1.notna() & (t.ref1 != t.alt1)   # a missense change
    return t


def protein_sequences(needed):
    """{NP accession.version: sequence} from step 02 output, the local extra cache and, if needed, NCBI."""
    tx = pd.read_csv(SRC / 'transcript_proteins.csv', dtype=str)
    seqs = {}
    for acc, seq in list(zip(tx.accession, tx.sequence)) + list(zip(tx.protein_id, tx.sequence)):
        assert seqs.setdefault(acc, seq) == seq, f'conflicting sequences for {acc} in transcript_proteins.csv'
    extra = pd.read_csv(EXTRA, dtype=str) if EXTRA.exists() else pd.DataFrame(
        columns=['accession', 'kind', 'protein_id', 'length', 'sequence'])
    rows = extra.to_dict('records')
    for acc, seq in zip(extra.accession, extra.sequence):
        assert seqs.setdefault(acc, seq) == seq, f'conflicting sequences for {acc}'
    not_in_step02 = sorted(set(needed) - set(tx.accession) - set(tx.protein_id))
    todo = sorted(set(needed) - set(seqs))
    for i in range(0, len(todo), 40):
        batch = todo[i:i + 40]
        for header, seq in fetch02.parse_fasta(fetch02.efetch('protein', batch, 'fasta')):
            acc = header.split()[0]
            if acc in batch and acc not in seqs:
                rows.append({'accession': acc, 'kind': 'NP', 'protein_id': acc, 'length': len(seq), 'sequence': seq})
                seqs[acc] = seq
        print(f'  fetched NP {i + len(batch)}/{len(todo)}', flush=True)
        time.sleep(0.4)
    if todo:
        SRC.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).drop_duplicates('accession').sort_values('accession').to_csv(EXTRA, index=False)
    return seqs, not_in_step02, todo, sorted(set(needed) - set(seqs))


def detect_renamed_symbols(table, refgenes):
    """Symbols absent from the panel whose NP accession is nevertheless a protein of a panel gene."""
    t2r = pd.read_csv(SRC / 'transcript_to_reference.csv', dtype=str)
    tx = pd.read_csv(SRC / 'transcript_proteins.csv', dtype=str)
    ref = pd.read_csv(SRC / 'reference_proteins.csv', usecols=['gene', 'reference_protein_id'])
    pid = dict(zip(tx.accession, tx.protein_id))
    owner = dict(zip(ref.reference_protein_id, ref.gene))
    for acc, gene in zip(t2r.accession, t2r.gene):
        owner.setdefault(acc, gene)
        owner.setdefault(pid.get(acc, acc), gene)
    pairs = table[~table.tollefson_gene.isin(refgenes)][['tollefson_gene', 'np_accession']].drop_duplicates()
    return {g: owner[a] for g, a in zip(pairs.tollefson_gene, pairs.np_accession) if a in owner}


def map_rows(t, refseq, seqs):
    """Apply acceptance rules (i)-(iv); returns status and ref_pos per row and a per (NP, gene) alignment summary."""
    aligner = canon.make_aligner()
    status = np.full(len(t), '', dtype=object)
    ref_pos = np.zeros(len(t), dtype=int)
    pair_rows = []
    in_set, parsed = t.gene.notna().to_numpy(), t.parsed.to_numpy()
    status[~in_set] = 'gene_not_in_reference_set'
    status[in_set & ~parsed] = 'unparsed_protein_change'
    todo = t[in_set & parsed]
    for (acc, gene), g in todo.groupby(['np_accession', 'gene'], sort=True):
        src, ref = seqs.get(acc), refseq[gene]
        idx = g.index.to_numpy()
        if src is None:
            status[idx] = 'no_transcript'
            continue
        pmap = canon.position_map(src, ref, aligner)
        runs = struct06.identical_run_lengths(src, ref, pmap)
        n_ver = sum(1 for n in runs.values() if n >= MIN_RUN)
        pair_rows.append({'np_accession': acc, 'gene': gene, 'n_rows': len(g), 'len_source': len(src),
                          'len_reference': len(ref), 'identical_sequence': pmap is None,
                          'n_identical_aligned': len(runs), 'n_context_verified': n_ver,
                          'n_positions_renumbered': 0 if pmap is None else sum(1 for s, d in pmap.items() if s != d)})
        if REQUIRE_CONTEXT and n_ver == 0:
            status[idx] = 'source_protein_unrelated'
            continue
        for i, pos, aa in zip(idx, g.src_pos.to_numpy(), g.ref1.to_numpy()):
            pos = int(pos)
            if pos > len(src) or src[pos - 1] != aa:
                status[i] = 'ref_mismatch_source'
                continue
            rpos = pos if pmap is None else pmap.get(pos)
            if rpos is None:
                status[i] = 'not_in_reference_isoform'
            elif rpos > len(ref) or ref[rpos - 1] != aa:
                status[i] = 'ref_mismatch_reference'
            elif REQUIRE_CONTEXT and runs.get(pos, 0) < MIN_RUN:
                status[i] = 'unverified_alignment_context'
            else:
                status[i], ref_pos[i] = 'ok', rpos
    return status, ref_pos, pd.DataFrame(pair_rows)


def sign_checks(d, label):
    """Descriptive statistics that reveal which sign of ddg means destabilising."""
    conf = d[d.af2_confidence >= 70]
    core = conf[(conf.norm_sasa < 0.2) & conf.ref1.isin(HYDROPHOBIC) & conf.alt1.isin(CHARGED)]
    return {'set': label, 'n': int(len(d)), 'mean_ddg': round(float(d.ddg.mean()), 4),
            'median_ddg': round(float(d.ddg.median()), 4),
            'frac_positive': round(float((d.ddg > 0).mean()), 4), 'frac_negative': round(float((d.ddg < 0).mean()), 4),
            'spearman_ddg_vs_norm_sasa': round(float(spearmanr(d.ddg, d.norm_sasa).statistic), 4),
            'spearman_ddg_vs_norm_sasa_plddt70': round(float(spearmanr(conf.ddg, conf.norm_sasa).statistic), 4),
            'mean_ddg_buried_lt0.2': round(float(d.ddg[d.norm_sasa < 0.2].mean()), 4),
            'mean_ddg_exposed_gt0.5': round(float(d.ddg[d.norm_sasa > 0.5].mean()), 4),
            'n_buried_hydrophobic_to_charged_plddt70': int(len(core)),
            'mean_ddg_buried_hydrophobic_to_charged_plddt70': round(float(core.ddg.mean()), 4) if len(core) else None}


def main():
    ref = pd.read_csv(SRC / 'reference_proteins.csv')
    refseq = dict(zip(ref.gene, ref.sequence))
    t = read_table()
    n_in = len(t)
    print(f'{n_in} rows, {t.tollefson_gene.nunique()} gene symbols, {t.np_accession.nunique()} NP accessions; '
          f'unparsed protein changes: {int((~t.parsed).sum())}', flush=True)

    # ---- gene symbols
    renamed = detect_renamed_symbols(t, set(refseq))
    unlisted = {g: r for g, r in renamed.items() if GENE_ALIASES.get(g) != r}
    assert not unlisted, f'symbols that belong to panel genes by accession but are not in GENE_ALIASES: {unlisted}'
    assert all(r in refseq and g not in refseq for g, r in GENE_ALIASES.items())
    t['gene'] = t.tollefson_gene.where(t.tollefson_gene.isin(refseq), t.tollefson_gene.map(GENE_ALIASES))
    outside = t[t.gene.isna()]
    alias_rows = t[t.tollefson_gene.isin(GENE_ALIASES)].groupby('tollefson_gene').size().to_dict()
    print(f'genes: {t[t.gene.notna()].tollefson_gene.nunique()} in the reference set '
          f'({t.tollefson_gene.isin(refseq).groupby(t.tollefson_gene).any().sum()} by symbol, renamed {alias_rows}; '
          f'confirmed by accession: {sorted(renamed)}); outside: {outside.tollefson_gene.nunique()} genes, '
          f'{len(outside)} rows', flush=True)

    # ---- source protein sequences
    needed = set(t.loc[t.gene.notna() & t.parsed, 'np_accession'])
    seqs, not_in_step02, fetched, missing = protein_sequences(needed)
    print(f'{len(needed)} NP accessions needed; {len(not_in_step02)} not in transcript_proteins.csv '
          f'(fetched in this run: {len(fetched)}); unavailable: {missing}', flush=True)

    # ---- mapping
    status, ref_pos, pairs = map_rows(t, refseq, seqs)
    t['status'], t['ref_pos'] = status, ref_pos
    for g, r in GENE_ALIASES.items():                       # a renamed symbol must be the same protein
        pr = pairs[pairs.gene == r]
        assert len(pr) and (pr.n_context_verified / pr[['len_source', 'len_reference']].min(axis=1)).min() > 0.8, (g, r)
    unrelated = pairs[pairs.n_context_verified == 0]
    ok = t[t.status == 'ok'].copy()
    ok['variant_id'] = ok.gene + ':' + ok.ref1 + ok.ref_pos.astype(str) + ok.alt1
    assert all(refseq[g][p - 1] == a for g, p, a in zip(ok.gene, ok.ref_pos, ok.ref1))

    # ---- duplicates: keep the first row in table order
    multi = ok[ok.duplicated('variant_id', keep=False)].groupby('variant_id', sort=False)
    dstat = pd.DataFrame({'n': multi.size(), 'n_np': multi.np_accession.nunique(),
                          'spread': multi.ddg.max() - multi.ddg.min()})
    out = ok.drop_duplicates('variant_id', keep='first')
    dup = {'duplicate_rows_removed': int(len(ok) - len(out)), 'variant_ids_with_duplicates': int(len(dstat)),
           'ddg_identical': int((dstat.spread == 0).sum()), 'ddg_differs': int((dstat.spread > 0).sum()),
           'max_abs_ddg_difference': float(dstat.spread.max()) if len(dstat) else 0.0,
           'same_np_groups': int((dstat.n_np == 1).sum()), 'same_np_ddg_differs': int(((dstat.n_np == 1) & (dstat.spread > 0)).sum()),
           'cross_np_groups': int((dstat.n_np > 1).sum()), 'cross_np_ddg_differs': int(((dstat.n_np > 1) & (dstat.spread > 0)).sum())}
    cols = ['variant_id', 'gene', 'ref_pos', 'ref1', 'alt1', 'ddg', 'abs_ddg', 'af2_confidence', 'sasa', 'norm_sasa',
            'tollefson_class', 'np_accession', 'src_pos']
    out = out[cols].sort_values(['gene', 'ref_pos', 'alt1'], kind='stable').reset_index(drop=True)
    assert out.variant_id.is_unique and np.allclose(out.ddg.abs(), out.abs_ddg)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    # ---- labelled variants and sign convention
    v = pd.read_csv(REV / 'data/variants_v8.csv', usecols=['variant_id', 'partition', 'y'])
    lab = v.merge(out[['variant_id', 'ddg', 'norm_sasa']], on='variant_id', how='left')
    cover = {p: {'n': int(len(g)), 'n_ddg': int(g.ddg.notna().sum()), 'frac_ddg': round(float(g.ddg.notna().mean()), 4)}
             for p, g in lab.groupby('partition')}
    cover['all'] = {'n': int(len(lab)), 'n_ddg': int(lab.ddg.notna().sum()), 'frac_ddg': round(float(lab.ddg.notna().mean()), 4)}
    has = lab[lab.ddg.notna()]
    by_y = {('P_LP' if y == 1 else 'B_LB'): {'n': int(len(g)), 'mean_ddg': round(float(g.ddg.mean()), 4),
                                               'median_ddg': round(float(g.ddg.median()), 4)}
            for y, g in has.groupby('y')}
    by_class = {str(c): {'n': int(len(g)), 'mean_ddg': round(float(g.ddg.mean()), 4)}
                for c, g in t.groupby('tollefson_class')}
    checks = [sign_checks(t, 'all rows of Table S4'), sign_checks(out, 'ddgun_v8.csv')]
    votes = [by_y['P_LP']['mean_ddg'] > by_y['B_LB']['mean_ddg'], checks[1]['spearman_ddg_vs_norm_sasa'] < 0,
             checks[1]['mean_ddg_buried_hydrophobic_to_charged_plddt70'] > 0, checks[1]['mean_ddg'] > 0]
    verdict = ('positive ddg = destabilising (opposite sign to native DDGun output)' if all(votes) else
               'negative ddg = destabilising (native DDGun sign)' if not any(votes) else 'INCONCLUSIVE - checks disagree')

    summary = {
        'rows_in': n_in, 'gene_symbols_in': int(t.tollefson_gene.nunique()),
        'genes_in_reference_set': int(t[t.gene.notna()].gene.nunique()), 'renamed_symbols': GENE_ALIASES,
        'renamed_symbol_rows': alias_rows, 'renamed_symbols_confirmed_by_accession': sorted(renamed),
        'genes_outside_reference_set': int(outside.tollefson_gene.nunique()), 'rows_outside_reference_set': int(len(outside)),
        'np_accessions_needed': len(needed), 'np_accessions_from_extra_file': not_in_step02,
        'np_accessions_unavailable': missing,
        'np_gene_pairs': int(len(pairs)), 'np_identical_to_reference': int(pairs.identical_sequence.sum()),
        'np_unrelated_to_reference': dict(zip(unrelated.np_accession, unrelated.gene)),
        'require_context_rule': REQUIRE_CONTEXT, 'min_identical_run': MIN_RUN,
        'rows_by_status': t.status.value_counts().to_dict(),
        'failures_by_gene': {s: g.gene.value_counts().to_dict() for s, g in
                             t[~t.status.isin(['ok', 'gene_not_in_reference_set'])].groupby('status')},
        'rows_ok': int(len(ok)), 'rows_ok_renumbered': int((ok.ref_pos != ok.src_pos).sum()),
        'duplicates': dup, 'rows_out': int(len(out)), 'genes_out': int(out.gene.nunique()),
        'labelled_variants_with_ddg': cover, 'labelled_ddg_by_y': by_y, 'ddg_by_tollefson_class_all_rows': by_class,
        'sign_checks': checks, 'sign_convention': verdict,
    }
    json.dump(summary, open(OUT_SUMMARY, 'w'), indent=2)
    print(json.dumps(summary, indent=1))
    print(f'wrote {OUT} ({len(out)} rows, {out.gene.nunique()} genes)')


if __name__ == '__main__':
    main()
