#!/usr/bin/env python3
"""
Step 06 - Structure / annotation features per residue of the REFERENCE protein and per labelled variant.

Problem being fixed: AlphaFold DB models and UniProt features are numbered on the UniProt canonical isoform
(and some downloaded "F1" files are in fact isoform models), whereas variants are numbered on the reference
protein (step 03). Every look-up therefore goes  reference residue -> UniProt canonical residue -> model
residue, and each hop is verified on the sequence itself, never on file names or residue numbers.

Inputs
  data/variants_v8.csv                               variant_id, gene, ref_pos, ref1, uniprot_pos, partition
  data/sources/reference_proteins.csv                gene, uniprot_id, sequence (reference protein)
  hearing/data/features/gene_sequences.csv           gene, uniprot_id, sequence (UniProt canonical)
  hearing/data/structures/<GENE>/*.pdb               AlphaFold models (pLDDT in the B-factor column)
  hearing/data/external/uniprot/<GENE>_<ACC>.json    UniProt REST entries (`features`)
  code/data/03_canonicalize.py                       make_aligner(), position_map() (imported, same alignment)

Outputs (data/annotations/)
  residue_structure_table.csv   one row per residue of every reference protein:
      gene, ref_pos, ref_aa, uniprot_pos, uniprot_pos_unverified, plddt, plddt_missing, plddt_source,
      plddt_n_models, in_domain, in_transmembrane, in_signal_or_propeptide, in_disulfide,
      in_binding_or_active_site, annot_missing, rel_pos
  structure_v8.csv              one row per variant_id of variants_v8.csv (same order):
      variant_id, gene, ref_pos, ref1 + the feature columns above (looked up at gene, ref_pos)
  structure_v8_summary.json     coverage and bookkeeping counts

Rules
  1. reference -> UniProt canonical: same global alignment as step 03 (position_map). `uniprot_pos` is the
     aligned canonical position if it carries the same amino acid (step-03 rule) AND the residue lies in an
     ungapped run of >= 11 consecutive residues identical in both sequences (MIN_RUN). The second condition
     removes single-residue chance identities inside isoform-specific segments that the global alignment
     pairs up (e.g. the alternative protease domain of MASP1). A step-03 position that fails it is NOT used;
     it is kept for traceability in `uniprot_pos_unverified` and uniprot_pos is left empty.
  2. A gene whose UniProt entry is a different protein (identical aligned residues < 50% of the shorter
     sequence; HARS2 and LRTOMT) gets no uniprot_pos at all, hence no pLDDT and no annotation.
  3. pLDDT = CA B-factor of the model residue that corresponds to uniprot_pos. A model is "canonical" if its
     CA sequence equals the UniProt canonical sequence. Any other model (isoform / fragment) is placed on the
     canonical sequence by unique exact substring match, else by position_map alignment, and a residue is
     used only if it lies in an ungapped run of >= 11 residues identical to the canonical sequence. A stretch
     of a model that the global alignment leaves unplaced is aligned once more on its own (same function,
     same test), so that an isoform lacking a long internal region still contributes both of its ends.
     Canonical model(s) take precedence; otherwise the mean over the distinct verified models that cover the
     residue (byte-identical duplicate files count once). The model residue always has the amino acid ref_aa
     (asserted). plddt_source = canonical | isoform; plddt_missing = 1 where no value exists (plddt empty).
  4. UniProt flags at uniprot_pos (1 if any feature of the listed types covers it, else 0):
        in_domain                  Domain, Repeat, Zinc finger, DNA binding
        in_transmembrane           Transmembrane, Intramembrane
        in_signal_or_propeptide    Signal, Propeptide, Transit peptide
        in_disulfide               Disulfide bond (the two bonded residues only, not the span)
        in_binding_or_active_site  Binding site, Active site, Site
     Features annotated on another isoform (`location.sequence`) are skipped: their coordinates are not
     canonical. A feature with an unknown end point contributes only its known end point.
     annot_missing = 1 if uniprot_pos is empty or the JSON is missing / not the canonical sequence; the five
     flags are then empty (not 0).
  5. rel_pos = ref_pos / length of the reference protein.
  Nothing is imputed. Deterministic and re-runnable (pure function of the inputs).
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'data'))
from paths import legacy_root  # noqa: E402

sys.dont_write_bytecode = True      # importing step 03 must not leave __pycache__ files behind

REV_ROOT = Path(__file__).resolve().parents[2]                   # the release folder
REV = REV_ROOT
VARIANTS = REV / 'data/variants_v8.csv'
REFERENCE = REV / 'data/sources/reference_proteins.csv'
UNIPROT_SEQS = legacy_root() / 'data/features/gene_sequences.csv'
STRUCT_DIR = legacy_root() / 'data/structures'
UNIPROT_DIR = legacy_root() / 'data/external/uniprot'
OUT_DIR = REV / 'data/annotations'

MIN_RUN = 11                 # residues of identical ungapped context required to trust an aligned position
UNRELATED_FRACTION = 0.5     # identical aligned residues / shorter sequence below this = different protein
KNOWN_UNRELATED = {'HARS2', 'LRTOMT'}   # UniProt entry stored for these genes is another protein (DTD1, LRRC51)
FLAG_TYPES = {
    'in_domain': {'Domain', 'Repeat', 'Zinc finger', 'DNA binding'},
    'in_transmembrane': {'Transmembrane', 'Intramembrane'},
    'in_signal_or_propeptide': {'Signal', 'Propeptide', 'Transit peptide'},
    'in_disulfide': {'Disulfide bond'},
    'in_binding_or_active_site': {'Binding site', 'Active site', 'Site'},
}
FLAGS = list(FLAG_TYPES)
AA3 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H',
       'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
       'TYR': 'Y', 'VAL': 'V'}

_spec = importlib.util.spec_from_file_location('canonicalize03', Path(__file__).resolve().parent / '03_canonicalize.py')
canon = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(canon)          # only defines functions/constants; main() is guarded


def identical_run_lengths(src, dst, pmap):
    """{1-based position in src: length of the ungapped run of consecutive identical aligned residues around it}.

    `pmap` is the output of position_map(src, dst) (None = identical sequences). Positions that are unaligned
    or aligned to a different amino acid are absent from the result.
    """
    if pmap is None:
        return {p: len(src) for p in range(1, len(src) + 1)}
    out, run = {}, []
    for s in sorted(pmap):
        d = pmap[s]
        if src[s - 1] != dst[d - 1]:
            continue
        if run and s == run[-1] + 1 and d == pmap[run[-1]] + 1:
            run.append(s)
        else:
            out.update((x, len(run)) for x in run)
            run = [s]
    out.update((x, len(run)) for x in run)
    return out


def read_ca_trace(path):
    """(CA sequence, CA B-factors) of an AlphaFold PDB file (single model, single chain)."""
    seq, bfac, chains = [], [], set()
    with open(path) as fh:
        for line in fh:
            if line.startswith('ENDMDL'):
                break
            if line.startswith('ATOM  ') and line[12:16] == ' CA ' and line[16] in ' A':
                seq.append(AA3.get(line[17:20], 'X'))
                bfac.append(float(line[60:66]))
                chains.add(line[21])
    assert len(chains) == 1, f'{path}: expected one chain, found {sorted(chains)}'
    return ''.join(seq), np.array(bfac, dtype=float)


def model_to_canonical(mseq, useq, aligner):
    """(kind, {canonical position (1-based): model index (0-based)}, n placed by re-alignment).

    Only verified residues of the model are returned. A stretch of the model that the global alignment leaves
    unplaced (an isoform lacking a long internal region makes the gap more expensive than dropping the rest)
    is aligned again on its own with the same position_map and has to pass the same run test.
    """
    if mseq == useq:
        return 'canonical', {k + 1: k for k in range(len(mseq))}, 0
    i = useq.find(mseq)
    if i >= 0 and useq.find(mseq, i + 1) < 0:            # unique exact fragment of the canonical sequence
        return 'isoform', {i + k + 1: k for k in range(len(mseq))}, 0
    placed, n_again = {}, 0                              # canonical position -> model index
    todo = [(0, len(mseq), False)]                       # [start, end) stretches of the model still to place
    while todo:
        a, b, again = todo.pop()
        seg = mseq[a:b]
        pmap = canon.position_map(seg, useq, aligner)
        runs = identical_run_lengths(seg, useq, pmap)
        good = sorted(s for s, n in runs.items() if n >= MIN_RUN)
        for s in good:
            assert pmap[s] not in placed, 'two model residues verified on the same canonical residue'
            placed[pmap[s]] = a + s - 1
        n_again += len(good) if again else 0
        if good:                                         # unplaced stretches between / around the verified runs
            edges = [0] + good + [len(seg) + 1]
            todo += [(a + lo, a + hi - 1, True) for lo, hi in zip(edges[:-1], edges[1:]) if hi - lo - 1 >= MIN_RUN]
    return 'isoform', placed, n_again


def load_models(gene, useq, aligner):
    """Per canonical position: pLDDT from canonical model(s) and from verified isoform/fragment models."""
    n = len(useq)
    by_kind = {'canonical': [], 'isoform': []}
    info, seen = [], set()
    for path in sorted((STRUCT_DIR / gene).glob('*.pdb')):
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        mseq, bfac = read_ca_trace(path)
        if digest in seen:
            info.append({'gene': gene, 'file': path.name, 'kind': 'duplicate', 'n_residues': len(mseq),
                         'n_verified': 0, 'n_realigned': 0})
            continue
        seen.add(digest)
        kind, m2c, n_again = model_to_canonical(mseq, useq, aligner)
        col = np.full(n + 1, np.nan)
        for upos, k in m2c.items():
            assert mseq[k] == useq[upos - 1]
            col[upos] = bfac[k]
        by_kind[kind].append(col)
        info.append({'gene': gene, 'file': path.name, 'kind': kind, 'n_residues': len(mseq),
                     'n_verified': len(m2c), 'n_realigned': n_again})
    return by_kind, info


def uniprot_flags(entry, n):
    """{flag: int8 array indexed by canonical position (1..n)} from a UniProt REST entry, plus bookkeeping counts."""
    flags = {f: np.zeros(n + 1, dtype=np.int8) for f in FLAGS}
    stats = {'used': 0, 'skipped_other_isoform': 0, 'open_ended': 0}
    for ft in entry.get('features', []):
        flag = next((f for f, types in FLAG_TYPES.items() if ft['type'] in types), None)
        if flag is None:
            continue
        loc = ft['location']
        if loc.get('sequence'):                       # coordinates refer to a non-canonical isoform
            stats['skipped_other_isoform'] += 1
            continue
        start, end = loc['start'].get('value'), loc['end'].get('value')
        if start is None or end is None:
            stats['open_ended'] += 1
        known = [p for p in (start, end) if p is not None]
        assert all(1 <= p <= n for p in known), (ft['type'], loc)
        if flag == 'in_disulfide' or len(known) < 2:
            for p in known:
                flags[flag][p] = 1
        else:
            assert start <= end, (ft['type'], loc)
            flags[flag][start:end + 1] = 1
        stats['used'] += 1
    return flags, stats


def main():
    ref = pd.read_csv(REFERENCE)
    uni = pd.read_csv(UNIPROT_SEQS)
    uniseq, uniid = dict(zip(uni.gene, uni.sequence)), dict(zip(uni.gene, uni.uniprot_id))
    v = pd.read_csv(VARIANTS, usecols=['variant_id', 'gene', 'ref_pos', 'ref1', 'uniprot_pos', 'partition'])
    aligner = canon.make_aligner()

    tables, model_info, gene_rows = [], [], []
    feat_stats = {'used': 0, 'skipped_other_isoform': 0, 'open_ended': 0}
    unrelated, no_json = [], []
    for gene, acc, rseq in zip(ref.gene, ref.uniprot_id, ref.sequence):
        useq, n_ref = uniseq[gene], len(rseq)
        assert uniid[gene] == acc
        # ---- reference -> canonical (step-03 rule, then context verification)
        pmap = canon.position_map(rseq, useq, aligner)
        runs = identical_run_lengths(rseq, useq, pmap)
        is_unrelated = len(runs) < UNRELATED_FRACTION * min(len(rseq), len(useq))
        if is_unrelated:
            unrelated.append(gene)
        pos = np.arange(1, n_ref + 1)
        raw = np.array([(p if pmap is None else pmap[p]) if p in runs else 0 for p in pos])       # step-03 uniprot_pos
        ok = np.array([runs.get(p, 0) >= MIN_RUN for p in pos]) & (not is_unrelated)
        upos = np.where(ok, raw, 0)                                                                # verified
        assert all(useq[u - 1] == rseq[p - 1] for p, u in zip(pos, raw) if u)

        # ---- pLDDT on canonical coordinates
        plddt = np.full(n_ref, np.nan)
        source = np.full(n_ref, '', dtype=object)
        n_models = np.zeros(n_ref, dtype=int)
        if not is_unrelated:
            by_kind, info = load_models(gene, useq, aligner)
            model_info += info
            for kind in ('canonical', 'isoform'):               # canonical first; isoform models only fill gaps
                if not by_kind[kind]:
                    continue
                stack = np.vstack(by_kind[kind])[:, upos]       # column 0 (no uniprot_pos) is NaN by construction
                cnt = (~np.isnan(stack)).sum(axis=0)
                mean = np.nansum(stack, axis=0) / np.where(cnt > 0, cnt, 1)
                fill = np.isnan(plddt) & (cnt > 0) & (upos > 0)
                plddt[fill], source[fill], n_models[fill] = mean[fill], kind, cnt[fill]

        # ---- UniProt feature flags on canonical coordinates
        path = UNIPROT_DIR / f'{gene}_{acc}.json'
        entry = json.load(open(path)) if path.exists() else None
        json_ok = (entry is not None and entry.get('primaryAccession') == acc
                   and entry.get('sequence', {}).get('value') == useq)
        if not json_ok:
            no_json.append(gene)
        annot = (upos > 0) & json_ok
        tab = pd.DataFrame({'gene': gene, 'ref_pos': pos, 'ref_aa': list(rseq),
                            'uniprot_pos': pd.array(np.where(upos > 0, upos, -1), dtype='Int64'),
                            'uniprot_pos_unverified': pd.array(np.where((raw > 0) & (upos == 0), raw, -1), dtype='Int64'),
                            'plddt': np.round(plddt, 3), 'plddt_missing': np.isnan(plddt).astype(int),
                            'plddt_source': source, 'plddt_n_models': n_models})
        for c in ('uniprot_pos', 'uniprot_pos_unverified'):
            tab[c] = tab[c].mask(tab[c] == -1)
        if json_ok:
            flags, st = uniprot_flags(entry, len(useq))
            for k in feat_stats:
                feat_stats[k] += 0 if is_unrelated else st[k]
        for f in FLAGS:
            vals = flags[f][upos] if json_ok else np.zeros(n_ref, dtype=np.int8)
            tab[f] = pd.array(np.where(annot, vals, -1), dtype='Int64')
            tab[f] = tab[f].mask(tab[f] == -1)
        tab['annot_missing'] = (~annot).astype(int)
        tab['rel_pos'] = np.round(pos / n_ref, 6)
        tables.append(tab)
        gene_rows.append({'gene': gene, 'n_residues': n_ref, 'n_uniprot_pos': int((upos > 0).sum()),
                          'n_unverified': int(((raw > 0) & (upos == 0)).sum()), 'n_plddt': int((~np.isnan(plddt)).sum()),
                          'plddt_coverage': round(float((~np.isnan(plddt)).mean()), 4),
                          'annot_coverage': round(float(annot.mean()), 4),
                          'plddt_sources': '+'.join(sorted(set(source) - {''})) or 'none'})

    assert KNOWN_UNRELATED <= set(unrelated), f'expected unrelated UniProt entries not detected: {unrelated}'
    res = pd.concat(tables, ignore_index=True)
    assert len(res) == int(ref.sequence.str.len().sum()) and not res.duplicated(['gene', 'ref_pos']).any()
    assert ((res.plddt_missing == 1) == res.plddt.isna()).all() and (res.plddt.dropna().between(0, 100)).all()
    assert (res.loc[res.annot_missing == 1, FLAGS].isna().all().all()
            and res.loc[res.annot_missing == 0, FLAGS].notna().all().all())
    assert res.loc[res.uniprot_pos.isna(), 'plddt'].isna().all()

    # ---- per-variant table = look-up of the residue table at (gene, ref_pos)
    feat_cols = ['uniprot_pos', 'uniprot_pos_unverified', 'plddt', 'plddt_missing', 'plddt_source', 'plddt_n_models',
                 *FLAGS, 'annot_missing', 'rel_pos']
    sv = v[['variant_id', 'gene', 'ref_pos', 'ref1']].merge(res[['gene', 'ref_pos', 'ref_aa', *feat_cols]],
                                                           on=['gene', 'ref_pos'], how='left', validate='many_to_one')
    assert len(sv) == len(v) and (sv.variant_id.values == v.variant_id.values).all()
    assert (sv.ref1 == sv.ref_aa).all(), 'variant reference residue differs from the reference protein'
    sv = sv.drop(columns='ref_aa')

    # ---- numbering shared with variants_v8.csv: verified positions must equal the step-03 uniprot_pos
    theirs = v.uniprot_pos.fillna(-1).astype(int).to_numpy()
    mine = sv.uniprot_pos.fillna(-1).astype(int).to_numpy()
    rejected = sv.uniprot_pos_unverified.fillna(-1).astype(int).to_numpy()
    both = (theirs > 0) & (mine > 0)
    assert (theirs[both] == mine[both]).all(), 'uniprot_pos disagrees with variants_v8.csv'
    is_dropped = (theirs > 0) & (mine < 0)             # step-03 position present but not verified here
    assert (rejected[is_dropped] == theirs[is_dropped]).all(), 'rejected positions differ from variants_v8.csv'
    dropped = sv[is_dropped]
    n_extra = int(((theirs < 0) & (mine > 0)).sum())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT_DIR / 'residue_structure_table.csv', index=False)
    sv.to_csv(OUT_DIR / 'structure_v8.csv', index=False)

    # ---- report
    gcov = pd.DataFrame(gene_rows)
    part = v[['partition']].assign(has_plddt=(sv.plddt_missing == 0).to_numpy(), has_annot=(sv.annot_missing == 0).to_numpy())
    by_part = {p: {'n': int(len(g)), 'n_plddt': int(g.has_plddt.sum()), 'frac_plddt': round(float(g.has_plddt.mean()), 4),
                   'n_annot': int(g.has_annot.sum()), 'frac_annot': round(float(g.has_annot.mean()), 4)}
               for p, g in part.groupby('partition')}
    by_part['all'] = {'n': int(len(part)), 'n_plddt': int(part.has_plddt.sum()),
                      'frac_plddt': round(float(part.has_plddt.mean()), 4), 'n_annot': int(part.has_annot.sum()),
                      'frac_annot': round(float(part.has_annot.mean()), 4)}
    low = gcov[gcov.plddt_coverage < 0.80].sort_values('plddt_coverage')
    minfo = pd.DataFrame(model_info)
    summary = {
        'n_genes': int(len(ref)), 'n_residues': int(len(res)), 'n_variants': int(len(sv)),
        'min_identical_run': MIN_RUN,
        'residues_with_uniprot_pos': int(res.uniprot_pos.notna().sum()),
        'residues_with_plddt': int((res.plddt_missing == 0).sum()),
        'residue_plddt_source': res.plddt_source.replace('', 'none').value_counts().to_dict(),
        'residues_with_annotation': int((res.annot_missing == 0).sum()),
        'genes_unrelated_uniprot_entry': unrelated, 'genes_without_usable_json': no_json,
        'genes_by_plddt_source': gcov.plddt_sources.value_counts().to_dict(),
        'model_files': minfo.kind.value_counts().to_dict() if len(minfo) else {},
        'isoform_models': minfo[minfo.kind == 'isoform'].drop(columns='kind').to_dict('records') if len(minfo) else [],
        'uniprot_features': feat_stats,
        'residues_step03_position_rejected': {g: int(n) for g, n in zip(gcov.gene, gcov.n_unverified) if n},
        'variants_step03_position_rejected': dropped.variant_id.tolist(),
        'variants_with_position_absent_in_variants_v8': n_extra,
        'variant_coverage_by_partition': by_part,
        'variant_flag_prevalence': {f: round(float(sv[f].astype('Float64').mean()), 4) for f in FLAGS},
        'genes_plddt_coverage_below_0.80': dict(zip(low.gene, low.plddt_coverage)),
    }
    json.dump(summary, open(OUT_DIR / 'structure_v8_summary.json', 'w'), indent=2)
    print(json.dumps(summary, indent=1))
    print(f'wrote residue_structure_table.csv ({len(res)} rows), structure_v8.csv ({len(sv)} rows)')


if __name__ == '__main__':
    main()
