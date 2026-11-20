#!/usr/bin/env python3
"""
Step 10 - ClinGen expert-panel (VCEP) classifications of missense variants on reference-protein numbering.

Purpose: an expert-curated label set with the applied ACMG/AMP evidence codes (PP3 / BP4 / PM2 ...), used to
(a) evaluate the model on expert classifications it has not seen and (b) relate the model score to the
computational-evidence codes actually applied by the Hearing Loss VCEP.

Inputs (read-only)
  - hearing/data/external/clingen_erepo_full_hhl.csv : official ClinGen Evidence Repository export for the
    panel genes (one row per curated variant x disease; includes retracted interpretations)
  - data/sources/reference_proteins.csv              : reference protein per gene (step 03)
  - data/sources/transcript_proteins.csv (+ transcript_proteins_extra.csv, transcript_proteins_extra2.csv
    when present)                                    : protein of every source RefSeq transcript (step 02)
  - data/sources/clinvar_2026-04.csv                 : only to resolve the protein change of ERepo rows whose
    name carries none (see rule 2) and to cross-check the mapping
  - data/sources/clinvar_2026-04_canon.csv, clinvar_2025-01_canon.csv : cross-check / 2025-01 state
  - data/variants_v8.csv                             : partition and cv_fold of labelled variants (step 04)

Rules
  1. Retracted interpretations are dropped.
  2. The protein change is parsed from the ERepo variant name with the same regular expression as the ClinVar
     extraction (step 01). Some ERepo rows have no protein change in the name ("NM_004004.5:c.677T>A",
     "NM_000441.1(SLC26A4):c.349C>T", or no name at all) although several of them are missense SNVs. For
     those rows, and only those, the ClinVar name of the same variant is used instead: the variant is
     looked up in the ClinVar 2026-04 extraction (SNVs only) by ClinVar Variation Id, else by an exact
     "NM_x.y:c.change" match among the row's HGVS expressions (unique match required); in both cases the
     GRCh38 genomic change of the ClinVar record must be present among the ERepo HGVS expressions. Such
     rows are flagged in `protein_change_source` (and `clinvar_variation_id` is filled from the matched
     record when ERepo gives none). Rows that cannot be resolved stay unparsed (= "other").
  3. missense = reference and alternate are two different standard amino acids (not Ter, not "=");
     everything else is "other" (step 01 `vclass`).
  4. Missense rows are mapped onto the gene's reference protein with step 03 `canonicalize` (global
     alignment of the protein of the NM accession in the name to the reference protein; a variant is kept
     only if the reported residue matches its source protein, the position aligns to the reference and the
     aligned reference residue is identical). NM accessions that are not yet in the transcript-protein
     tables are fetched exactly as in step 02 (efetch nuccore fasta_cds_aa) and appended to
     data/sources/transcript_proteins_extra2.csv. No other fall-back is used.
  5. One row per variant_id: if a protein variant is curated in more than one row (several diseases or
     several nucleotide changes) the most recently published row is kept (ties: latest approval date, then
     Uuid) and the number of rows is stored in `n_erepo_rows`.
  6. y_expert = 1 for P / LP, 0 for B / LB, missing for VUS. Evidence-code flags are derived from "Applied
     Evidence Codes (Met)" only; a code counts at any strength (PP3, PP3_Moderate, ...). 0 therefore means
     "not applied", which includes "not evaluated" (e.g. BA1-only curations); `codes_not_met` keeps the
     explicitly not-met codes.

Outputs (data/annotations/)
  - clingen_expert_v8.csv          : one row per expert-curated missense variant
        variant_id, gene, expert_panel, assertion (P/LP/VUS/LB/B), y_expert, codes_met (pipe-separated, as
        given), pp3_met, pp3_strength, bp4_met, bp4_strength, pm2_met, clinvar_variation_id, approval_date,
        published_date, mode_of_inheritance, partition, cv_fold (missing when the variant is not in
        variants_v8.csv), then audit columns: ref_pos, ref1, alt1, disease, codes_not_met, n_erepo_rows,
        erepo_name, protein_change_source, uuid, cv25_definitive, cv25_state, cv25_stars
        (ClinVar 2025-01 protein-level state: definitive = a record with status P or B and >= 1 star).
  - clingen_expert_v8_summary.json : validation counts (panel x assertion, mapping, partitions, evidence
        codes, ClinVar 2025-01 state, cross-check of every variant_id against the ClinVar 2026-04 record
        with the same VariationID, agreement of the labelled set with the expert assertion, and which
        version of variants_v8.csv was joined)
  - data/sources/transcript_proteins_extra2.csv is created / extended only if an accession had to be fetched.

partition / cv_fold are a snapshot of data/variants_v8.csv at run time (step 04b rewrites cv_fold in place):
re-run this step whenever that file changes. Deterministic; no imputation. Network is used only when a
needed NM accession is not cached.
"""
import importlib.util
import json
import re
import time
from pathlib import Path

import pandas as pd

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'data'))
from paths import legacy_root  # noqa: E402

REV_ROOT = Path(__file__).resolve().parents[2]                   # the release folder
REV = REV_ROOT
HERE = REV / 'code/data'
SRC = REV / 'data/sources'
EREPO = legacy_root() / 'data/external/clingen_erepo_full_hhl.csv'
UNISEQ = legacy_root() / 'data/features/gene_sequences.csv'
OUT = REV / 'data/annotations/clingen_expert_v8.csv'
SUMMARY = REV / 'data/annotations/clingen_expert_v8_summary.json'
EXTRA2 = SRC / 'transcript_proteins_extra2.csv'
TX_FILES = ['transcript_proteins.csv', 'transcript_proteins_extra.csv', 'transcript_proteins_extra2.csv']
LOOKUP_SNAPSHOT, TRAIN_SNAPSHOT = '2026-04', '2025-01'

ASSERTION = {'pathogenic': 'P', 'likely pathogenic': 'LP', 'uncertain significance': 'VUS',
             'likely benign': 'LB', 'benign': 'B'}
Y_EXPERT = {'P': 1.0, 'LP': 1.0, 'B': 0.0, 'LB': 0.0}
ASSERTION_ORDER = ['P', 'LP', 'VUS', 'LB', 'B']


def load_step(filename):
    """Import a numbered pipeline step as a module (their file names are not valid identifiers)."""
    spec = importlib.util.spec_from_file_location('step_' + filename.split('_')[0], HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_transcript_proteins():
    """accession -> protein sequence, accession -> table it came from (first file wins)."""
    frames = []
    for name in TX_FILES:
        path = SRC / name
        if path.exists():
            t = pd.read_csv(path, dtype=str)
            t = t[t.sequence.notna()]
            t = t[t.sequence.str.len() == t.length.astype(int)]      # guard against a truncated row
            frames.append(t.assign(table=name))
    t = pd.concat(frames, ignore_index=True).drop_duplicates('accession')
    return dict(zip(t.accession, t.sequence)), dict(zip(t.accession, t.table))


def fetch_missing_transcripts(accessions, tx, s02):
    """Fetch NM proteins that are not cached (same call as step 02) and append them to EXTRA2."""
    todo = sorted(a for a in set(accessions) if isinstance(a, str) and a.startswith('NM_') and a not in tx)
    new = []
    for i in range(0, len(todo), 40):
        batch = todo[i:i + 40]
        text = s02.efetch('nuccore', batch, 'fasta_cds_aa')
        for header, seq in s02.parse_fasta(text):
            m = re.match(r'lcl\|(N[MR]_\d+\.\d+)_prot_([A-Z]P_\d+\.\d+)', header)
            if m and m.group(1) in batch:
                new.append({'accession': m.group(1), 'kind': 'NM', 'protein_id': m.group(2),
                            'length': len(seq), 'sequence': seq})
        time.sleep(0.4)
    if new:
        add = pd.DataFrame(new).drop_duplicates('accession')
        if EXTRA2.exists():
            add = pd.concat([pd.read_csv(EXTRA2, dtype=str), add.astype(str)], ignore_index=True)
            add = add.drop_duplicates('accession')
        add.to_csv(EXTRA2, index=False)
        for r in new:
            tx.setdefault(r['accession'], r['sequence'])
    return todo, sorted({r['accession'] for r in new})


def resolve_names_via_clinvar(kept, need):
    """Rule 2: ClinVar name for ERepo rows without a protein change in their own name.

    Returns {row index: (clinvar_name, source, clinvar_variation_id)}; a record is accepted only if its
    GRCh38 genomic change is listed among the row's HGVS expressions.
    """
    cv = pd.read_csv(SRC / f'clinvar_{LOOKUP_SNAPSHOT}.csv', dtype=str,
                     usecols=['VariationID', 'Name', 'nm', 'c_change', 'p_ref', 'Chromosome', 'PositionVCF',
                              'ReferenceAlleleVCF', 'AlternateAlleleVCF'])
    cv = cv[cv.p_ref.notna()]                                  # only records whose name has a protein change
    assert cv.VariationID.is_unique
    by_id = cv.set_index('VariationID')
    by_hgvs = {}
    for vid, nm, cc in zip(cv.VariationID, cv.nm, cv.c_change):
        by_hgvs.setdefault(f'{nm}:{cc}', set()).add(vid)
    out = {}
    for idx in kept.index[need]:
        row = kept.loc[idx]
        exprs = [x.strip() for x in str(row['HGVS Expressions']).split(',')]
        vid, source = None, None
        if row['ClinVar Variation Id'] in by_id.index:
            vid, source = row['ClinVar Variation Id'], 'clinvar_name_by_variation_id'
        else:
            hits = set().union(*[by_hgvs.get(x, set()) for x in exprs])
            if len(hits) == 1:
                vid, source = next(iter(hits)), 'clinvar_name_by_hgvs'
        if vid is None:
            continue
        r = by_id.loc[vid]
        chrom = {'X': '23', 'Y': '24'}.get(r.Chromosome, r.Chromosome)
        if not str(chrom).isdigit():
            continue
        g_change = f':g.{r.PositionVCF}{r.ReferenceAlleleVCF}>{r.AlternateAlleleVCF}'
        if any(x.startswith(f'NC_{int(chrom):06d}.') and x.endswith(g_change) for x in exprs):
            out[idx] = (r.Name, source, vid)
    return out


def clinvar_state(snapshot, variant_ids):
    """Protein-level ClinVar state of the given variant_ids: {variant_id: (definitive, state, stars)}.

    definitive = at least one record with status P or B and >= 1 star; state uses the same precedence as
    step 04 `clinvar_protein_level`; variant_ids without any record are absent from the result.
    """
    c = pd.read_csv(SRC / f'clinvar_{snapshot}_canon.csv', dtype=str,
                    usecols=['variant_id', 'canon_status', 'status', 'stars'])
    c = c[(c.canon_status == 'ok') & c.variant_id.isin(set(variant_ids))].copy()
    c['stars'] = c.stars.astype(int)
    c['definitive'] = c.status.isin(['P', 'B']) & (c.stars >= 1)
    rows = {}
    for vid, g in c.groupby('variant_id', sort=False):
        d = g[g.definitive]
        has_p, has_b = (d.status == 'P').any(), (d.status == 'B').any()
        if has_p and has_b:
            state = 'PB_CONFLICT'
        elif has_p:
            state = 'P'
        elif has_b:
            state = 'B'
        elif (g.status == 'CONFLICT').any():
            state = 'CONFLICT'
        elif g.status.isin(['P', 'B']).any():
            state = 'LOWSTAR'
        elif (g.status == 'VUS').any():
            state = 'VUS'
        else:
            state = 'OTHER'
        rows[vid] = (int(len(d) > 0), state, int(d.stars.max()) if len(d) else int(g.stars.max()))
    return rows


def load_labelled(cols):
    """Columns `cols` of data/variants_v8.csv plus a provenance record.

    Other steps rewrite this file in place (04b overwrites cv_fold), so the row count is checked against
    data/split_summary.json to detect a partially written file, and the modification time and fold sizes
    of the version that was joined are stored in the summary.
    """
    path = REV / 'data/variants_v8.csv'
    v = pd.read_csv(path, usecols=cols, low_memory=False)
    with open(REV / 'data/split_summary.json') as fh:
        expect = sum(p['n'] for p in json.load(fh)['partitions'].values())
    assert len(v) == expect and v.variant_id.is_unique, \
        f'{path}: {len(v)} rows, expected {expect} unique variants (file being rewritten?)'
    info = {'rows': int(len(v)),
            'modified': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(path.stat().st_mtime))}
    if 'cv_fold' in v:
        info['cv_fold_sizes'] = {int(k): int(n) for k, n in v.cv_fold.value_counts().sort_index().items()}
    return v, info


def split_codes(s):
    return [t.strip() for t in s.split(',') if t.strip()] if isinstance(s, str) else []


def jsonable(o):
    """Recursively convert numpy / pandas scalars (also as dict keys) to plain Python for json.dump."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if o is None or o is pd.NA:
        return None
    if hasattr(o, 'item'):
        o = o.item()
    if isinstance(o, float) and o != o:
        return None
    return o


def crosstab_dict(a, b):
    return {str(k): {str(kk): int(vv) for kk, vv in v.items() if vv} for k, v in pd.crosstab(a, b).T.to_dict().items()}


def main():
    s01, s02, s03 = (load_step(f) for f in ['01_extract_sources.py', '02_fetch_transcript_proteins.py',
                                            '03_canonicalize.py'])
    summary = {}

    # ---- 1. read, drop retracted
    e = pd.read_csv(EREPO, dtype=str)
    retracted = e['Retracted'].str.strip().str.lower() == 'true'
    kept = e[~retracted].copy()
    kept['gene'] = kept['HGNC Gene Symbol'].str.strip()
    unknown = sorted(set(kept['Assertion'].str.strip().str.lower()) - set(ASSERTION))
    assert not unknown, f'unknown assertion terms: {unknown}'
    kept['assertion'] = kept['Assertion'].str.strip().str.lower().map(ASSERTION)
    summary['input'] = {'rows': int(len(e)), 'retracted_dropped': int(retracted.sum()), 'rows_kept': int(len(kept)),
                        'published_date_range': [kept['Published Date'].min(), kept['Published Date'].max()]}
    summary['all_kept_rows_panel_by_assertion'] = crosstab_dict(kept['Expert Panel'], kept.assertion)

    # ---- 2. protein change from the name (ClinVar name only where the ERepo name has none)
    kept['name_used'] = kept['#Variation']
    kept['protein_change_source'] = 'erepo_name'
    parsed = kept['name_used'].str.extract(s01.NAME_RE)
    need = parsed[3].isna()
    resolved = resolve_names_via_clinvar(kept, need.to_numpy())
    kept['clinvar_variation_id'] = kept['ClinVar Variation Id'].where(
        kept['ClinVar Variation Id'].str.fullmatch(r'\d+', na=False), pd.NA)
    for idx, (name, source, vid) in resolved.items():
        kept.loc[idx, 'name_used'] = name
        kept.loc[idx, 'protein_change_source'] = source
        kept.loc[idx, 'clinvar_variation_id'] = vid      # equal to the ERepo id, or fills a missing one
    parsed = kept['name_used'].str.extract(s01.NAME_RE)
    parsed.columns = ['nm', 'name_gene', 'c_change', 'p_ref', 'p_pos', 'p_alt']
    kept = pd.concat([kept, parsed], axis=1)
    kept['vclass'] = [s01.vclass(r if isinstance(r, str) else None, a if isinstance(a, str) else None)
                      for r, a in zip(kept.p_ref, kept.p_alt)]
    kept['is_missense'] = kept.vclass == 'missense'
    summary['protein_change'] = {
        'rows_without_protein_change_in_erepo_name': int(need.sum()),
        'resolved_via_clinvar': {str(kept.loc[i, 'ClinVar Variation Id']) + '/' + str(kept.loc[i, 'Allele Registry Id']):
                                 {'erepo_name': None if pd.isna(kept.loc[i, '#Variation']) else kept.loc[i, '#Variation'],
                                  'clinvar_name': n, 'source': s, 'clinvar_variation_id': v,
                                  'assertion': kept.loc[i, 'assertion'], 'vclass': kept.loc[i, 'vclass']}
                                 for i, (n, s, v) in resolved.items()},
        'vclass_of_kept_rows': kept.vclass.value_counts().to_dict(),
        'missense_rows': int(kept.is_missense.sum()), 'other_rows': int((~kept.is_missense).sum()),
        'missense_rows_by_protein_change_source': kept[kept.is_missense].protein_change_source.value_counts().to_dict()}
    mis = kept[kept.is_missense].copy()
    gene_mismatch = mis[mis.name_gene.notna() & (mis.name_gene != mis.gene)]
    assert gene_mismatch.empty, f'gene in name differs from HGNC symbol:\n{gene_mismatch[["#Variation", "gene"]]}'

    # ---- 4. map onto the reference protein (step 03 logic)
    refdf = pd.read_csv(SRC / 'reference_proteins.csv')
    refseq = dict(zip(refdf.gene, refdf.sequence))
    uni = pd.read_csv(UNISEQ)
    uniseq = dict(zip(uni.gene, uni.sequence))
    tx, tx_table = load_transcript_proteins()
    requested, fetched = fetch_missing_transcripts(mis.nm, tx, s02)
    aligner = s03.make_aligner()
    maps = {(acc, gene): s03.position_map(tx[acc], refseq[gene], aligner)
            for acc, gene in sorted(set(zip(mis.nm, mis.gene))) if acc in tx and gene in refseq}
    ref2uni = {g: (s03.position_map(refseq[g], uniseq[g], aligner) if g in uniseq else None)
               for g in sorted(set(mis.gene)) if g in refseq}
    mis = s03.canonicalize(mis, 'nm', tx, refseq, maps, ref2uni, uniseq)
    ok = mis.canon_status == 'ok'
    summary['mapping'] = {
        'transcripts_not_cached': requested, 'transcripts_fetched_now': fetched,
        'transcripts_served_from_extra2': sorted(a for a in set(mis.nm) if a in fetched
                                                 or tx_table.get(a) == EXTRA2.name),
        'source_transcripts': {f'{a}|{g}': ('identical_to_reference' if m is None else 'aligned')
                               for (a, g), m in maps.items()},
        'missense_rows': int(len(mis)), 'mapped_rows': int(ok.sum()),
        'by_status': mis.canon_status.value_counts().to_dict(),
        'renumbered_rows': int((ok & (mis.ref_pos.astype('float') != mis.p_pos.astype(float))).sum()),
        'failures': [{'name': n, 'gene': g, 'assertion': a, 'reason': s} for n, g, a, s in
                     zip(mis.name_used[~ok], mis.gene[~ok], mis.assertion[~ok], mis.canon_status[~ok])]}

    # cross-check: same variant_id as the ClinVar 2026-04 record with the same VariationID
    cvc = pd.read_csv(SRC / f'clinvar_{LOOKUP_SNAPSHOT}_canon.csv', dtype=str,
                      usecols=['VariationID', 'canon_status', 'variant_id'])
    cv_vid = dict(zip(cvc.VariationID[cvc.canon_status == 'ok'], cvc.variant_id[cvc.canon_status == 'ok']))
    chk = mis[ok & mis.clinvar_variation_id.isin(cv_vid)]
    same = chk.variant_id == chk.clinvar_variation_id.map(cv_vid)
    summary['mapping']['crosscheck_vs_clinvar_2026-04_by_variation_id'] = {
        'compared': int(len(chk)), 'identical_variant_id': int(same.sum()),
        'different': [{'erepo': a, 'clinvar': cv_vid[b]} for a, b in
                      zip(chk.variant_id[~same], chk.clinvar_variation_id[~same])],
        'mapped_rows_not_in_clinvar_canon': int((ok & ~mis.clinvar_variation_id.isin(cv_vid)).sum())}

    # ---- 5./6. one row per variant_id, labels and evidence-code flags
    m = mis[ok].copy()
    m = m.sort_values(['variant_id', 'Published Date', 'Approval Date', 'Uuid'], ascending=[True, False, False, True])
    grp = m.groupby('variant_id', sort=False)
    m['n_erepo_rows'] = grp.variant_id.transform('size')
    dup = m[m.n_erepo_rows > 1]
    summary['duplicates'] = {
        'variant_ids_with_more_than_one_row': int(dup.variant_id.nunique()),
        'rows_collapsed': int(len(dup) - dup.variant_id.nunique()),
        'detail': [{'variant_id': v, 'rows': [{'name': n, 'disease': d, 'assertion': a, 'published': p}
                                              for n, d, a, p in zip(g.name_used, g.Disease, g.assertion, g['Published Date'])]}
                   for v, g in dup.groupby('variant_id', sort=False)],
        'with_discordant_assertions': int((dup.groupby('variant_id').assertion.nunique() > 1).sum())}
    m = m.drop_duplicates('variant_id', keep='first')

    met = m['Applied Evidence Codes (Met)'].map(split_codes)
    notmet = m['Applied Evidence Codes (Not Met)'].map(split_codes)

    def strength(codes, prefix):
        hit = [c for c in codes if c.startswith(prefix)]
        return '|'.join(hit) if hit else pd.NA

    out = pd.DataFrame({
        'variant_id': m.variant_id, 'gene': m.gene, 'expert_panel': m['Expert Panel'], 'assertion': m.assertion,
        'y_expert': m.assertion.map(Y_EXPERT),
        'codes_met': met.map(lambda c: '|'.join(c) if c else pd.NA),
        'pp3_met': met.map(lambda c: int(any(x.startswith('PP3') for x in c))),
        'pp3_strength': met.map(lambda c: strength(c, 'PP3')),
        'bp4_met': met.map(lambda c: int(any(x.startswith('BP4') for x in c))),
        'bp4_strength': met.map(lambda c: strength(c, 'BP4')),
        'pm2_met': met.map(lambda c: int(any(x.startswith('PM2') for x in c))),
        'clinvar_variation_id': m.clinvar_variation_id,
        'approval_date': m['Approval Date'], 'published_date': m['Published Date'],
        'mode_of_inheritance': m['Mode of Inheritance'],
    })
    lab, summary['variants_v8_used'] = load_labelled(['variant_id', 'partition', 'cv_fold', 'y'])
    out = out.merge(lab[['variant_id', 'partition', 'cv_fold']], on='variant_id', how='left')
    # consistency of the labelled set with the expert assertion (reported only; nothing is changed)
    cmp_ = out[['variant_id', 'assertion', 'y_expert']].merge(lab, on='variant_id', how='inner')
    cdef, cvus = cmp_[cmp_.y_expert.notna()], cmp_[cmp_.y_expert.isna()]
    summary['labelled_set_vs_expert'] = {
        'expert_PLP_BLB_in_variants_v8': int(len(cdef)),
        'label_y_equals_expert': int((cdef.y == cdef.y_expert).sum()),
        'label_y_differs': [{'variant_id': v, 'assertion': a, 'partition': p, 'y': float(y)} for v, a, p, y in
                            zip(cdef.variant_id, cdef.assertion, cdef.partition, cdef.y) if y != Y_EXPERT[a]],
        'expert_PLP_BLB_not_in_variants_v8': sorted(set(out.variant_id[out.y_expert.notna()]) - set(lab.variant_id)),
        'expert_VUS_with_label_in_variants_v8': int(len(cvus)),
        'expert_VUS_with_label_by_partition_and_y': {p: {str(int(y)): int(n) for y, n in g.y.value_counts().sort_index().items()}
                                                     for p, g in cvus.groupby('partition')}}
    out['cv_fold'] = out.cv_fold.astype('Int64')
    aud = pd.DataFrame({
        'variant_id': m.variant_id, 'ref_pos': m.ref_pos.astype(int), 'ref1': m.ref1, 'alt1': m.alt1,
        'disease': m.Disease, 'codes_not_met': notmet.map(lambda c: '|'.join(c) if c else pd.NA),
        'n_erepo_rows': m.n_erepo_rows, 'erepo_name': m['#Variation'],
        'protein_change_source': m.protein_change_source, 'uuid': m.Uuid})
    out = out.merge(aud, on='variant_id', how='left')
    st25 = clinvar_state(TRAIN_SNAPSHOT, out.variant_id)
    out['cv25_definitive'] = out.variant_id.map(lambda v: st25.get(v, (0, 'ABSENT', pd.NA))[0])
    out['cv25_state'] = out.variant_id.map(lambda v: st25.get(v, (0, 'ABSENT', pd.NA))[1])
    out['cv25_stars'] = out.variant_id.map(lambda v: st25.get(v, (0, 'ABSENT', pd.NA))[2]).astype('Int64')
    out = out.sort_values(['gene', 'ref_pos', 'alt1']).reset_index(drop=True)
    assert out.variant_id.is_unique and out.variant_id.notna().all()
    assert (out.variant_id == out.gene + ':' + out.ref1 + out.ref_pos.astype(str) + out.alt1).all()
    assert all(refseq[g][p - 1] == a for g, p, a in zip(out.gene, out.ref_pos, out.ref1))
    out.to_csv(OUT, index=False)

    # ---- validation summary
    part = out.partition.fillna('not_in_variants_v8')
    defin = out[out.y_expert.notna()]
    lab_state = defin.cv25_state.map({'P': 1.0, 'B': 0.0})
    summary['output'] = {
        'file': str(OUT), 'rows': int(len(out)), 'genes': out.gene.value_counts().to_dict(),
        'panel_by_assertion': crosstab_dict(out.expert_panel, out.assertion),
        'assertion': {a: int((out.assertion == a).sum()) for a in ASSERTION_ORDER},
        'y_expert': {'1': int((out.y_expert == 1).sum()), '0': int((out.y_expert == 0).sum()),
                     'missing_VUS': int(out.y_expert.isna().sum())},
        'by_partition': part.value_counts().to_dict(),
        'partition_by_assertion': crosstab_dict(part, out.assertion),
        'partition_by_panel': crosstab_dict(part, out.expert_panel),
        'protein_change_source': out.protein_change_source.value_counts().to_dict(),
        'missing_mode_of_inheritance': int(out.mode_of_inheritance.isna().sum()),
        'rows_without_any_met_code': int(out.codes_met.isna().sum())}
    summary['evidence_codes'] = {
        'pp3_met': int(out.pp3_met.sum()), 'bp4_met': int(out.bp4_met.sum()), 'pm2_met': int(out.pm2_met.sum()),
        'pp3_and_bp4_both_met': int((out.pp3_met & out.bp4_met).sum()),
        'pp3_strength': out.pp3_strength.value_counts().to_dict(),
        'bp4_strength': out.bp4_strength.value_counts().to_dict(),
        'pm2_codes': met.map(lambda c: strength(c, 'PM2')).value_counts().to_dict(),
        'pp3_met_by_assertion': {a: int(out.pp3_met[out.assertion == a].sum()) for a in ASSERTION_ORDER},
        'bp4_met_by_assertion': {a: int(out.bp4_met[out.assertion == a].sum()) for a in ASSERTION_ORDER},
        'pp3_met_by_panel': out.groupby('expert_panel').pp3_met.sum().astype(int).to_dict(),
        'bp4_met_by_panel': out.groupby('expert_panel').bp4_met.sum().astype(int).to_dict(),
        'pp3_explicitly_not_met': int(notmet.map(lambda c: any(x.startswith('PP3') for x in c)).sum()),
        'bp4_explicitly_not_met': int(notmet.map(lambda c: any(x.startswith('BP4') for x in c)).sum()),
        'pp3_met_by_partition': out.groupby(part).pp3_met.sum().astype(int).to_dict(),
        'bp4_met_by_partition': out.groupby(part).bp4_met.sum().astype(int).to_dict()}
    summary[f'clinvar_{TRAIN_SNAPSHOT}_state_of_expert_PLP_BLB'] = {
        'n_expert_PLP_BLB': int(len(defin)),
        'definitive_ge1star': int(defin.cv25_definitive.sum()),
        'not_definitive': int((defin.cv25_definitive == 0).sum()),
        'by_state': defin.cv25_state.value_counts().to_dict(),
        'definitive_by_stars': defin[defin.cv25_definitive == 1].cv25_stars.value_counts().sort_index().to_dict(),
        'definitive_same_direction_as_expert': int((lab_state == defin.y_expert).sum()),
        'definitive_opposite_or_PB_conflict': int(((defin.cv25_definitive == 1) & (lab_state != defin.y_expert)).sum()),
        'not_definitive_by_partition': part[defin.index][defin.cv25_definitive == 0].value_counts().to_dict(),
        'definitive_by_partition': part[defin.index][defin.cv25_definitive == 1].value_counts().to_dict(),
        'not_definitive_by_assertion': defin[defin.cv25_definitive == 0].assertion.value_counts().to_dict()}
    s = jsonable(summary)
    SUMMARY.write_text(json.dumps(s, indent=2) + '\n')

    # ---- compact print
    print(f"ERepo rows {s['input']['rows']}; retracted dropped {s['input']['retracted_dropped']}; kept {s['input']['rows_kept']}")
    print(f"kept rows, expert panel x assertion: {s['all_kept_rows_panel_by_assertion']}")
    print(f"protein change: {s['protein_change']['vclass_of_kept_rows']}; no protein change in ERepo name "
          f"{s['protein_change']['rows_without_protein_change_in_erepo_name']}, resolved via ClinVar "
          f"{len(resolved)} -> {[(v['clinvar_name'], v['source']) for v in s['protein_change']['resolved_via_clinvar'].values()]}")
    print(f"missense rows {s['mapping']['missense_rows']}; mapped {s['mapping']['mapped_rows']}; by status "
          f"{s['mapping']['by_status']}; renumbered {s['mapping']['renumbered_rows']}; fetched {fetched}")
    print(f"failures: {s['mapping']['failures']}")
    print(f"cross-check vs ClinVar {LOOKUP_SNAPSHOT}: {s['mapping']['crosscheck_vs_clinvar_2026-04_by_variation_id']}")
    print(f"duplicates: {s['duplicates']['variant_ids_with_more_than_one_row']} variant_ids, "
          f"{s['duplicates']['rows_collapsed']} rows collapsed")
    print(f"OUTPUT {len(out)} expert missense variants; panel x assertion: {s['output']['panel_by_assertion']}")
    print(f"variants_v8.csv used: {s['variants_v8_used']}")
    print(f"by partition: {s['output']['by_partition']}")
    print(f"partition x assertion: {s['output']['partition_by_assertion']}")
    print(f"labelled set vs expert: {s['labelled_set_vs_expert']}")
    print(f"PP3 met {s['evidence_codes']['pp3_met']} {s['evidence_codes']['pp3_strength']}; BP4 met "
          f"{s['evidence_codes']['bp4_met']} {s['evidence_codes']['bp4_strength']}; PM2 met "
          f"{s['evidence_codes']['pm2_met']} {s['evidence_codes']['pm2_codes']}")
    print(f"PP3 by assertion {s['evidence_codes']['pp3_met_by_assertion']}; BP4 by assertion "
          f"{s['evidence_codes']['bp4_met_by_assertion']}")
    print(f"ClinVar {TRAIN_SNAPSHOT} state of expert P/LP/B/LB: {s[f'clinvar_{TRAIN_SNAPSHOT}_state_of_expert_PLP_BLB']}")
    print(f'wrote {OUT} and {SUMMARY}')


if __name__ == '__main__':
    main()
