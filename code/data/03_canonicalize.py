#!/usr/bin/env python3
"""
Step 03 - Put every reported protein change on ONE reference protein per gene.

Problem being fixed: ClinVar reports protein changes on the RefSeq transcript given in the variant Name
(normally MANE Select); the Deafness Variation Database (DVD) reports them on its own curated RefSeq
protein; protein-level resources (AlphaFold DB, AlphaMissense proteome table, UniProt features) use the
UniProt canonical isoform. For ~30 panel genes these isoforms differ in length / numbering, so using a
reported position directly on another isoform silently places the substitution on the wrong residue.

Reference protein per gene (R_g): the protein encoded by the RefSeq transcript that ClinVar uses for the
largest number of missense records of that gene in the 2026-04 snapshot (the clinical reporting
transcript). Every variant from every source is re-mapped onto R_g by global pairwise alignment of its
own source protein to R_g. A variant is kept only if
  (i)   the reported reference residue matches its source protein at the reported position,
  (ii)  that position aligns to a residue of R_g, and
  (iii) the aligned R_g residue is identical to the reported reference residue.
A second alignment R_g -> UniProt canonical gives `uniprot_pos` for structure / annotation look-ups.

Outputs (data/sources/):
  - reference_proteins.csv       : gene, reference accession, protein id, sequence, relation to UniProt
  - transcript_to_reference.csv  : per source accession, alignment summary
  - clinvar_<snapshot>_canon.csv, dvd_missense_canon.csv : missense rows + ref_pos, uniprot_pos, variant_id
  - canonicalize_summary.json
"""
import json
from pathlib import Path

import pandas as pd
from Bio import Align

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'data'))
from paths import legacy_root  # noqa: E402

REV_ROOT = Path(__file__).resolve().parents[2]                   # the release folder
SRC = REV_ROOT / 'data/sources'
SNAPSHOTS = ['2025-01', '2025-06', '2026-01', '2026-04']
REF_SNAPSHOT = '2026-04'
AA3TO1 = {'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C', 'Gln': 'Q', 'Glu': 'E', 'Gly': 'G',
          'His': 'H', 'Ile': 'I', 'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P', 'Ser': 'S',
          'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V'}


def make_aligner():
    al = Align.PairwiseAligner()
    al.mode = 'global'
    al.match_score = 2
    al.mismatch_score = -1
    al.open_gap_score = -10
    al.extend_gap_score = -0.5
    al.target_end_gap_score = 0     # isoforms often differ at the termini
    al.query_end_gap_score = 0
    return al


def position_map(src, dst, aligner):
    """{1-based position in src: 1-based position in dst} for aligned columns; None means identity."""
    if src == dst:
        return None
    aln = aligner.align(src, dst)[0]
    pmap = {}
    for (s0, s1), (d0, d1) in zip(*aln.aligned):
        for k in range(s1 - s0):
            pmap[s0 + k + 1] = d0 + k + 1
    return pmap


def canonicalize(df, acc_col, tx, refseq, maps, ref2uni, uniseq):
    ref_pos, uni_pos, status = [], [], []
    for gene, acc, ref3, pos in zip(df.gene, df[acc_col], df.p_ref, df.p_pos):
        aa = AA3TO1.get(ref3)
        pos = int(pos)
        src, ref = tx.get(acc), refseq.get(gene)
        if src is None or ref is None:
            ref_pos.append(pd.NA); uni_pos.append(pd.NA); status.append('no_transcript'); continue
        if pos > len(src) or src[pos - 1] != aa:
            ref_pos.append(pd.NA); uni_pos.append(pd.NA); status.append('ref_mismatch_source'); continue
        pmap = maps[(acc, gene)]
        rpos = pos if pmap is None else pmap.get(pos)
        if rpos is None:
            ref_pos.append(pd.NA); uni_pos.append(pd.NA); status.append('not_in_reference_isoform'); continue
        if rpos > len(ref) or ref[rpos - 1] != aa:
            ref_pos.append(pd.NA); uni_pos.append(pd.NA); status.append('ref_mismatch_reference'); continue
        umap, useq = ref2uni[gene], uniseq.get(gene)
        upos = rpos if umap is None else umap.get(rpos)
        if upos is not None and useq is not None and (upos > len(useq) or useq[upos - 1] != aa):
            upos = None
        ref_pos.append(rpos); uni_pos.append(upos if upos is not None else pd.NA); status.append('ok')
    df = df.copy()
    df['ref_pos'] = pd.array(ref_pos, dtype='Int64')
    df['uniprot_pos'] = pd.array(uni_pos, dtype='Int64')
    df['canon_status'] = status
    df['ref1'] = df.p_ref.map(AA3TO1)
    df['alt1'] = df.p_alt.map(AA3TO1)
    ok = df.canon_status == 'ok'
    df['variant_id'] = pd.NA
    df.loc[ok, 'variant_id'] = (df.loc[ok, 'gene'] + ':' + df.loc[ok, 'ref1']
                                + df.loc[ok, 'ref_pos'].astype(str) + df.loc[ok, 'alt1'])
    return df


def main():
    seqs = pd.read_csv(legacy_root() / 'data/features/gene_sequences.csv')
    uniseq = dict(zip(seqs.gene, seqs.sequence))
    uniid = dict(zip(seqs.gene, seqs.uniprot_id))
    txdf = pd.read_csv(SRC / 'transcript_proteins.csv', dtype=str)
    tx = dict(zip(txdf.accession, txdf.sequence))
    tx_pid = dict(zip(txdf.accession, txdf.protein_id))
    aligner = make_aligner()

    clin = {s: pd.read_csv(SRC / f'clinvar_{s}.csv', dtype=str) for s in SNAPSHOTS}
    clin = {s: d[d.vclass == 'missense'].copy() for s, d in clin.items()}
    dvd = pd.read_csv(SRC / 'dvd_missense.csv', dtype=str)
    dvd = dvd[dvd.vclass == 'missense'].copy()

    # ---- reference protein per gene: dominant ClinVar transcript in the reference snapshot
    cref = clin[REF_SNAPSHOT]
    cref = cref[cref.nm.isin(tx)]
    dom = cref.groupby(['gene', 'nm']).size().reset_index(name='n').sort_values(['gene', 'n'], ascending=[True, False])
    dom = dom.drop_duplicates('gene')
    ref_acc = dict(zip(dom.gene, dom.nm))
    rows, refseq, ref2uni = [], {}, {}
    for gene in sorted(uniseq):
        if gene.startswith('MT-'):
            continue
        acc = ref_acc.get(gene)
        seq = tx[acc] if acc else uniseq[gene]
        refseq[gene] = seq
        umap = position_map(seq, uniseq[gene], aligner)
        ref2uni[gene] = umap
        n_ident = len(seq) if umap is None else sum(1 for s, c in umap.items() if seq[s - 1] == uniseq[gene][c - 1])
        rows.append({'gene': gene, 'uniprot_id': uniid[gene], 'reference_accession': acc or f'UniProt:{uniid[gene]}',
                     'reference_protein_id': tx_pid.get(acc, uniid[gene]) if acc else uniid[gene],
                     'reference_length': len(seq), 'uniprot_length': len(uniseq[gene]),
                     'identical_to_uniprot_canonical': umap is None,
                     'frac_reference_mappable_to_uniprot': round(n_ident / len(seq), 4),
                     'n_clinvar_missense_on_reference': int(dom.set_index('gene').n.get(gene, 0)),
                     'sequence': seq})
    refdf = pd.DataFrame(rows)
    refdf.to_csv(SRC / 'reference_proteins.csv', index=False)
    print(f'reference proteins: {len(refdf)} genes; identical to UniProt canonical: '
          f'{int(refdf.identical_to_uniprot_canonical.sum())}; from UniProt fallback: '
          f'{int(refdf.reference_accession.str.startswith("UniProt").sum())}')
    print(refdf[~refdf.identical_to_uniprot_canonical][['gene', 'reference_accession', 'reference_length',
          'uniprot_length', 'frac_reference_mappable_to_uniprot']].sort_values(
              'frac_reference_mappable_to_uniprot').to_string())

    # ---- source transcript -> reference maps
    pairs = set()
    for d in clin.values():
        pairs |= set(zip(d.nm, d.gene))
    pairs |= set(zip(dvd.np, dvd.gene))
    pairs = {(a, g) for a, g in pairs if isinstance(a, str) and a in tx and g in refseq}
    maps, trows = {}, []
    for acc, gene in sorted(pairs):
        src, ref = tx[acc], refseq[gene]
        pmap = position_map(src, ref, aligner)
        maps[(acc, gene)] = pmap
        n_ident = len(src) if pmap is None else sum(1 for s, c in pmap.items() if src[s - 1] == ref[c - 1])
        shifted = 0 if pmap is None else sum(1 for s, c in pmap.items() if s != c)
        trows.append({'accession': acc, 'gene': gene, 'is_reference': acc == ref_acc.get(gene),
                      'len_source': len(src), 'len_reference': len(ref), 'identical_sequence': pmap is None,
                      'frac_source_identical': round(n_ident / len(src), 4), 'n_positions_renumbered': shifted})
    tmap = pd.DataFrame(trows)
    tmap.to_csv(SRC / 'transcript_to_reference.csv', index=False)
    print(f'{len(tmap)} source transcript-gene pairs; identical to reference: {int(tmap.identical_sequence.sum())}; '
          f'needing renumbering: {int((tmap.n_positions_renumbered > 0).sum())}')

    summary = {'n_genes': int(len(refdf)),
               'n_reference_identical_to_uniprot': int(refdf.identical_to_uniprot_canonical.sum()),
               'n_source_transcripts': int(len(tmap)),
               'n_source_identical_to_reference': int(tmap.identical_sequence.sum())}
    for s, d in clin.items():
        c = canonicalize(d, 'nm', tx, refseq, maps, ref2uni, uniseq)
        c.to_csv(SRC / f'clinvar_{s}_canon.csv', index=False)
        ok = c.canon_status == 'ok'
        summary[f'clinvar_{s}'] = {**c.canon_status.value_counts().to_dict(),
                                   'renumbered': int((ok & (c.ref_pos.astype('float') != c.p_pos.astype(float))).sum()),
                                   'ok_without_uniprot_pos': int((ok & c.uniprot_pos.isna()).sum())}
        print(s, summary[f'clinvar_{s}'])
    c = canonicalize(dvd, 'np', tx, refseq, maps, ref2uni, uniseq)
    c.to_csv(SRC / 'dvd_missense_canon.csv', index=False)
    ok = c.canon_status == 'ok'
    summary['dvd'] = {**c.canon_status.value_counts().to_dict(),
                      'renumbered': int((ok & (c.ref_pos.astype('float') != c.p_pos.astype(float))).sum()),
                      'ok_without_uniprot_pos': int((ok & c.uniprot_pos.isna()).sum())}
    print('dvd', summary['dvd'])
    bad = c[~ok]
    print('dvd failures by gene:', bad.groupby(['gene', 'canon_status']).size().sort_values(ascending=False).head(15).to_dict())
    json.dump(summary, open(SRC / 'canonicalize_summary.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
