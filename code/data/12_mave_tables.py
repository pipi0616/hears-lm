#!/usr/bin/env python3
"""
Step 12 - MaveDB functional scores of hearing-loss genes on reference-protein numbering.

Purpose: an orthogonal (non-ClinVar) validation set. Every multiplexed-assay score set of a panel gene is
re-numbered onto the gene's reference protein so that it can be joined to the other tables by variant_id.

Problem being fixed: the "domainome 1.0" score sets (Beltran et al., abundance PCA of single domains) report
protein positions relative to the ASSAYED DOMAIN FRAGMENT (p.Asn1Ala = first residue of the fragment), not
to the full-length protein; the KCNQ4 / KCNQ1 / KCNE1 sets use full-length numbering. Using the reported
position directly puts most domainome variants on the wrong residue.

Inputs
  - hearing/data/external/mavedb_hhl_scoresets.csv : the 30 score-set URNs of panel genes (gene, urn)
  - MaveDB API  https://api.mavedb.org/api/v1/score-sets/<urn>         (metadata incl. target sequence)
                https://api.mavedb.org/api/v1/score-sets/<urn>/scores  (scores as deposited)
    Both are cached in data/sources/mavedb/ (<urn>.json, <urn>.scores.csv; ':' replaced by '_'); a re-run
    uses the cache and needs no network. If the scores cannot be downloaded, the earlier download in
    hearing/data/external/mavedb_hhl/ is used (same content; it holds no target sequences, so the metadata
    must come from the API or the cache).
  - data/sources/reference_proteins.csv            : reference protein per gene (step 03)

Rules
  1. Target protein = targetGenes[0].targetSequence (protein, or DNA translated in frame 1).
  2. Only single amino-acid substitutions are used (p.Xaa<pos>Yaa with two different standard residues);
     synonymous / wild-type rows, nonsense variants, multi-mutants and anything else are counted and
     skipped. Rows without a score are skipped (no imputation).
  3. Re-numbering = step 03 logic, unchanged: the target protein is aligned to the reference protein with
     `make_aligner` / `position_map`, and `canonicalize` keeps a variant only if (i) the wild-type residue
     in hgvs_pro matches the target sequence at the reported position, (ii) that position aligns to a
     reference residue and (iii) the aligned reference residue is identical.
  4. Scores are copied as deposited (column `score`); nothing is rescaled, averaged or sign-flipped.
     score_sd = deposited `sd`; score_se = deposited `se`, or the DiMSum error estimate `sigma` for the
     domainome sets (see `score_error_column`); n_replicates = deposited `replicates`.
  5. Assay label, score direction and the `primary` flag are curated per score set in CURATION below. The
     direction is taken from the MaveDB description; every quoted sentence is checked to be present
     verbatim (whitespace-normalised) in the API metadata, and the field it was found in is recorded.
     primary = one benchmark per gene / assay: the homozygous KCNQ4 current set, the KCNE1 functional set
     and each domainome set. Not primary: V1/2 sets (not monotonic), co-expression subsets (selected on
     the homozygous phenotype), synonymous-only sets, the test upload, the KCNE1 trafficking sets and the
     62-variant KCNQ1 set.

Outputs (data/annotations/)
  - mave_v8.csv            one row per score set x mapped missense variant:
        variant_id, gene, ref_pos, ref1, alt1, urn, score, score_sd, score_se, n_replicates, category
        (author class, where deposited), hgvs_pro and target_pos (as deposited), primary
  - mave_scoresets_v8.csv  one row per URN: urn, gene, title, assay_type, n_variants_in (single
        substitutions with a score), n_mapped, ref_start, ref_end (fragment on the reference protein),
        offset (ref_pos - reported position), identity, published_date, higher_score_means,
        direction_evidence, primary, then: primary_note, direction_note, direction_evidence_field,
        direction_check (medians nonsense < missense < synonymous where such rows exist; a check, not the
        source of the direction), author_normal_min / _max (range the depositors call normal, where a
        numeric range is deposited), row classes, mapping failures by reason, frac_ref_match_before /
        _after (share of n_variants_in whose wild-type residue equals the reference residue at the
        REPORTED position / at the re-numbered position), alignment details, the UniProt offset declared
        in MaveDB and whether the alignment agrees with it, provenance
  - mave_v8_summary.json   validation counts; overlap and Spearman correlation (>= 10 shared variants)
        between score sets of the same gene

Deterministic given the cache (delete data/sources/mavedb/ to refresh from MaveDB); CPU and network only.
"""
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from Bio.Seq import Seq

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'data'))
from paths import legacy_root  # noqa: E402

REV_ROOT = Path(__file__).resolve().parents[2]                   # the release folder
REV = REV_ROOT
HERE = REV / 'code/data'
SRC = REV / 'data/sources'
CACHE = SRC / 'mavedb'
EXT = legacy_root() / 'data/external'
UNISEQ = legacy_root() / 'data/features/gene_sequences.csv'
OUT_VAR = REV / 'data/annotations/mave_v8.csv'
OUT_SETS = REV / 'data/annotations/mave_scoresets_v8.csv'
SUMMARY = REV / 'data/annotations/mave_v8_summary.json'
API = 'https://api.mavedb.org/api/v1/score-sets/'
HGVS_RE = re.compile(r'^p\.\(?([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2}|=|\*)\)?$')
MIN_N_SPEARMAN = 10

UP, V12 = 'more functional/stable', 'not monotonic (V1/2)'
# Sentences quoted from the MaveDB metadata (verified against the API text at run time).
Q_CURRENT = ['Whole-cell current amplitude scores at +40 mV for all missense SNVs of KCNQ4 were normalized by '
             'whole-cell current of WT KCNQ4 at +40 mV.',
             'Variants with whole-cell current amplitude score less than 0.54 were classified as '
             'current-reduced variants.']
Q_V12 = ['Variants with shifted V1/2 more than +10 mV were classified as LOF.',
         'Variants with shifted V1/2 more than -10 mV were classified as GOF.']
Q_TRAFFIC = ['Next, the weighted averages were linearly transformed so that synonymous variants would have a '
             'median score of 1, and trafficking-null nonsense variants would have a median score of 0.']
Q_FUNCTION = ['Scores were linearly transformed as follows: synonymous variants would have a median score of 1, '
              'and trafficking-null early nonsense variants would have a median score of 0.']
Q_DOMAINOME = ['Unstable domains are degraded by the cell, reducing DHFR fragment concentration and impeding cell '
               'growth in the presence of methotrexate.',
               "Growth rates and growth rate errors were normalized within each protein domain by linearly scaling "
               "the data such that the WT normalized fitness ('score') equals zero, and the 2.5th percentile of the "
               "distribution of growth rates of all missense variants plus the WT equals -1."]
N_GOF_CURRENT = ('scores above 2.08 were classified by the authors as current-increased (abnormal), so the relation '
                 'to function is monotonic only below that value')
N_COEXP = 'variant co-expressed with wild-type KCNQ4 (heterozygous model); subset selected on the homozygous phenotype'


def _c(assay, direction, quotes, primary, why, note=''):
    return {'assay_type': assay, 'higher_score_means': direction, 'quotes': quotes, 'primary': primary,
            'primary_note': why, 'direction_note': note}


P = 'urn:mavedb:'
CURATION = {
    P + '00000094-a-2': _c('KCNQ4 current (homozygous)', UP, Q_CURRENT, True,
                           'all 4,085 missense SNVs; homozygous whole-cell current', N_GOF_CURRENT),
    P + '00000094-a-3': _c('KCNQ4 V1/2 (homozygous)', V12, Q_V12, False, 'V1/2 is not monotonic in function'),
    P + '00000094-a-4': _c('KCNQ4 current (co-expressed with WT; current-reduced subset)', UP, Q_CURRENT, False,
                           'co-expression subset', N_COEXP),
    P + '00000094-a-5': _c('KCNQ4 V1/2 (co-expressed with WT; current-reduced subset)', V12, Q_V12, False,
                           'co-expression subset; V1/2', N_COEXP),
    P + '00000094-a-6': _c('KCNQ4 current (co-expressed with WT; right-shifted subset)', UP, Q_CURRENT, False,
                           'co-expression subset', N_COEXP),
    P + '00000094-a-7': _c('KCNQ4 V1/2 (co-expressed with WT; right-shifted subset)', V12, Q_V12, False,
                           'co-expression subset; V1/2', N_COEXP),
    P + '00000094-a-8': _c('KCNQ4 current (co-expressed with WT; current-enhanced subset)', UP, Q_CURRENT, False,
                           'co-expression subset', N_COEXP),
    P + '00000094-a-9': _c('KCNQ4 V1/2 (co-expressed with WT; current-enhanced subset)', V12, Q_V12, False,
                           'co-expression subset; V1/2', N_COEXP),
    P + '00000094-a-10': _c('KCNQ4 current (co-expressed with WT; left-shifted subset)', UP, Q_CURRENT, False,
                            'co-expression subset', N_COEXP),
    P + '00000094-a-11': _c('KCNQ4 V1/2 (co-expressed with WT; left-shifted subset)', V12, Q_V12, False,
                            'co-expression subset; V1/2', N_COEXP),
    P + '00000094-a-12': _c('KCNQ4 V1/2 (synonymous controls)', V12, Q_V12, False, 'synonymous variants only'),
    P + '00000094-a-13': _c('KCNQ4 current (synonymous controls)', UP, Q_CURRENT, False, 'synonymous variants only'),
    P + '00000094-a-15': _c('KCNQ1 current (homozygous; 62 variants)', UP,
                            ['13. Normalized currents of 62 KCNQ1 missense SNVs in the homozygous state.'] + Q_CURRENT,
                            False, 'small set (62 variants) deposited under the KCNQ4 experiment',
                            'the normalisation / threshold sentences describe the KCNQ4 experiment this set belongs to'),
    P + '00000094-a-1': _c('test upload', 'undetermined (test upload)', ['try_dataset_KCNQ4'], False,
                           'test upload with a single variant'),
    P + '00000674-a-2': _c('KCNE1 trafficking +KCNQ1', UP, Q_TRAFFIC, False,
                           'cell-surface trafficking; the functional set is the primary KCNE1 benchmark'),
    P + '00000674-c-1': _c('KCNE1 trafficking -KCNQ1', UP, Q_TRAFFIC, False,
                           'cell-surface trafficking without KCNQ1; the functional set is the primary KCNE1 benchmark'),
    P + '00000674-b-1': _c('KCNE1 function', UP, Q_FUNCTION, True, 'channel-function selection (KCNQ1-S140G)',
                           'scores above 1.53 were classified by the authors as gain-of-function'),
}
DOMAINOME = _c('domainome abundance', UP, Q_DOMAINOME, True, 'abundance PCA of one domain (domainome 1.0)',
               'measures folding stability / abundance of the isolated domain, not protein function')
N_V12 = ('values are centred on 0 (MaveDB baseline score 0), i.e. the shift of V1/2 relative to wild type in mV; the '
         'authors classify shifts beyond +10 mV as loss-of-function and beyond -10 mV as gain-of-function')
for _cur in CURATION.values():
    if _cur['higher_score_means'] == V12:
        _cur['direction_note'] = '; '.join(x for x in [N_V12, _cur['direction_note']] if x)


def load_step(filename):
    """Import a numbered pipeline step as a module (their file names are not valid identifiers)."""
    spec = importlib.util.spec_from_file_location('step_' + filename.split('_')[0], HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def http_get(url):
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=120, headers={'User-Agent': 'hearslm_revision (shaopeiye55@gmail.com)'})
            if r.status_code == 200 and r.text.strip():
                return r.text
            print(f'  GET {url} -> {r.status_code}, retry {attempt}', file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            print(f'  GET {url} error {e!r}, retry {attempt}', file=sys.stderr)
        time.sleep(2 * (attempt + 1))
    return None


def get_metadata(urn):
    """Score-set record from the MaveDB API (cached). Returns (record, downloaded_now)."""
    path = CACHE / (urn.replace(':', '_') + '.json')
    if path.exists():
        return json.loads(path.read_text()), False
    text = http_get(API + urn)
    if text is None:
        raise RuntimeError(f'MaveDB metadata for {urn} is neither cached nor downloadable; the target sequence '
                           f'is not available locally ({EXT / "mavedb_hhl"} holds scores only)')
    meta = json.loads(text)
    assert meta.get('urn') == urn
    path.write_text(text)
    time.sleep(0.3)
    return meta, True


def get_scores(urn):
    """Scores as deposited: accession, hgvs_nt, hgvs_splice, hgvs_pro, score, <extra columns>.

    Returns (table, origin, downloaded_now); origin is 'mavedb_api' (downloaded now or cached from an
    earlier run) or 'legacy_local_file'.
    """
    path = CACHE / (urn.replace(':', '_') + '.scores.csv')
    downloaded = False
    if not path.exists():
        text = http_get(API + urn + '/scores')
        if text is not None and text.startswith('accession'):
            path.write_text(text)
            downloaded = True
            time.sleep(0.3)
        else:
            legacy = pd.read_csv(EXT / 'mavedb_hhl' / (urn.replace(':', '_') + '.csv'))
            return legacy.drop(columns=['gene', 'scoreset_urn', 'scoreset_title']), 'legacy_local_file', False
    return pd.read_csv(path), 'mavedb_api', downloaded


def compare_with_legacy(urn, scores):
    """Is the earlier local download identical to the scores used (same accessions, hgvs_pro and score)?"""
    path = EXT / 'mavedb_hhl' / (urn.replace(':', '_') + '.csv')
    if not path.exists():
        return None
    old = pd.read_csv(path)
    if len(old) != len(scores) or set(old.accession) != set(scores.accession):
        return False
    m = scores[['accession', 'hgvs_pro', 'score']].merge(old[['accession', 'hgvs_pro', 'score']], on='accession')
    same_score = np.isclose(m.score_x.astype(float), m.score_y.astype(float), rtol=0, atol=1e-9, equal_nan=True)
    return bool((m.hgvs_pro_x == m.hgvs_pro_y).all() and same_score.all())


def target_protein(meta):
    genes = meta['targetGenes']
    assert len(genes) == 1, f'{meta["urn"]}: {len(genes)} target genes (multi-target score sets are not handled)'
    ts = genes[0].get('targetSequence')
    assert ts and ts.get('sequence'), f'{meta["urn"]}: no target sequence (accession-based target)'
    seq, kind = ts['sequence'].strip().upper(), ts['sequenceType'].lower()
    if kind == 'protein':
        return seq, kind
    if kind == 'dna':                                          # hgvs_pro positions = codons of frame 1
        return str(Seq(seq[:len(seq) // 3 * 3]).translate()).rstrip('*'), kind
    raise ValueError(f'{meta["urn"]}: unsupported target sequence type {kind}')


def classify(hgvs_pro, aa3):
    """-> (class, ref3, pos, alt3); class in missense / synonymous / nonsense / other."""
    if not isinstance(hgvs_pro, str):
        return 'other', None, None, None
    if hgvs_pro in ('p.=', 'p.(=)', '_wt'):
        return 'synonymous', None, None, None
    m = HGVS_RE.match(hgvs_pro)
    if not m:
        return 'other', None, None, None                       # multi-mutants, indels, ...
    ref, pos, alt = m.groups()
    if alt == '=' or alt == ref:
        return 'synonymous', ref, int(pos), alt
    if alt in ('Ter', '*'):
        return 'nonsense', ref, int(pos), alt
    if ref in aa3 and alt in aa3:
        return 'missense', ref, int(pos), alt
    return 'other', ref, int(pos), alt


def find_quotes(meta, quotes):
    """Check that every quoted sentence occurs verbatim (whitespace-normalised) in the metadata."""
    exp = meta.get('experiment') or {}
    fields = [('scoreSet.methodText', meta.get('methodText')), ('scoreSet.abstractText', meta.get('abstractText')),
              ('scoreSet.shortDescription', meta.get('shortDescription')), ('scoreSet.title', meta.get('title')),
              ('experiment.methodText', exp.get('methodText')), ('experiment.abstractText', exp.get('abstractText'))]
    norm = [(name, ' '.join((text or '').split())) for name, text in fields]
    found = []
    for q in quotes:
        hit = next((name for name, text in norm if ' '.join(q.split()) in text), None)
        assert hit is not None, f'{meta["urn"]}: quoted sentence not found in the MaveDB metadata: {q!r}'
        found.append(hit)
    return found


def author_normal_range(meta):
    """Score range the depositing authors call 'normal' (MaveDB calibration 'Investigator-provided
    functional classes'), where a numeric range was deposited; (None, None) otherwise."""
    for cal in meta.get('scoreCalibrations') or []:
        if cal.get('title') == 'Investigator-provided functional classes':
            for fc in cal.get('functionalClassifications') or []:
                if fc.get('functionalClassification') == 'normal' and fc.get('range'):
                    return fc['range'][0], fc['range'][1]
    return None, None


def alignment_summary(target, ref, pmap, aligner):
    if pmap is None:
        return {'target_len': len(target), 'ref_len': len(ref), 'target_identical_to_reference': True,
                'ref_start': 1, 'ref_end': len(ref), 'offset': 0, 'n_aligned': len(target),
                'n_identical': len(target), 'identity': 1.0, 'n_exact_occurrences_in_reference': 1,
                'n_optimal_alignments': 1}
    offsets = {r - t for t, r in pmap.items()}
    n_ident = sum(target[t - 1] == ref[r - 1] for t, r in pmap.items())
    try:
        n_opt = len(aligner.align(target, ref))
    except OverflowError:
        n_opt = -1
    return {'target_len': len(target), 'ref_len': len(ref), 'target_identical_to_reference': False,
            'ref_start': min(pmap.values()), 'ref_end': max(pmap.values()),
            'offset': offsets.pop() if len(offsets) == 1 else pd.NA, 'n_aligned': len(pmap),
            'n_identical': n_ident, 'identity': round(n_ident / len(target), 4),
            'n_exact_occurrences_in_reference': sum(ref.startswith(target, i) for i in range(len(ref))),
            'n_optimal_alignments': n_opt}


def spearman(a, b):
    """Spearman correlation = Pearson correlation of average ranks."""
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def urn_key(urn):
    num, letter, n = urn.split(':')[-1].split('-')
    return int(num), letter, int(n)


def jsonable(o):
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


def main():
    s03 = load_step('03_canonicalize.py')
    aa3 = s03.AA3TO1
    CACHE.mkdir(parents=True, exist_ok=True)
    listing = pd.read_csv(EXT / 'mavedb_hhl_scoresets.csv')
    assert listing.urn.is_unique
    refdf = pd.read_csv(SRC / 'reference_proteins.csv')
    refseq = dict(zip(refdf.gene, refdf.sequence))
    refacc = dict(zip(refdf.gene, refdf.reference_accession))
    uni = pd.read_csv(UNISEQ)
    uniseq = dict(zip(uni.gene, uni.sequence))
    uniid = dict(zip(refdf.gene, refdf.uniprot_id))
    aligner = s03.make_aligner()
    ref2uni = {g: s03.position_map(refseq[g], uniseq[g], aligner) for g in sorted(set(listing.gene))}

    set_rows, var_frames, n_downloaded = [], [], 0
    for urn, gene in sorted(zip(listing.urn, listing.gene), key=lambda x: (x[1], urn_key(x[0]))):
        meta, new_meta = get_metadata(urn)
        scores, scores_source, new_scores = get_scores(urn)
        n_downloaded += int(new_meta) + int(new_scores)
        cur = CURATION.get(urn) or (DOMAINOME if meta['title'].endswith('domainome 1.0') else None)
        assert cur is not None, f'{urn} ({meta["title"]}): no curation entry - add assay type / direction / primary'
        assert gene in refseq, f'{urn}: gene {gene} has no reference protein'
        tg = meta['targetGenes'][0]
        assert tg['name'].split()[0] == gene, f'{urn}: target {tg["name"]!r} does not belong to {gene}'
        fields = find_quotes(meta, cur['quotes'])
        target, seq_type = target_protein(meta)
        ref = refseq[gene]
        pmap = s03.position_map(target, ref, aligner)
        aln = alignment_summary(target, ref, pmap, aligner)

        # ---- row classes
        cls = [classify(h, aa3) for h in scores.hgvs_pro]
        d = scores.copy()
        d['vclass'], d['p_ref'], d['p_pos'], d['p_alt'] = zip(*cls) if cls else ([], [], [], [])
        d['score'] = pd.to_numeric(d.score, errors='coerce')
        n_class = d.vclass.value_counts()
        mis_all = d[d.vclass == 'missense']
        mis = mis_all[mis_all.score.notna()].copy()
        assert not mis.hgvs_pro.duplicated().any(), f'{urn}: duplicated hgvs_pro'

        # ---- re-numbering with the step-03 rules
        mis['gene'], mis['urn'] = gene, urn
        mis['p_pos'] = mis.p_pos.astype(int)
        mapped = s03.canonicalize(mis, 'urn', {urn: target}, refseq, {(urn, gene): pmap}, ref2uni, uniseq)
        ok = mapped.canon_status == 'ok'
        before = np.array([p <= len(ref) and ref[p - 1] == aa3[a] for p, a in zip(mis.p_pos, mis.p_ref)], dtype=bool)
        status = mapped.canon_status.value_counts()

        # ---- offset declared in MaveDB (UniProt numbering) vs alignment (reference -> UniProt)
        ext = [x for x in tg.get('externalIdentifiers', []) if x['identifier']['dbName'] == 'UniProt']
        api_offset = ext[0].get('offset') if ext else None
        umap = ref2uni[gene]
        uni_start = aln['ref_start'] if umap is None else umap.get(aln['ref_start'])
        agrees = pd.NA if api_offset is None or pmap is None else bool(uni_start is not None and uni_start - 1 == api_offset)

        # ---- empirical direction check (medians by class; not used to set the direction)
        med = {c: (float(d.score[d.vclass == c].median()) if (d.vclass == c).any() and d.score[d.vclass == c].notna().any()
                   else np.nan) for c in ['missense', 'synonymous', 'nonsense']}
        if cur['higher_score_means'] == UP and not np.isnan(med['nonsense']) and not np.isnan(med['synonymous']):
            check = 'consistent' if med['nonsense'] < med['missense'] < med['synonymous'] else 'INCONSISTENT'
        else:
            check = 'not testable (no nonsense / synonymous rows or non-monotonic score)'

        err_col = 'se' if 'se' in d else ('sigma' if 'sigma' in d else None)
        normal_min, normal_max = author_normal_range(meta)
        pubs = (meta.get('primaryPublicationIdentifiers')
                or (meta.get('experiment') or {}).get('primaryPublicationIdentifiers') or [])
        set_rows.append({
            'urn': urn, 'gene': gene, 'title': meta['title'], 'assay_type': cur['assay_type'],
            'n_variants_in': int(len(mis)), 'n_mapped': int(ok.sum()),
            'ref_start': aln['ref_start'], 'ref_end': aln['ref_end'], 'offset': aln['offset'], 'identity': aln['identity'],
            'published_date': meta.get('publishedDate'),
            'higher_score_means': cur['higher_score_means'], 'direction_evidence': ' | '.join(cur['quotes']),
            'primary': bool(cur['primary']),
            'primary_note': cur['primary_note'], 'direction_note': cur['direction_note'],
            'direction_evidence_field': ' | '.join(fields), 'direction_check': check,
            'median_missense': med['missense'], 'median_synonymous': med['synonymous'], 'median_nonsense': med['nonsense'],
            'author_normal_min': normal_min, 'author_normal_max': normal_max,
            'n_rows': int(len(d)), 'n_missense': int(len(mis_all)),
            'n_synonymous': int(n_class.get('synonymous', 0)), 'n_nonsense': int(n_class.get('nonsense', 0)),
            'n_other': int(n_class.get('other', 0)), 'n_missense_no_score': int(len(mis_all) - len(mis)),
            'n_ref_mismatch_source': int(status.get('ref_mismatch_source', 0)),
            'n_not_in_reference': int(status.get('not_in_reference_isoform', 0)),
            'n_ref_mismatch_reference': int(status.get('ref_mismatch_reference', 0)),
            'frac_ref_match_before': round(float(before.mean()), 4) if len(mis) else np.nan,
            'frac_ref_match_after': round(float(ok.mean()), 4) if len(mis) else np.nan,
            'n_ref_match_before': int(before.sum()),
            'target_name': tg['name'], 'target_sequence_type': seq_type, 'target_len': aln['target_len'],
            'reference_accession': refacc[gene], 'ref_len': aln['ref_len'],
            'target_identical_to_reference': aln['target_identical_to_reference'],
            'n_aligned': aln['n_aligned'], 'n_identical': aln['n_identical'],
            'n_exact_occurrences_in_reference': aln['n_exact_occurrences_in_reference'],
            'n_optimal_alignments': aln['n_optimal_alignments'],
            'uniprot_id': uniid[gene], 'uniprot_start': uni_start,
            'mavedb_uniprot_offset': api_offset, 'offset_agrees_with_mavedb': agrees,
            'score_error_column': err_col, 'score_min': float(mis.score.min()) if len(mis) else np.nan,
            'score_max': float(mis.score.max()) if len(mis) else np.nan,
            'experiment_urn': (meta.get('experiment') or {}).get('urn'),
            'publication': '; '.join(f"{p.get('dbName')}:{p.get('identifier')}" for p in pubs),
            'mavedb_num_variants': meta.get('numVariants'), 'license': (meta.get('license') or {}).get('shortName'),
            'scores_source': scores_source, 'legacy_download_identical': compare_with_legacy(urn, scores),
        })
        assert set_rows[-1]['mavedb_num_variants'] in (None, len(d)), f'{urn}: row count differs from numVariants'

        m = mapped[ok]
        var_frames.append(pd.DataFrame({
            'variant_id': m.variant_id, 'gene': gene, 'ref_pos': m.ref_pos.astype(int), 'ref1': m.ref1, 'alt1': m.alt1,
            'urn': urn, 'score': m.score,
            'score_sd': m['sd'] if 'sd' in m else np.nan,
            'score_se': m[err_col] if err_col else np.nan,
            'n_replicates': m['replicates'] if 'replicates' in m else np.nan,
            'category': m['category'] if 'category' in m else pd.NA,
            'hgvs_pro': m.hgvs_pro, 'target_pos': m.p_pos, 'primary': bool(cur['primary'])}))

    sets = pd.DataFrame(set_rows)
    for col in ['offset_agrees_with_mavedb', 'legacy_download_identical']:
        sets[col] = sets[col].astype('boolean')
    for col in ['offset', 'uniprot_start', 'mavedb_uniprot_offset']:
        sets[col] = sets[col].astype('Int64')
    var = pd.concat([f for f in var_frames if len(f)], ignore_index=True)
    var['n_replicates'] = var.n_replicates.astype('Int64')
    order = {u: i for i, u in enumerate(sets.urn)}
    var = var.assign(_o=var.urn.map(order)).sort_values(['_o', 'ref_pos', 'alt1']).drop(columns='_o').reset_index(drop=True)

    # ---- invariants
    assert not var.duplicated(['urn', 'variant_id']).any()
    assert (var.variant_id == var.gene + ':' + var.ref1 + var.ref_pos.astype(str) + var.alt1).all()
    assert all(refseq[g][p - 1] == a for g, p, a in zip(var.gene, var.ref_pos, var.ref1))
    assert var.score.notna().all()
    assert (var.groupby('urn').size().reindex(sets.urn).fillna(0).astype(int).to_numpy() == sets.n_mapped.to_numpy()).all()
    var.to_csv(OUT_VAR, index=False)
    sets.to_csv(OUT_SETS, index=False)

    # ---- overlaps and Spearman between score sets of the same gene
    pairs = []
    by_urn = {u: g.set_index('variant_id').score for u, g in var.groupby('urn')}
    for gene, g in sets[sets.n_mapped > 0].groupby('gene', sort=True):
        urns = list(g.urn)
        for i in range(len(urns)):
            for j in range(i + 1, len(urns)):
                a, b = by_urn[urns[i]], by_urn[urns[j]]
                common = a.index.intersection(b.index).sort_values()
                rho = spearman(a[common], b[common]) if len(common) >= MIN_N_SPEARMAN else None
                pairs.append({'gene': gene, 'urn_a': urns[i], 'urn_b': urns[j],
                              'assay_a': g.assay_type.iloc[i], 'assay_b': g.assay_type.iloc[j],
                              'n_a': int(len(a)), 'n_b': int(len(b)), 'n_overlap': int(len(common)),
                              'spearman': None if rho is None else round(rho, 4)})
    dom = sets[sets.assay_type == 'domainome abundance']
    full = sets[sets.assay_type != 'domainome abundance']
    summary = {
        'score_sets': int(len(sets)), 'genes': sorted(sets.gene.unique()),
        'rows_all_score_sets': int(sets.n_rows.sum()),
        'single_substitution_rows_with_score (n_variants_in)': int(sets.n_variants_in.sum()),
        'mapped_rows (mave_v8.csv)': int(len(var)), 'unique_variant_ids': int(var.variant_id.nunique()),
        'primary_score_sets': int(sets.primary.sum()), 'primary_rows': int(var.primary.sum()),
        'primary_unique_variant_ids': int(var.variant_id[var.primary].nunique()),
        'reference_residue_match': {
            'all_sets_before': round(float(sets.n_ref_match_before.sum() / sets.n_variants_in.sum()), 4),
            'all_sets_after': round(float(sets.n_mapped.sum() / sets.n_variants_in.sum()), 4),
            'domainome_before': round(float(dom.n_ref_match_before.sum() / dom.n_variants_in.sum()), 4),
            'domainome_after': round(float(dom.n_mapped.sum() / dom.n_variants_in.sum()), 4),
            'full_length_sets_before': round(float(full.n_ref_match_before.sum() / full.n_variants_in.sum()), 4),
            'full_length_sets_after': round(float(full.n_mapped.sum() / full.n_variants_in.sum()), 4)},
        'not_mapped': {'ref_mismatch_source': int(sets.n_ref_mismatch_source.sum()),
                       'not_in_reference_isoform': int(sets.n_not_in_reference.sum()),
                       'ref_mismatch_reference': int(sets.n_ref_mismatch_reference.sum())},
        'skipped_rows': {'synonymous_or_wt': int(sets.n_synonymous.sum()), 'nonsense': int(sets.n_nonsense.sum()),
                         'other': int(sets.n_other.sum()), 'missense_without_score': int(sets.n_missense_no_score.sum())},
        # substitutions of the initiator Met are kept (they are single substitutions) but are start-loss variants
        'start_codon_substitutions_kept_by_urn': var[var.ref_pos == 1].groupby('urn', sort=False).size().to_dict(),
        'domainome_offset_agrees_with_mavedb': f"{int(dom.offset_agrees_with_mavedb.fillna(False).sum())}/{len(dom)}",
        'direction_check': sets.direction_check.value_counts().to_dict(),
        'scores_source': sets.scores_source.value_counts().to_dict(),
        'legacy_download_identical': f"{int(sets.legacy_download_identical.fillna(False).sum())}/{len(sets)}",
        'mavedb_cache': {'dir': str(CACHE), 'files': len(list(CACHE.glob('urn_mavedb_*'))),
                         'retrieved': sorted({time.strftime('%Y-%m-%d', time.localtime(f.stat().st_mtime))
                                              for f in CACHE.glob('urn_mavedb_*')})},
        'per_score_set': {r.urn: {'gene': r.gene, 'assay_type': r.assay_type, 'primary': r.primary,
                                  'n_rows': r.n_rows, 'n_variants_in': r.n_variants_in, 'n_mapped': r.n_mapped,
                                  'frac_ref_match_before': r.frac_ref_match_before,
                                  'frac_ref_match_after': r.frac_ref_match_after,
                                  'ref_range': f'{r.ref_start}-{r.ref_end}', 'offset': r.offset,
                                  'identity': r.identity, 'higher_score_means': r.higher_score_means}
                          for r in sets.itertuples()},
        'same_gene_pairs': pairs,
        'outputs': [str(OUT_VAR), str(OUT_SETS)],
    }
    s = jsonable(summary)
    SUMMARY.write_text(json.dumps(s, indent=2) + '\n')

    # ---- compact print
    print(f'MaveDB files downloaded in this run: {n_downloaded} (the rest from {CACHE})')
    print(f"{s['score_sets']} score sets, {len(s['genes'])} genes; rows {s['rows_all_score_sets']}; single "
          f"substitutions with score {s['single_substitution_rows_with_score (n_variants_in)']}; mapped "
          f"{s['mapped_rows (mave_v8.csv)']} ({s['unique_variant_ids']} unique variant_ids); primary sets "
          f"{s['primary_score_sets']} with {s['primary_rows']} rows")
    print(f"reference-residue match before/after: {s['reference_residue_match']}")
    print(f"not mapped: {s['not_mapped']}; skipped: {s['skipped_rows']}")
    print(f"domainome offset agrees with MaveDB-declared UniProt offset: {s['domainome_offset_agrees_with_mavedb']}; "
          f"direction check: {s['direction_check']}; scores from {s['scores_source']}; legacy download identical "
          f"{s['legacy_download_identical']}")
    print('urn | gene | assay | primary | n_in -> n_mapped | match before -> after | ref range (offset) | identity')
    for r in sets.itertuples():
        off = '?' if pd.isna(r.offset) else f'{int(r.offset):+d}'
        print(f"{r.urn.split(':')[-1]:>14} | {r.gene:6} | {r.assay_type[:44]:44} | {'P' if r.primary else '-'} | "
              f"{r.n_variants_in:>4} -> {r.n_mapped:>4} | {r.frac_ref_match_before:.3f} -> {r.frac_ref_match_after:.3f} | "
              f"{r.ref_start}-{r.ref_end} ({off}) | {r.identity:.4f}")
    print(f'same-gene pairs with >= {MIN_N_SPEARMAN} shared variants (Spearman):')
    for p in pairs:
        if p['spearman'] is not None:
            print(f"  {p['gene']} {p['urn_a'].split(':')[-1]} [{p['assay_a']}] vs {p['urn_b'].split(':')[-1]} "
                  f"[{p['assay_b']}]: n={p['n_overlap']}, rho={p['spearman']:+.3f}")
    print('same-gene pairs without any shared variant (different regions; test upload omitted): '
          + '; '.join(f"{p['gene']} {p['urn_a'].split(':')[-1]}/{p['urn_b'].split(':')[-1]}" for p in pairs
                      if p['n_overlap'] == 0 and min(p['n_a'], p['n_b']) >= MIN_N_SPEARMAN))
    print(f'wrote {OUT_VAR}, {OUT_SETS} and {SUMMARY}')


if __name__ == '__main__':
    main()
