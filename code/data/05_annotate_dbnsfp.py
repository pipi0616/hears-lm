#!/usr/bin/env python3
"""
Step 05 - Annotate every variant of data/variants_v8.csv with dbNSFP predictor scores, conservation scores and
gnomAD allele frequencies, looked up BY GENOMIC COORDINATE (GRCh38) through the MyVariant.info API.

Problem being fixed: the previous pipeline looked scores up by protein-change strings and by identifiers that
did not denote the same variant, which attached scores of other substitutions. Here the ONLY key is the GRCh38
genomic variant of the table (chrom, pos, ref_nt, alt_nt); every returned record is checked against that key
and against the amino-acid change of the variant before a score is used. Nothing is imputed, no row is dropped.

Lookup
  id        "chr<chrom>:g.<pos><ref_nt>><alt_nt>" (HGVS genomic, GRCh38 numbering; the id format is the same as
            for hg19, the assembly is selected with assembly=hg38), e.g. GJB2:M34T -> chr13:g.20189481A>G
  request   POST https://myvariant.info/v1/variant, <= 500 ids per request, assembly=hg38,
            fields = the whole `dbnsfp` object + every top-level gnomAD object of the hg38 index
  probe     before anything else ONE known variant (GJB2 p.Met34Thr) is fetched with fields=all and its JSON key
            structure is printed, because field names differ between dbNSFP versions / MyVariant.info parsers.
            The top-level gnomAD objects are taken from this document and from /v1/metadata/fields.
            NB the field list of /v1/metadata/fields is incomplete (e.g. dbnsfp.cadd.phred and the gnomAD v3
            population fields are served but not listed), so whether a score "exists" is decided from the
            retrieved documents, not from that list.
  politeness sequential requests, 1 s pause, retries with exponential backoff on time-outs / HTTP 429 / 5xx
  cache     data/annotations/_myvariant_cache/ : batch__<sha1>.json.gz (request + raw response, verbatim) and
            metadata__<utc>.json (release metadata, field list and probe document at access time).
            A re-run reads the cache and only queries ids that are not cached; with a complete cache no network
            access happens and the recorded versions / access date remain those of the original queries.

Rows that are not queried (left unannotated, counted): alleles that are not a single A/C/G/T substitution -
multi-nucleotide substitutions (ref_nt / alt_nt longer than 1), IUPAC ambiguity codes, indels.

Rules (also written to data/annotations/dbnsfp_rules.json)
  R1 multi-transcript rule. dbNSFP gives many scores once per Ensembl transcript; MyVariant.info returns a list
     (transcripts without a value are dropped, so the list is NOT aligned with the transcript list) or a scalar
     when a single value is left. Scalars and lists are treated uniformly and the MOST DAMAGING value across
     transcripts is used: max for scores where higher = more deleterious, min where lower = more deleterious
     (SIFT, SIFT4G, ESM1b log-likelihood ratio). The direction of every predictor is in PREDICTORS below.
     Only raw scores are used, never rankscores.
  R2 several amino-acid changes for one genomic variant. dbNSFP has one row per (genomic variant, amino-acid
     change); a variant in overlapping reading frames / genes, or one that is also a splice-site SNV of another
     transcript, has several rows. The MyVariant.info dbNSFP parser keeps ALL fields of the FIRST row and only
     appends the `aa` object of the other rows, i.e. the served scores (and genename, ensembl ids) belong to
     dbnsfp.aa[0] only. Therefore
       dbnsfp_aa_match       = 1 if ANY amino-acid annotation of the record (any dbnsfp.aa row, or any
                               dbnsfp.hgvsp string) has the same ref and alt amino acid as ref1 / alt1;
                               0 if a dbnsfp block exists but none matches; NaN without dbnsfp block
       dbnsfp_score_aa_match = 1 if the FIRST dbnsfp.aa row (the row the scores belong to) has ref1 / alt1
       dbnsfp_aapos_match    = 1 if a matching annotation also has the position ref_pos (numbering check
                               against the gene's reference protein; 0 can be a mere isoform-numbering
                               difference, the scores still belong to the same genomic variant and change)
     When dbnsfp_score_aa_match is 0 the served scores describe ANOTHER amino-acid substitution; the
     substitution-specific predictor columns are then set to NaN (dbnsfp_scores_masked = 1) unless
     MASK_SCORES_OF_OTHER_AA_CHANGE is False. CADD (defined per genomic variant), conservation (per genomic
     position) and allele frequencies (per genomic variant) are kept.
  R3 conservation scores are per genomic position; if several distinct values were returned the maximum would
     be used and the event is counted.
  R4 allele frequencies. af_<object> = global AF of each top-level gnomAD object (<object>.af.af).
     af_<object>_popmax_derived = max AF over the continental groups afr, amr, eas, nfe, sas of that object
     (the groups gnomAD uses for popmax / grpmax; asj, fin, ami, mid, oth are excluded) - DERIVED here because
     the official popmax / grpmax / faf95 fields are not served. af_any = max over the available global AFs,
     NaN if the variant is in no gnomAD source; af_popmax_any_derived likewise; af_present = 1 / 0 (NaN for rows
     that were not queried). AFs are reported whatever the gnomAD QC filter; the filter is given in
     <object>_filter ("PASS" when the record carries no filter flag).
  R5 documents. `found` = 1 if exactly one document with the queried _id was returned. `notfound` answers,
     several documents for one id, an _id different from the query, and blocks whose own coordinates
     (dbnsfp.chrom / hg38.start / ref / alt; <gnomad>.chrom / pos / ref / alt) contradict the query are counted
     and never used.

Outputs (data/annotations/)
  dbnsfp_v8.csv            one row per variant_id (same order as variants_v8.csv): bookkeeping, sc_*, cons_*, af_*
  dbnsfp_rules.json        the rules above, the source field and the direction of every column
  dbnsfp_v8_summary.json   validation summary (coverage overall / per partition, amino-acid match, sanity checks)
                           with the sha256 of the variants_v8.csv it was built from
  annotation_versions.json key "myvariant_dbnsfp_gnomad": dbNSFP / gnomAD release information and access date
Re-run this step whenever variants_v8.csv is regenerated (the outputs follow its rows; cached ids are not queried
again).
"""
import gzip
import hashlib
import io
import json
import math
import os
import re
import sys
import textwrap
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]


def _rel(path):
    """Path relative to the project root when inside it (for provenance records), else absolute."""
    try:
        return Path(path).resolve().relative_to(ROOT)
    except ValueError:
        return Path(path).resolve()
ANN = ROOT / 'data/annotations'
CACHE = ANN / '_myvariant_cache'
# optional overrides (used to annotate other variant tables, e.g. the VUS table, with identical rules):
#   ANNOT_IN=<csv with the columns of variants_v8.csv>  ANNOT_OUT=<output csv>  ANNOT_VERSIONS_KEY=<key>
IN_CSV = Path(os.environ.get('ANNOT_IN', ROOT / 'data/variants_v8.csv'))
OUT_CSV = Path(os.environ.get('ANNOT_OUT', ANN / 'dbnsfp_v8.csv'))
# the rule and version records of the project are only (re)written when the output goes into the project's annotation
# folder; a query written elsewhere (e.g. by the command-line scorer) keeps its records next to its own output
_IN_PROJECT = OUT_CSV.resolve().parent == ANN.resolve()
OUT_RULES = (ANN if _IN_PROJECT else OUT_CSV.parent) / 'dbnsfp_rules.json'
OUT_SUMMARY = OUT_CSV.with_name(OUT_CSV.stem + '_summary.json')
OUT_VERSIONS = (ANN if _IN_PROJECT else OUT_CSV.parent) / 'annotation_versions.json'
VERSIONS_KEY = os.environ.get('ANNOT_VERSIONS_KEY', 'myvariant_dbnsfp_gnomad')
SCRIPT = 'code/data/05_annotate_dbnsfp.py'

API = 'https://myvariant.info/v1'
ASSEMBLY = 'hg38'
BATCH_SIZE = 500            # ids per POST (API maximum is 1000)
PAUSE_S = 1.0               # pause between requests
TIMEOUT_S = 180
MAX_TRIES = 6
BACKOFF_S, BACKOFF_MAX_S = 2.0, 60.0
RETRY_STATUS = {429, 500, 502, 503, 504}
USER_AGENT = 'hearing-v2-revision 05_annotate_dbnsfp.py (python-requests)'
PROBE_ID = 'chr13:g.20189481A>G'            # GJB2 NM_004004.6:c.101T>C p.Met34Thr
MASK_SCORES_OF_OTHER_AA_CHANGE = True       # rule R2
NOT_FOUND_ALERT = 0.03

PARTITIONS = ['fit', 'dev', 'test_db', 'test_time', 'test_gene']
SANITY_PARTITIONS = ['test_time', 'test_gene']
KNOWN_VARIANTS = ['GJB2:M34T', 'GJB2:V37I', 'SLC26A4:L236P', 'MYO7A:R212H', 'KCNQ4:G285S']
CHROMS = [str(i) for i in range(1, 23)] + ['X', 'Y', 'MT']

# column, dbNSFP column, candidate MyVariant.info paths (the first one that carries values is used),
# higher_is_pathogenic, substitution-specific (masked by rule R2), note
PREDICTORS = [
    ('sc_alphamissense', 'AlphaMissense_score', ['dbnsfp.alphamissense.score'], True, True, ''),
    ('sc_revel', 'REVEL_score', ['dbnsfp.revel.score'], True, True, ''),
    ('sc_metarnn', 'MetaRNN_score', ['dbnsfp.metarnn.score'], True, True,
     'uses population allele frequencies as input features'),
    ('sc_bayesdel_addaf', 'BayesDel_addAF_score', ['dbnsfp.bayesdel.add_af.score'], True, True,
     'uses population allele frequency as input feature'),
    ('sc_bayesdel_noaf', 'BayesDel_noAF_score', ['dbnsfp.bayesdel.no_af.score'], True, True, ''),
    ('sc_vest4', 'VEST4_score', ['dbnsfp.vest4.score'], True, True, ''),
    ('sc_primateai', 'PrimateAI_score', ['dbnsfp.primateai.score'], True, True, ''),
    ('sc_esm1b', 'ESM1b_score', ['dbnsfp.esm1b.score'], False, True,
     'log-likelihood ratio; more negative = more deleterious'),
    ('sc_polyphen2_hvar', 'Polyphen2_HVAR_score', ['dbnsfp.polyphen2.hvar.score'], True, True, ''),
    ('sc_mutpred', 'MutPred_score', ['dbnsfp.mutpred.score'], True, True, 'MutPred (v1) general score'),
    ('sc_cadd_phred', 'CADD_phred', ['dbnsfp.cadd.phred'], True, False,
     'defined per genomic variant (GRCh38 model), therefore not masked by rule R2'),
    ('sc_eve', 'EVE_score', ['dbnsfp.eve.score'], True, True, ''),
    ('sc_sift', 'SIFT_score', ['dbnsfp.sift.score'], False, True, 'smaller = more deleterious'),
    ('sc_sift4g', 'SIFT4G_score', ['dbnsfp.sift4g.score'], False, True,
     'smaller = more deleterious; separate column, never mixed with sc_sift'),
    ('sc_clinpred', 'ClinPred_score', ['dbnsfp.clinpred.score'], True, True,
     'uses population allele frequencies as input features'),
    ('sc_varity_r', 'VARITY_R_score', ['dbnsfp.varity.r.score'], True, True, ''),
    ('sc_mpc', 'MPC_score', ['dbnsfp.mpc.score'], True, True, ''),
    ('sc_deogen2', 'DEOGEN2_score', ['dbnsfp.deogen2.score'], True, True, ''),
]
# column, candidate paths (the first one that carries values is used), description
CONSERVATION = [
    ('cons_phylop100', ['dbnsfp.phylop.100way_vertebrate.score'], 'phyloP 100-way vertebrate'),
    ('cons_phylop_mammal', ['dbnsfp.phylop.470way_mammalian.score', 'dbnsfp.phylop.30way_mammalian.score'],
     'phyloP mammalian (470-way; 30-way in dbNSFP < 4.4)'),
    ('cons_phylop_primate', ['dbnsfp.phylop.17way_primate.score'], 'phyloP 17-way primate'),
    ('cons_phastcons100', ['dbnsfp.phastcons.100way_vertebrate.score'], 'phastCons 100-way vertebrate'),
    ('cons_gerp_rs', ['dbnsfp.gerp++.rs'], 'GERP++ rejected substitutions'),
]
CANDIDATES = {p[0]: p[2] for p in PREDICTORS}
CANDIDATES.update({c[0]: c[1] for c in CONSERVATION})
USE_MAX = {p[0]: p[3] for p in PREDICTORS}              # rule R1: max if higher = pathogenic, else min
USE_MAX.update({c[0]: True for c in CONSERVATION})      # rule R3
AA_SPECIFIC = {p[0]: p[4] for p in PREDICTORS}          # rule R2
POPMAX_GROUPS = ['afr', 'amr', 'eas', 'nfe', 'sas']

# Which gnomAD release is behind each top-level gnomAD object of the hg38 index. The API metadata gives ONE
# version string for the whole "gnomad" source ("2.1.1"); the split per object is documented in the uploader of
# the MyVariant.info hub and is checked at run time against the data (largest AN, population keys).
GNOMAD_RELEASE = {
    'gnomad_exome': {'release': 'gnomAD v2.1.1 exomes, lifted over to GRCh38 (125,748 exomes)', 'max_an': 2 * 125748},
    'gnomad_genome': {'release': 'gnomAD v3.1.1 genomes, GRCh38 (76,156 genomes)', 'max_an': 2 * 76156},
}
GNOMAD_RELEASE_BASIS = (
    'MyVariant.info hub uploader, module docstring "Versions of data" (gnomAD_exome_hg38: v2.1.1 => '
    '2.1.1/exomes/liftover_grch38; gnomAD_genome_hg38: v3.1.1 => 3.1.1/genomes): https://github.com/biothings/'
    'myvariant.info/blob/master/src/hub/dataload/sources/gnomad/gnomad_upload.py (last changed by commit b0f5003, '
    '2021-06-10; consulted 2026-10-06). The parsers of that hub read AC / AN / AF / nhomalt per population only, '
    'not popmax / grpmax / faf95; a record without FILTER flag (PASS) is stored without `filter` field.')
DBNSFP_PARSER_BASIS = (
    'MyVariant.info dbNSFP parser named in the API metadata (src.dbnsfp.code): https://github.com/biothings/'
    'myvariant.info/blob/1c767a2/src/hub/dataload/sources/dbnsfp/dbnsfp_parser_48a_v1.py (consulted 2026-10-06): '
    'no gnomAD column is mapped; ";"-separated per-transcript values are split, missing entries dropped and a '
    'single remaining value returned as scalar; rows with the same genomic id keep the fields of the first row '
    'and only collect the `aa` objects of all rows (function load_file).')

AA3 = {'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C', 'Gln': 'Q', 'Glu': 'E', 'Gly': 'G', 'His': 'H',
       'Ile': 'I', 'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P', 'Ser': 'S', 'Thr': 'T', 'Trp': 'W',
       'Tyr': 'Y', 'Val': 'V', 'Sec': 'U', 'Ter': 'X', '*': 'X'}
HGVSP_RE = re.compile(r'p\.([A-Z][a-z]{2}|[A-Z])(\d+)([A-Z][a-z]{2}|[A-Z*])$')


# ----------------------------------------------------------------------------------------------- small helpers
def now_utc():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def now_local_date():
    return datetime.now().astimezone().strftime('%Y-%m-%d')


def write_json(path, obj, gz=False):
    """Atomic write (temporary file + rename): an interrupted run never leaves a truncated file."""
    tmp = path.with_name(path.name + '.tmp')
    if gz:
        with gzip.open(tmp, 'wt', encoding='utf-8') as fh:
            json.dump(obj, fh)
    else:
        with open(tmp, 'w', encoding='utf-8') as fh:
            json.dump(obj, fh, indent=1, allow_nan=False)
    os.replace(tmp, path)


def read_json(path):
    opener = gzip.open if path.name.endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as fh:
        return json.load(fh)


def clean(o):
    """numpy / pandas scalars -> plain Python, NaN -> None (strict JSON)."""
    if isinstance(o, dict):
        return {str(k): clean(x) for k, x in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [clean(x) for x in o]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(o) else None
    if o is pd.NA:
        return None
    return o


def flatten(o):
    if isinstance(o, list):
        for x in o:
            yield from flatten(x)
    else:
        yield o


def leaves(obj, path):
    """All values found at the dotted `path`. MyVariant.info returns a scalar where one value exists and a list
    otherwise, and an object may itself be a list of objects - lists are entered at every level."""
    cur = [obj]
    for key in path.split('.'):
        cur = [o[key] for o in flatten(cur) if isinstance(o, dict) and key in o]
    return [x for x in flatten(cur) if x is not None]


def numbers(values):
    """(finite numbers, count of values that are not finite numbers) - nothing is coerced."""
    good = [float(x) for x in values if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)]
    return good, len(values) - len(good)


def as_list(x):
    return [] if x is None else list(flatten(x)) if isinstance(x, list) else [x]


# ------------------------------------------------------------------------------------------------------- HTTP
def http_json(session, method, url, **kw):
    """One API call with polite retries: exponential backoff (2, 4, 8 ... <= 60 s; a numeric Retry-After header
    is honoured) on connection errors, time-outs, HTTP 429 / 5xx and undecodable bodies."""
    err = 'no attempt made'
    for attempt in range(1, MAX_TRIES + 1):
        wait = min(BACKOFF_S * 2 ** (attempt - 1), BACKOFF_MAX_S)
        try:
            r = session.request(method, url, timeout=TIMEOUT_S, **kw)
            if r.status_code == 200:
                return r.json(), attempt
            err = f'HTTP {r.status_code}: {r.text[:200]!r}'
            if r.status_code not in RETRY_STATUS:
                break
            if r.headers.get('Retry-After', '').isdigit():
                wait = max(wait, float(r.headers['Retry-After']))
        except (requests.RequestException, ValueError) as e:
            err = repr(e)
        if attempt < MAX_TRIES:
            print(f'    {method} {url}: {err}; retry {attempt}/{MAX_TRIES - 1} in {wait:.0f} s', file=sys.stderr,
                  flush=True)
            time.sleep(wait)
    raise RuntimeError(f'{method} {url} failed: {err}')


# ------------------------------------------------------------------------- server state: metadata, fields, probe
def fetch_snapshot(session):
    """State of the server at access time: release metadata of the hg38 index (and of the default endpoint), the
    field list of the index, and ONE known variant with every field (the probe)."""
    urls = {'metadata_hg38': f'{API}/metadata?assembly={ASSEMBLY}', 'metadata_default': f'{API}/metadata',
            'fields_hg38': f'{API}/metadata/fields?assembly={ASSEMBLY}',
            'probe': f'{API}/variant/{PROBE_ID}?assembly={ASSEMBLY}&fields=all'}
    snap = {'fetched_utc': now_utc(), 'fetched_local_date': now_local_date(), 'urls': urls, 'probe_id': PROBE_ID}
    for key in ('metadata_hg38', 'metadata_default', 'fields_hg38'):
        snap[key], _ = http_json(session, 'GET', urls[key])
        time.sleep(PAUSE_S)
    doc, _ = http_json(session, 'GET', f'{API}/variant/{PROBE_ID}', params={'assembly': ASSEMBLY, 'fields': 'all'})
    snap['probe'] = doc[0] if isinstance(doc, list) and doc else doc
    time.sleep(PAUSE_S)
    path = CACHE / f"metadata__{re.sub('[^0-9TZ]', '', snap['fetched_utc'])}.json"
    write_json(path, snap)
    return dict(snap, _file=path.name, _fetched_this_run=True)


def latest_snapshot():
    files = sorted(CACHE.glob('metadata__*.json'))
    return dict(read_json(files[-1]), _file=files[-1].name, _fetched_this_run=False) if files else None


def server_info(snap):
    m = snap['metadata_hg38']
    src = m.get('src', {})
    return {'build_version': m.get('build_version'), 'build_date': m.get('build_date'),
            'dbnsfp_version': src.get('dbnsfp', {}).get('version'),
            'gnomad_version_string': src.get('gnomad', {}).get('version'), 'metadata_file': snap['_file']}


def gnomad_objects(snap):
    """Top-level gnomAD objects: union of the index field list and of the keys of the probe document."""
    names = {k.split('.')[0] for k in snap['fields_hg38']} | set(snap['probe'])
    return sorted(n for n in names if n.lower().startswith('gnomad'))


def request_fields(snap):
    if 'dbnsfp' not in snap['probe']:
        raise SystemExit(f'the probe {PROBE_ID} has no `dbnsfp` object in the {ASSEMBLY} index of MyVariant.info')
    return ','.join(['dbnsfp'] + gnomad_objects(snap))


def structure(o, path=''):
    """(dotted path, type, example) for every leaf; a list of objects is described through its first element."""
    if isinstance(o, dict):
        for k, x in o.items():
            yield from structure(x, f'{path}.{k}' if path else k)
    elif isinstance(o, list) and o and all(isinstance(x, dict) for x in o):
        yield path, f'list[{len(o)}] object', ''
        yield from structure(o[0], path + '[]')
    elif isinstance(o, list):
        kinds = '/'.join(sorted({type(x).__name__ for x in o})) or 'empty'
        yield path, f'list[{len(o)}] {kinds}', json.dumps(o)[:72]
    else:
        yield path, type(o).__name__, json.dumps(o)[:72]


def print_probe(snap):
    """The JSON key structure really served for one known variant (full for dbnsfp, compact for gnomAD)."""
    doc = snap['probe']
    print(f"\n== probe {snap['probe_id']} (GJB2 p.Met34Thr), assembly={ASSEMBLY}, fields=all, fetched {snap['fetched_utc']}")
    print(f'top-level keys: {sorted(doc)}')
    not_listed = []
    for top in sorted(doc):
        if top != 'dbnsfp' and not top.lower().startswith('gnomad'):
            continue
        rows = list(structure(doc[top], top))
        not_listed += [p.replace('[]', '') for p, _, _ in rows if p.replace('[]', '') not in snap['fields_hg38']]
        print(f'-- {top}: {len(rows)} leaf fields')
        groups = defaultdict(list)
        for row in rows:
            groups[row[0].rsplit('.', 1)[0]].append(row)
        done = set()
        for pth, typ, ex in rows:
            parent = pth.rsplit('.', 1)[0]
            if top != 'dbnsfp' and len(groups[parent]) > 12:    # per-population gnomAD blocks: one compact entry
                if parent not in done:
                    done.add(parent)
                    g = groups[parent]
                    kinds = sorted({x[1] for x in g})
                    print(f"  {parent}.*  {len(g)} keys ({kinds[0] if len(kinds) == 1 else 'mixed types'}), "
                          f'e.g. {g[0][0]} = {g[0][2]}')
                    print(textwrap.fill(', '.join(x[0].rsplit('.', 1)[1] for x in g), 116, initial_indent='      ',
                                        subsequent_indent='      '))
            else:
                print(f'  {pth:<50} {typ:<17} {ex}')
    print(f'-- {len(not_listed)} of these leaf fields are NOT in /metadata/fields (list incomplete), e.g. '
          f"{[p for p in not_listed if not p.endswith('_license')][:6]}")


# ---------------------------------------------------------------------------------------------- variant table
def classify_allele(chrom, pos, ref, alt):
    """'' for a queryable single-nucleotide substitution, otherwise the reason why the row is not queried."""
    if chrom not in CHROMS or not pos.isdigit():
        return 'invalid_coordinate'
    if not ref or not alt or re.search('[^ACGT]', ref + alt):
        return 'ambiguous_or_invalid_allele'
    if len(ref) == 1 and len(alt) == 1:
        return 'ref_equals_alt' if ref == alt else ''
    return 'multi_nucleotide_substitution' if len(ref) == len(alt) else 'indel'


def differs_at_one_base(ref, alt):
    """True if a longer allele pair differs at exactly one base (it would be an SNV written with context)."""
    return len(ref) == len(alt) > 1 and sum(a != b for a, b in zip(ref, alt)) == 1


def load_variants():
    """The variant table and its fingerprint (the table is regenerated by step 04; the fingerprint written to the
    summary tells which version the annotation table belongs to). The bytes that are hashed are the bytes parsed."""
    raw = IN_CSV.read_bytes()
    v = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False)
    need = ['variant_id', 'gene', 'ref_pos', 'ref1', 'alt1', 'chrom', 'pos', 'ref_nt', 'alt_nt', 'partition', 'y']
    missing = [c for c in need if c not in v.columns]
    if missing or not v.variant_id.is_unique:
        raise SystemExit(f'{IN_CSV}: missing columns {missing} or duplicated variant_id')
    v['chrom'] = v.chrom.str.strip().str.replace(r'^chr', '', regex=True)
    v['pos'] = v.pos.str.strip()
    v['ref_nt'], v['alt_nt'] = v.ref_nt.str.strip().str.upper(), v.alt_nt.str.strip().str.upper()
    v['skip_reason'] = [classify_allele(*t) for t in zip(v.chrom, v.pos, v.ref_nt, v.alt_nt)]
    v['hgvs_id'] = np.where(v.skip_reason == '', 'chr' + v.chrom + ':g.' + v.pos + v.ref_nt + '>' + v.alt_nt, '')
    info = {'file': str(_rel(IN_CSV)), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw),
            'modified_utc': datetime.fromtimestamp(IN_CSV.stat().st_mtime, timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
            'n_rows': int(len(v))}
    return v, info


# ------------------------------------------------------------------------------------- extraction of one record
def aa1(s):
    return AA3.get(s, s)


def aa_annotations(db, stats):
    """rows  = one (set of (ref, alt) pairs, [positions]) per dbNSFP row, in file order - rows[0] is the row all
               served scores belong to (rule R2); a splice-site row has no pair;
       hgvsp = {(ref, pos, alt)} parsed from dbnsfp.hgvsp (HGVSp_ANNOVAR / snpEff / VEP columns of that first
               row). These strings can also name the change of another reading frame, so they count for
               dbnsfp_aa_match but never decide which change the scores belong to."""
    aa = db.get('aa')
    rows = []
    for r in (aa if isinstance(aa, list) else [aa]):
        if not isinstance(r, dict):
            continue
        refs, alts = as_list(r.get('ref')), as_list(r.get('alt'))
        if not refs and not alts:
            stats['aa_rows_without_amino_acid_change'] += 1     # splice-site row of dbNSFP (aapos = -1)
        elif len(refs) != 1 or len(alts) != 1:
            stats['aa_rows_with_non_scalar_ref_or_alt'] += 1
        pairs = set(zip(refs, alts)) if len(refs) == len(alts) else {(a, b) for a in refs for b in alts}
        rows.append((pairs, [p for p in as_list(r.get('pos')) if isinstance(p, int) and not isinstance(p, bool)]))
    hg = set()
    for s in as_list(db.get('hgvsp')):
        m = HGVSP_RE.search(str(s))
        if m:
            hg.add((aa1(m.group(1)), int(m.group(2)), aa1(m.group(3))))
    return rows, hg


def block_coordinates_ok(chrom, pos, ref, alt, v):
    """True / False if the block's own coordinates agree / disagree with the query, None if it carries none."""
    if chrom is None or pos is None or ref is None or alt is None:
        return None
    return str(chrom) == v['chrom'] and str(pos) == v['pos'] and str(ref).upper() == v['ref_nt'] \
        and str(alt).upper() == v['alt_nt']


def extract(hits, v, gobj, stats, af_keys):
    """All values of one variant from the result objects returned for its id (rules R1-R5). Scores are stored
    under '<column>@<path>' for every candidate path; the path used per column is chosen afterwards."""
    rec = {'found': 0, 'has_dbnsfp': 0, 'dbnsfp_scores_masked': 0, 'af_present': 0}
    docs = [h for h in hits if isinstance(h, dict) and not h.get('notfound')]
    if not docs:
        stats['notfound'] += 1
        return rec
    if len(docs) > 1:
        stats['ids_with_several_documents'] += 1
    exact = [d for d in docs if d.get('_id') == v['hgvs_id']]
    if len(exact) != 1:     # never guess between documents
        stats['ids_without_unique_exact_document'] += 1
        rec['doc_problem'] = f"{len(docs)} document(s) with _id {sorted(str(d.get('_id')) for d in docs)}"
        return rec
    doc = exact[0]
    rec['found'] = 1

    # ---- dbNSFP block
    db = doc.get('dbnsfp')
    if isinstance(db, list):
        stats['dbnsfp_block_is_list'] += 1
        db = db[0] if len(db) == 1 else None
    if isinstance(db, dict):
        hg38 = db.get('hg38')
        ok = block_coordinates_ok(db.get('chrom'), hg38.get('start') if isinstance(hg38, dict) else None,
                                  db.get('ref'), db.get('alt'), v)
        stats['dbnsfp_coordinates_' + {True: 'agree', False: 'CONTRADICT', None: 'not_given'}[ok]] += 1
        if ok is False:
            db = None
    if isinstance(db, dict):
        rec['has_dbnsfp'] = 1
        rows, hg = aa_annotations(db, stats)
        target, ref_pos = (v['ref1'], v['alt1']), int(v['ref_pos'])
        hg_pairs = {(r, a) for r, _, a in hg}
        rec['dbnsfp_aa_match'] = int(any(target in prs for prs, _ in rows) or target in hg_pairs)
        rec['dbnsfp_score_aa_match'] = int(target in (rows[0][0] if rows else hg_pairs))
        rec['dbnsfp_aapos_match'] = int(any(target in prs and ref_pos in pos for prs, pos in rows)
                                        or (v['ref1'], ref_pos, v['alt1']) in hg)
        rec['dbnsfp_n_aa_rows'] = len(rows)
        rec['dbnsfp_aa_annot'] = '|'.join(
            '/'.join(f'{r}{p}{a}' for r, a in sorted(prs) for p in (list(dict.fromkeys(pos)) or ['?'])) or 'noAA'
            for prs, pos in rows)
        rec['hgvsp_strings'] = ';'.join(sorted(str(s) for s in as_list(db.get('hgvsp'))))
        if len(rows) > 1:
            stats['dbnsfp_records_with_several_aa_rows'] += 1
        if rows and hg:
            stats['hgvsp_checked_against_first_aa_row'] += 1
            if not hg_pairs <= rows[0][0]:
                stats['hgvsp_NOT_consistent_with_first_aa_row'] += 1
        mask = MASK_SCORES_OF_OTHER_AA_CHANGE and rec['dbnsfp_score_aa_match'] == 0
        rec['dbnsfp_scores_masked'] = int(mask)
        for col, paths in CANDIDATES.items():
            for path in paths:
                key = f'{col}@{path}'
                vals, bad = numbers(leaves(doc, path))
                stats[f'nonnumeric::{key}'] += bad
                if not vals:
                    continue
                stats[f'n::{key}'] += 1
                stats[f'shape::{key}::' + ('one' if len(vals) == 1 else 'several_identical'
                                           if len(set(vals)) == 1 else 'several_distinct')] += 1
                if mask and AA_SPECIFIC.get(col, False):
                    stats[f'masked::{key}'] += 1
                    continue
                rec[key] = max(vals) if USE_MAX[col] else min(vals)

    # ---- gnomAD objects
    afs, popmaxes = [], []
    for obj in gobj:
        g = doc.get(obj)
        if g is None:
            continue
        if isinstance(g, list):
            stats[f'{obj}_block_is_list'] += 1
        coords = [block_coordinates_ok(x.get('chrom'), x.get('pos'), x.get('ref'), x.get('alt'), v)
                  for x in as_list(g) if isinstance(x, dict)]
        if any(c is False for c in coords):
            stats[f'{obj}_coordinates_CONTRADICT'] += 1
            continue
        stats[f'{obj}_coordinates_' + ('agree' if coords and all(coords) else 'not_given')] += 1
        stats[f'{obj}_records'] += 1
        flt = sorted({str(x) for x in leaves(doc, f'{obj}.filter')})
        rec[f'{obj}_filter'] = ';'.join(flt) if flt else 'PASS'
        af, bad = numbers(leaves(doc, f'{obj}.af.af'))
        stats[f'nonnumeric::af_{obj}'] += bad
        if af:
            stats[f'{obj}_several_distinct_global_af'] += len(set(af)) > 1
            rec[f'af_{obj}'] = max(af)
            afs.append(max(af))
        else:
            stats[f'{obj}_records_without_global_af'] += 1
        pm = [x for grp in POPMAX_GROUPS for x in numbers(leaves(doc, f'{obj}.af.af_{grp}'))[0]]
        if pm:
            rec[f'af_{obj}_popmax_derived'] = max(pm)
            popmaxes.append(max(pm))
        official = numbers(leaves(doc, f'{obj}.af.af_popmax'))[0]
        if official:
            rec[f'af_{obj}_popmax_official'] = max(official)
        an = numbers(leaves(doc, f'{obj}.an.an'))[0]
        if an:
            stats[f'max_an::{obj}'] = max(stats[f'max_an::{obj}'], int(max(an)))
        for x in as_list(g):
            if isinstance(x, dict) and isinstance(x.get('af'), dict):
                af_keys[obj].update(x['af'])
    if afs:
        rec['af_any'], rec['af_present'] = max(afs), 1
    if popmaxes:
        rec['af_popmax_any_derived'] = max(popmaxes)
    return rec


# ------------------------------------------------------------------------------------------- cache and queries
def query_batch(session, ids, fields, server):
    resp, attempts = http_json(session, 'POST', f'{API}/variant',
                               data={'ids': ','.join(ids), 'assembly': ASSEMBLY, 'fields': fields})
    answered = {h.get('query') for h in resp if isinstance(h, dict)} if isinstance(resp, list) else set()
    if not set(ids) <= answered:
        raise RuntimeError(f'{len(set(ids) - answered)} id(s) of a batch were not answered at all')
    key = json.dumps({'assembly': ASSEMBLY, 'fields': fields, 'ids': ids}, sort_keys=True)
    path = CACHE / f"batch__{hashlib.sha1(key.encode()).hexdigest()[:16]}.json.gz"
    batch = {'request': {'method': 'POST', 'url': f'{API}/variant', 'assembly': ASSEMBLY, 'fields': fields,
                         'ids': ids},
             'fetched_utc': now_utc(), 'fetched_local_date': now_local_date(), 'attempts': attempts,
             'server': server, 'response': resp}
    write_json(path, batch, gz=True)
    return dict(batch, _file=path.name)


class Annotator:
    """Collects one extracted record per variant_id from cached and fresh batches."""

    def __init__(self, rows_by_id, snap):
        self.rows_by_id, self.snap = rows_by_id, snap
        self.gobj, self.fields = gnomad_objects(snap), request_fields(snap)
        self.records, self.stats, self.used = {}, Counter(), []
        self.af_keys = {o: set() for o in self.gobj}

    def pending(self, ids):
        return [i for i in ids if not all(r['variant_id'] in self.records for r in self.rows_by_id[i])]

    def ingest(self, batch):
        by_query = defaultdict(list)
        for h in batch['response']:
            if isinstance(h, dict) and 'query' in h:
                by_query[h['query']].append(h)
        n = 0
        for q in batch['request']['ids']:
            if q in by_query and q in self.rows_by_id and self.pending([q]):
                for r in self.rows_by_id[q]:
                    self.records[r['variant_id']] = extract(by_query[q], r, self.gobj, self.stats, self.af_keys)
                n += 1
        if n:
            self.used.append({'file': batch['_file'], 'fetched_utc': batch['fetched_utc'],
                              'fetched_local_date': batch.get('fetched_local_date'), 'n_ids_used': n,
                              'server': batch.get('server', {})})

    def read_cache(self):
        batches = []
        for f in sorted(CACHE.glob('batch__*.json.gz')):
            b = read_json(f)
            if b['request'].get('assembly') == ASSEMBLY and b['request'].get('fields') == self.fields:
                batches.append(dict(b, _file=f.name))
        for b in sorted(batches, key=lambda b: b['fetched_utc'], reverse=True):     # newest answer wins
            self.ingest(b)


def annotate(session, rows_by_id, ids):
    snap = latest_snapshot() or fetch_snapshot(session)
    print_probe(snap)
    ann = Annotator(rows_by_id, snap)
    ann.read_cache()
    todo = ann.pending(ids)
    print(f'\ncache {_rel(CACHE)}: {len(ids) - len(todo)} ids answered from cache, {len(todo)} to query '
          f'(fields={ann.fields})')
    if todo and not snap['_fetched_this_run']:      # the server may have been rebuilt since the cached snapshot
        new = fetch_snapshot(session)
        if server_info(new)['build_version'] != server_info(snap)['build_version']:
            print(f"  WARNING: server build changed {server_info(snap)['build_version']} -> "
                  f"{server_info(new)['build_version']}; the cache mixes builds (recorded per batch)")
        if request_fields(new) != ann.fields:
            print('  WARNING: the set of gnomAD objects changed; every id is queried again with the new fields')
            ann = Annotator(rows_by_id, new)
            ann.read_cache()
            todo = ann.pending(ids)
        ann.snap = snap = new
    chunks = [todo[i:i + BATCH_SIZE] for i in range(0, len(todo), BATCH_SIZE)]
    for k, chunk in enumerate(chunks, 1):
        batch = query_batch(session, chunk, ann.fields, server_info(snap))
        ann.ingest(batch)
        print(f"  batch {k}/{len(chunks)}: {len(chunk)} ids, {batch['attempts']} attempt(s) -> {batch['_file']}",
              flush=True)
        time.sleep(PAUSE_S)
    return ann


# ---------------------------------------------------------------------------------------------- sanity checks
def auroc(y, s):
    """Mann-Whitney AUROC (average ranks for ties); s must be oriented so that higher = pathogenic."""
    y, s = np.asarray(y, dtype=int), np.asarray(s, dtype=float)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float('nan')
    r = pd.Series(s).rank(method='average').to_numpy()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def spearman(a, b):
    m = a.notna() & b.notna()
    if m.sum() < 3:
        return float('nan'), int(m.sum())
    return float(np.corrcoef(a[m].rank().to_numpy(), b[m].rank().to_numpy())[0, 1]), int(m.sum())


def rate(num, den):
    d = int(den.sum())
    return {'n': int((num & den).sum()), 'of': d, 'frac': round(float((num & den).sum() / d), 4) if d else None}


def f3(x):
    return f'{x:>10.3f}' if isinstance(x, float) and math.isfinite(x) else f"{'nan':>10}"


# -------------------------------------------------------------------------------------------------------- main
def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers['User-Agent'] = USER_AGENT

    v, input_info = load_variants()
    queried = v.skip_reason == ''
    rows_by_id = defaultdict(list)
    for r in v[queried].to_dict('records'):
        rows_by_id[r['hgvs_id']].append(r)
    ids = sorted(rows_by_id, key=lambda i: (CHROMS.index(rows_by_id[i][0]['chrom']), int(rows_by_id[i][0]['pos']), i))
    print(f"{input_info['file']} (sha256 {input_info['sha256'][:12]}, modified {input_info['modified_utc']}): {len(v)} "
          f'variants; {int(queried.sum())} single-nucleotide substitutions -> {len(ids)} unique ids; not queried: '
          f'{v.skip_reason[~queried].value_counts().to_dict()}')

    ann = annotate(session, rows_by_id, ids)
    snap, stats, gobj, used = ann.snap, ann.stats, ann.gobj, ann.used
    listed = set(snap['fields_hg38'])

    # ---- which candidate path carries values (decided from the retrieved documents)
    path_of = {c: next((p for p in paths if stats[f'n::{c}@{p}'] > 0), None) for c, paths in CANDIDATES.items()}
    pred_cols = [p[0] for p in PREDICTORS if path_of[p[0]]]
    cons_cols = [c[0] for c in CONSERVATION if path_of[c[0]]]
    pred_absent = [p[0] for p in PREDICTORS if not path_of[p[0]]]
    cons_absent = [c[0] for c in CONSERVATION if not path_of[c[0]]]

    # ---- output table: one row per variant_id, input order
    recs = pd.DataFrame([ann.records.get(i, {}) for i in v.variant_id], index=v.index)
    out = v[['variant_id', 'hgvs_id', 'skip_reason']].copy()
    out['queried'] = queried.astype(int)
    book = ['found', 'has_dbnsfp', 'dbnsfp_aa_match', 'dbnsfp_score_aa_match', 'dbnsfp_aapos_match',
            'dbnsfp_n_aa_rows', 'dbnsfp_aa_annot', 'dbnsfp_scores_masked']
    official = [c for c in (f'af_{o}_popmax_official' for o in gobj) if c in recs and recs[c].notna().any()]
    af_cols = [f'af_{o}' for o in gobj] + [f'af_{o}_popmax_derived' for o in gobj] + official \
        + ['af_any', 'af_popmax_any_derived', 'af_present']
    filt_cols = [f'{o}_filter' for o in gobj]
    for c in book + af_cols + filt_cols + ['hgvsp_strings', 'doc_problem']:
        out[c] = recs[c] if c in recs else np.nan
    for c in pred_cols + cons_cols:
        key = f'{c}@{path_of[c]}'
        out[c] = recs[key].astype(float) if key in recs else np.nan     # absent key: every value was masked
    for c in ['found', 'has_dbnsfp', 'dbnsfp_scores_masked']:
        out[c] = out[c].fillna(0).astype(int)
    for c in af_cols:
        out[c] = out[c].astype(float)
    out.loc[~queried, 'af_present'] = np.nan                # not queried: presence in gnomAD is unknown
    for c in ['dbnsfp_aa_match', 'dbnsfp_score_aa_match', 'dbnsfp_aapos_match', 'dbnsfp_n_aa_rows', 'af_present']:
        out[c] = out[c].astype(float).astype('Int64')
    assert len(out) == len(v) and (out.variant_id == v.variant_id).all()
    assert int((out.found == 1).sum()) + stats['notfound'] + stats['ids_without_unique_exact_document'] == int(queried.sum())
    csv_cols = ['variant_id', 'hgvs_id', 'queried', 'skip_reason'] + book + pred_cols + cons_cols + af_cols + filt_cols
    out[csv_cols].to_csv(OUT_CSV, index=False)

    # ---- release information and access date -> annotation_versions.json
    src = snap['metadata_hg38'].get('src', {})
    dates = sorted({u['fetched_local_date'] for u in used if u.get('fetched_local_date')})
    servers = Counter()
    for u in used:
        servers[json.dumps({k: u['server'].get(k) for k in ('build_version', 'build_date', 'dbnsfp_version',
                                                            'gnomad_version_string')}, sort_keys=True)] += u['n_ids_used']
    gn_fields = {}
    for o in gobj:
        rel = GNOMAD_RELEASE.get(o, {'release': 'UNKNOWN - object not known to this script', 'max_an': None})
        max_an = stats.get(f'max_an::{o}', 0)
        gn_fields[o] = {
            'columns': [f'af_{o}', f'af_{o}_popmax_derived', 'af_any', 'af_popmax_any_derived', 'af_present'],
            'source_path_global_af': f'{o}.af.af', 'gnomad_release': rel['release'],
            'evidence_in_retrieved_records': {
                'n_records': stats[f'{o}_records'], 'max_AN_observed': max_an,
                'max_AN_possible_for_release': rel['max_an'],
                'max_AN_consistent_with_release': (max_an <= rel['max_an']) if rel['max_an'] else None,
                'af_keys': sorted(ann.af_keys[o])},
            'documents_in_index': {k: n for k, n in src.get('gnomad', {}).get('stats', {}).items()
                                   if o.split('_')[-1] in k}}
    versions = {
        'written_by': SCRIPT,
        'access_date': dates[0] if len(dates) == 1 else dates,
        'access_utc': {'first_query': min((u['fetched_utc'] for u in used), default=None),
                       'last_query': max((u['fetched_utc'] for u in used), default=None)},
        'service': {'name': 'MyVariant.info', 'endpoint': f'POST {API}/variant', 'assembly': ASSEMBLY,
                    'fields_requested': ann.fields, 'build_version': snap['metadata_hg38'].get('build_version'),
                    'build_date': snap['metadata_hg38'].get('build_date'), 'metadata_urls': snap['urls'],
                    'metadata_snapshot': f"{_rel(CACHE)}/{snap['_file']}",
                    'metadata_fetched_utc': snap['fetched_utc'],
                    'server_state_per_cached_answer': [dict(json.loads(k), n_ids=n) for k, n in servers.items()]},
        'dbnsfp': {
            'version': src.get('dbnsfp', {}).get('version'),
            'metadata_src_dbnsfp_hg38_index': src.get('dbnsfp'),
            'metadata_src_dbnsfp_default_endpoint': snap['metadata_default'].get('src', {}).get('dbnsfp'),
            'coordinates': 'GRCh38; every dbnsfp block is checked (dbnsfp.chrom / hg38.start / ref / alt = query)',
            'parser_notes': DBNSFP_PARSER_BASIS,
            'gnomad_fields_inside_dbnsfp': sorted(
                {k for k in listed | {p for p, _, _ in structure(snap['probe'].get('dbnsfp', {}), 'dbnsfp')}
                 if k.startswith('dbnsfp.') and 'gnomad' in k.lower()}),
            'other_population_blocks_inside_dbnsfp_not_used': sorted(
                {k.split('.')[1] for k in listed if k.startswith('dbnsfp.') and k.endswith('.af')})},
        'gnomad': {
            'metadata_src_gnomad_hg38_index': src.get('gnomad'),
            'note': 'The API metadata gives ONE version string for the whole gnomAD source; the release behind '
                    'each object is given in af_fields.',
            'release_basis': GNOMAD_RELEASE_BASIS,
            'top_level_gnomad_objects_in_hg38_index': gobj,
            'af_fields': gn_fields,
            'not_served': {
                'gnomAD v4.x': 'not in MyVariant.info (exomes are v2.1.1, genomes v3.1.1)',
                'joint (exome + genome) AF': 'no such field',
                'gnomAD fields inside dbnsfp': 'none served',
                'official popmax / grpmax / faf95': {
                    'status': 'not parsed by the hub; af_popmax is in the index field list but carries no value',
                    **{f'{o}.af.af_popmax': {
                        'in_index_field_list': f'{o}.af.af_popmax' in listed,
                        'n_values_in_retrieved_records': int(recs[f'af_{o}_popmax_official'].notna().sum())
                        if f'af_{o}_popmax_official' in recs else 0} for o in gobj}}}},
        'probe': {'id': snap['probe_id'], 'top_level_keys': sorted(snap['probe'])},
    }
    existing = {}
    if OUT_VERSIONS.exists():       # shared file: only this script's key is replaced
        try:
            existing = read_json(OUT_VERSIONS)
        except ValueError:
            existing = {}
        if not isinstance(existing, dict):
            existing = {'_previous_content': existing}
    existing[VERSIONS_KEY] = clean(versions)
    write_json(OUT_VERSIONS, existing)

    # ---- rules -> dbnsfp_rules.json
    def source(c):
        p = path_of[c]
        return {'source_path': p, 'available_in_served_version': p is not None,
                'n_records_with_value': stats[f'n::{c}@{p}'] if p else 0,
                'in_index_field_list': (p in listed) if p else None}

    rules = {
        'written_by': SCRIPT,
        'lookup': {'key': 'GRCh38 genomic variant (chrom, pos, ref_nt, alt_nt) of data/variants_v8.csv',
                   'id_format': 'chr<chrom>:g.<pos><ref_nt>><alt_nt> with assembly=hg38',
                   'endpoint': f'POST {API}/variant', 'fields': ann.fields, 'max_ids_per_request': BATCH_SIZE},
        'not_queried': 'rows whose alleles are not a single A/C/G/T substitution (multi-nucleotide substitution, '
                       'IUPAC ambiguity code, indel) are left unannotated: queried = 0, skip_reason gives the cause',
        'missing_values': 'NaN (empty cell) wherever a value is not served; nothing is imputed; no row is dropped',
        'R1_multi_transcript_rule': 'A score served as a list has one value per transcript (transcripts without '
                                    'value dropped, so not aligned with transcript ids); scalars and lists are '
                                    'treated uniformly and the MOST DAMAGING value across transcripts is used: max '
                                    'where higher_is_pathogenic is true, min where it is false. Raw scores only, no '
                                    'rankscores.',
        'R2_several_amino_acid_changes': 'A genomic variant with several amino-acid changes (overlapping reading '
                                         'frames / genes, or a splice-site row of another transcript) has several '
                                         'dbNSFP rows; MyVariant.info serves the scores of the FIRST row only '
                                         '(dbnsfp.aa[0]). If that row is not the ref1>alt1 '
                                         'change of the variant (dbnsfp_score_aa_match = 0), predictors with '
                                         'masked_if_other_aa_change = true are set to NaN and dbnsfp_scores_masked '
                                         '= 1.',
        'mask_scores_of_other_aa_change': MASK_SCORES_OF_OTHER_AA_CHANGE,
        'availability_rule': 'a column exists if at least one retrieved record carries a value at its source path '
                             '(the field list of /metadata/fields is incomplete and is reported for information)',
        'predictors': {p[0]: {'dbnsfp_column': p[1], **source(p[0]), 'higher_is_pathogenic': p[3],
                              'aggregation_across_transcripts': 'max' if p[3] else 'min',
                              'masked_if_other_aa_change': p[4], 'note': p[5]} for p in PREDICTORS},
        'predictors_not_available': pred_absent,
        'sift_rule': 'SIFT and SIFT4G are both served and are given as separate columns (sc_sift, sc_sift4g); '
                     'SIFT4G is never substituted for a missing SIFT value.',
        'conservation': {c[0]: {'description': c[2], **source(c[0]), 'higher_is_more_conserved': True,
                                'aggregation': 'one value per genomic position; max if several distinct values'}
                         for c in CONSERVATION},
        'conservation_not_available': cons_absent,
        'allele_frequency': {
            **{f'af_{o}': {'source_path': f'{o}.af.af', 'meaning': 'global allele frequency',
                           'gnomad_release': GNOMAD_RELEASE.get(o, {}).get('release', 'UNKNOWN')} for o in gobj},
            **{f'af_{o}_popmax_derived': {
                'source_paths': [f'{o}.af.af_{g}' for g in POPMAX_GROUPS],
                'meaning': 'DERIVED max allele frequency over the continental groups ' + ', '.join(POPMAX_GROUPS)
                           + ' (not the official gnomAD popmax / faf95 field, which is not served)',
                'gnomad_release': GNOMAD_RELEASE.get(o, {}).get('release', 'UNKNOWN')} for o in gobj},
            'af_any': 'max over the available global AFs (' + ', '.join(f'af_{o}' for o in gobj)
                      + '); NaN if the variant is in none of these gnomAD sources',
            'af_popmax_any_derived': 'max over the available af_*_popmax_derived',
            'af_present': '1 if a global AF exists in at least one gnomAD source, 0 if not, NaN if not queried',
            'qc_filter': 'AFs are given whatever the gnomAD QC filter; <object>_filter holds the filter flags '
                         '("PASS" = record without filter flag)',
            'not_available': ['joint exome+genome AF', 'official popmax / grpmax / faf95', 'gnomAD v4.x',
                              'gnomAD fields inside dbnsfp']},
        'bookkeeping': {
            'hgvs_id': 'id that was queried (empty if not queried)', 'queried': '1 if the row was queried',
            'skip_reason': 'why a row was not queried',
            'found': '1 if exactly one document with the queried _id was returned',
            'has_dbnsfp': '1 if that document has a dbnsfp block whose coordinates agree with the query',
            'dbnsfp_aa_match': '1 if ANY amino-acid annotation of the dbNSFP record (any dbnsfp.aa row, any '
                               'dbnsfp.hgvsp string) has ref1 and alt1; 0 if a dbnsfp block exists but none '
                               'matches; NaN without dbnsfp block',
            'dbnsfp_score_aa_match': '1 if the first dbnsfp.aa row - the row the served scores belong to - has '
                                     'ref1 and alt1; 0 otherwise; NaN without dbnsfp block',
            'dbnsfp_aapos_match': '1 if a matching annotation also has amino-acid position ref_pos',
            'dbnsfp_n_aa_rows': 'number of dbNSFP rows (distinct amino-acid changes) of the genomic variant',
            'dbnsfp_aa_annot': 'amino-acid annotations found: dbNSFP rows separated by "|" (the first one is the row '
                               'the served scores belong to), transcript positions by "/", "noAA" = row '
                               'without amino-acid change (splice-site row of another transcript)',
            'dbnsfp_scores_masked': '1 if rule R2 set the substitution-specific predictors to NaN',
            '<gnomad object>_filter': 'gnomAD QC filter of the record, NaN if the variant is not in that source'},
        'documents': 'notfound answers, several documents for one id, a returned _id different from the query and '
                     'blocks whose own coordinates contradict the query are counted in the summary and never used',
    }
    write_json(OUT_RULES, clean(rules))

    # ---- validation summary -> dbnsfp_v8_summary.json
    part, y = v.partition, pd.to_numeric(v.y, errors='coerce')
    has_db = out.has_dbnsfp == 1
    # plain float copies of the nullable 0/1 flags: comparisons give ordinary booleans (NaN compares False)
    aa_m, score_m, pos_m, n_rows, present = (out[c].astype(float) for c in (
        'dbnsfp_aa_match', 'dbnsfp_score_aa_match', 'dbnsfp_aapos_match', 'dbnsfp_n_aa_rows', 'af_present'))
    value_cols = pred_cols + cons_cols + [c for c in af_cols if c != 'af_present']
    groups = [('overall', pd.Series(True, index=v.index))] + [(p, part == p) for p in PARTITIONS]
    groups += [(p, part == p) for p in sorted(set(part) - set(PARTITIONS))]
    coverage = {g: {'n_variants': int(m.sum()),
                    **{c: {'n': int(out.loc[m, c].notna().sum()),
                           'frac': round(float(out.loc[m, c].notna().mean()), 4) if m.any() else None}
                       for c in value_cols}} for g, m in groups}
    # genes in which a column has no value at all (missingness is largely gene-wise, not random)
    has_value, n_gene = out[pred_cols + cons_cols].notna().groupby(v.gene).any(), v.groupby('gene').size()
    no_value = {c: {g: int(n_gene[g]) for g in has_value.index[~has_value[c]]} for c in pred_cols + cons_cols}
    flags = {g: {'queried': rate(queried, m), 'found': rate(out.found == 1, m), 'has_dbnsfp': rate(has_db, m),
                 'dbnsfp_aa_match_among_dbnsfp': rate(aa_m == 1, m & has_db),
                 'dbnsfp_score_aa_match_among_dbnsfp': rate(score_m == 1, m & has_db),
                 'dbnsfp_aapos_match_among_dbnsfp': rate(pos_m == 1, m & has_db),
                 'af_present_among_queried': rate(present == 1, m & queried)} for g, m in groups}

    def examples(mask, n=30):
        return [{'variant_id': r.variant_id, 'hgvs_id': r.hgvs_id, 'dbnsfp_aa_annot': r.dbnsfp_aa_annot,
                 'dbnsfp_hgvsp': r.hgvsp_strings} for r in out[mask].head(n).itertuples()]

    def by_gene(mask):
        return {g: int(n) for g, n in v.gene[mask].value_counts().items()}

    no_match = has_db & (aa_m == 0)
    other_row = has_db & (aa_m == 1) & (score_m == 0)
    pos_diff = has_db & (aa_m == 1) & (pos_m == 0)
    skipped = v[~queried]
    notfound = queried & (out.found == 0)
    aa = {
        'definition': rules['bookkeeping']['dbnsfp_aa_match'],
        'n_with_dbnsfp': int(has_db.sum()),
        'dbnsfp_aa_match': {'n_1': int((aa_m == 1).sum()), 'n_0': int(no_match.sum()),
                            'rate': round(float((aa_m == 1).sum() / max(int(has_db.sum()), 1)), 5)},
        'mismatch_by_gene': by_gene(no_match), 'mismatch_examples': examples(no_match),
        'match_only_in_a_non_first_dbnsfp_row': {
            'meaning': 'the genomic variant has the ref1>alt1 change, but the scores served belong to another '
                       'amino-acid change of the same genomic variant (rule R2)',
            'n': int(other_row.sum()), 'by_gene': by_gene(other_row), 'examples': examples(other_row)},
        'records_with_several_aa_rows': int((n_rows > 1).sum()),
        'records_with_several_aa_rows_by_gene': by_gene(n_rows > 1),
        'hgvsp_vs_first_aa_row': {'n_checked': stats['hgvsp_checked_against_first_aa_row'],
                                  'n_not_consistent': stats['hgvsp_NOT_consistent_with_first_aa_row']},
        'position_check': {
            'definition': rules['bookkeeping']['dbnsfp_aapos_match'],
            'n_substitution_matches_but_position_differs': int(pos_diff.sum()),
            'by_gene': by_gene(pos_diff), 'examples': examples(pos_diff)},
        'scores_masked': {'n_variants': int(out.dbnsfp_scores_masked.sum()),
                          'masking_enabled': MASK_SCORES_OF_OTHER_AA_CHANGE,
                          'n_values_per_column': {c: stats[f'masked::{c}@{path_of[c]}'] for c in pred_cols
                                                  if stats[f'masked::{c}@{path_of[c]}']}},
    }
    shapes = {c: {'one_value': stats[f'shape::{c}@{path_of[c]}::one'],
                  'several_identical_values': stats[f'shape::{c}@{path_of[c]}::several_identical'],
                  'several_distinct_values': stats[f'shape::{c}@{path_of[c]}::several_distinct'],
                  'non_numeric_values_ignored': stats[f'nonnumeric::{c}@{path_of[c]}']}
              for c in pred_cols + cons_cols}
    gn = {}
    pass_any = pd.Series(False, index=out.index)
    for o in gobj:
        pass_any |= out[f'af_{o}'].notna() & (out[f'{o}_filter'] == 'PASS')
        gn[o] = {'release': GNOMAD_RELEASE.get(o, {}).get('release', 'UNKNOWN'),
                 'n_records': stats[f'{o}_records'], 'n_with_global_af': int(out[f'af_{o}'].notna().sum()),
                 'n_records_without_global_af': stats[f'{o}_records_without_global_af'],
                 'n_global_af_equal_0': int((out[f'af_{o}'] == 0).sum()),
                 'filter_counts': out[f'{o}_filter'].value_counts().to_dict(),
                 'coordinates_vs_query': {k: stats[f'{o}_coordinates_{k}'] for k in ('agree', 'not_given', 'CONTRADICT')},
                 'n_block_is_list': stats[f'{o}_block_is_list'],
                 'n_several_distinct_global_af': stats[f'{o}_several_distinct_global_af'],
                 'n_non_numeric_af_ignored': stats[f'nonnumeric::af_{o}'],
                 'max_AN_observed': stats.get(f'max_an::{o}', 0),
                 'popmax_groups_served': {g: f'af_{g}' in ann.af_keys[o] for g in POPMAX_GROUPS}}
    gn['any_source'] = {'af_present_1': int((present == 1).sum()), 'af_present_0': int((present == 0).sum()),
                        'af_present_unknown_not_queried': int(present.isna().sum()),
                        'present_only_in_records_failing_gnomad_qc': int(((present == 1) & ~pass_any).sum()),
                        'af_any_equal_0': int((out.af_any == 0).sum()),
                        'in_all_sources': int(out[[f'af_{o}' for o in gobj]].notna().all(axis=1).sum())}

    # sanity (a): AUROC of each predictor, oriented by its direction, on the variants it covers
    auc = {}
    for g, m in groups:
        auc[g] = {}
        for c in pred_cols:
            mm = m & out[c].notna() & y.isin([0, 1])
            auc[g][c] = {'n': int(mm.sum()), 'n_pathogenic': int((y[mm] == 1).sum()),
                         'n_benign': int((y[mm] == 0).sum()),
                         'auroc': round(auroc(y[mm], out.loc[mm, c] if USE_MAX[c] else -out.loc[mm, c]), 4)}
    rho, n_rho = spearman(out.sc_alphamissense, out.sc_revel) if {'sc_alphamissense', 'sc_revel'} <= set(
        pred_cols) else (float('nan'), 0)
    known = []
    for vid in KNOWN_VARIANTS:
        r = out[out.variant_id == vid]
        known.append({'variant_id': vid, 'in_table': False} if r.empty else
                     {'in_table': True, **{c: r.iloc[0][c] for c in csv_cols}})

    n_q = int(queried.sum())
    summary = {
        'written_by': SCRIPT, 'input': input_info, 'output': str(_rel(OUT_CSV)),
        'n_variants': int(len(v)), 'n_rows_written': int(len(out)),
        'partition_sizes': {g: int(m.sum()) for g, m in groups[1:]},
        'query': {
            'n_queried': n_q, 'n_unique_ids': len(ids), 'n_not_queried': int((~queried).sum()),
            'not_queried_by_reason': skipped.skip_reason.value_counts().to_dict(),
            'not_queried_variants': [{'variant_id': r.variant_id, 'chrom': r.chrom, 'pos': r.pos, 'ref_nt': r.ref_nt,
                                      'alt_nt': r.alt_nt, 'reason': r.skip_reason, 'partition': r.partition}
                                     for r in skipped.itertuples()],
            'not_queried_that_differ_at_one_base_only': int(sum(differs_at_one_base(a, b) for a, b in
                                                                zip(skipped.ref_nt, skipped.alt_nt))),
            'n_found': int(out.found.sum()), 'n_not_found': int(notfound.sum()),
            'frac_not_found': round(float(notfound.sum() / max(n_q, 1)), 5),
            'not_found_variants': [{'variant_id': r.variant_id, 'hgvs_id': r.hgvs_id,
                                    'doc_problem': None if pd.isna(r.doc_problem) else r.doc_problem}
                                   for r in out[notfound].head(200).itertuples()],
            'n_notfound_answers': stats['notfound'],
            'n_ids_with_several_documents': stats['ids_with_several_documents'],
            'n_ids_without_unique_exact_document': stats['ids_without_unique_exact_document']},
        'dbnsfp': {
            'version': src.get('dbnsfp', {}).get('version'),
            'n_with_dbnsfp_block': int(has_db.sum()),
            'n_found_without_dbnsfp_block': int(((out.found == 1) & ~has_db).sum()),
            'found_without_dbnsfp_block': [{'variant_id': r.variant_id, 'hgvs_id': r.hgvs_id}
                                           for r in out[(out.found == 1) & ~has_db].head(200).itertuples()],
            'coordinates_vs_query': {k: stats[f'dbnsfp_coordinates_{k}'] for k in ('agree', 'not_given', 'CONTRADICT')},
            'n_block_is_list': stats['dbnsfp_block_is_list'],
            'n_aa_rows_without_amino_acid_change': stats['aa_rows_without_amino_acid_change'],
            'n_aa_rows_with_non_scalar_ref_or_alt': stats['aa_rows_with_non_scalar_ref_or_alt']},
        'columns': {'predictors': pred_cols, 'predictors_not_available': pred_absent,
                    'conservation': cons_cols, 'conservation_not_available': cons_absent,
                    'allele_frequency': af_cols,
                    'allele_frequency_not_available': rules['allele_frequency']['not_available'],
                    'source_paths': {c: path_of[c] for c in pred_cols + cons_cols}},
        'flags_by_group': flags, 'coverage_by_group': coverage,
        'genes_without_any_value': {c: g for c, g in no_value.items() if g}, 'values_per_record': shapes,
        'amino_acid_match': aa, 'gnomad': gn,
        'sanity': {'auroc_note': 'plausibility check only: each predictor oriented by higher_is_pathogenic, label y, '
                                 'variants covered by that predictor',
                   'auroc': auc, 'spearman_alphamissense_revel': {'rho': round(rho, 4), 'n': n_rho},
                   'known_variants': known},
        'versions': versions,
    }
    write_json(OUT_SUMMARY, clean(summary))

    # ---- console report
    q = summary['query']
    print(f'\n== validation summary ({_rel(OUT_SUMMARY)})')
    print(f"MyVariant.info build {versions['service']['build_version']}; dbNSFP {versions['dbnsfp']['version']}; "
          f"gnomAD metadata string {src.get('gnomad', {}).get('version')!r}; access date {versions['access_date']}")
    for o in gobj:
        print(f"  {o}: {gn[o]['release']}; {gn[o]['n_with_global_af']} variants with AF; max AN observed "
              f"{gn[o]['max_AN_observed']}; filters {gn[o]['filter_counts']}")
    print(f"variants {len(v)} | queried {q['n_queried']} | not queried {q['n_not_queried']} {q['not_queried_by_reason']}"
          f" | found {q['n_found']} | not found {q['n_not_found']} ({100 * q['frac_not_found']:.2f}%) | several "
          f"documents {q['n_ids_with_several_documents']} | with dbnsfp block {int(has_db.sum())}")
    if q['frac_not_found'] > NOT_FOUND_ALERT:
        print(f'  WARNING: more than {100 * NOT_FOUND_ALERT:.0f}% of the queried variants were not found - investigate')
    print(f"dbnsfp coordinates vs query: {summary['dbnsfp']['coordinates_vs_query']}")
    print(f"dbnsfp_aa_match: {aa['dbnsfp_aa_match']} | mismatches by gene {aa['mismatch_by_gene']}")
    print(f"several amino-acid rows: {aa['records_with_several_aa_rows']} | scores of another amino-acid change "
          f"(masked={MASK_SCORES_OF_OTHER_AA_CHANGE}): {aa['scores_masked']['n_variants']} | same substitution, other "
          f"position: {aa['position_check']['n_substitution_matches_but_position_differs']} "
          f"{aa['position_check']['by_gene']}")
    for title, ex in [('no amino-acid annotation matches', aa['mismatch_examples']),
                      ('scores belong to another amino-acid change', aa['match_only_in_a_non_first_dbnsfp_row']['examples']),
                      ('same substitution, other position', aa['position_check']['examples'][:10])]:
        for e in ex:
            print(f"    [{title}] {e['variant_id']:<18} {e['hgvs_id']:<26} dbNSFP aa: {e['dbnsfp_aa_annot']}  "
                  f"hgvsp: {e['dbnsfp_hgvsp']}")
    print(f'not available in the served version: predictors {pred_absent}, conservation {cons_absent}, '
          f"AF {rules['allele_frequency']['not_available']}")
    names = [g for g, _ in groups]
    print('\ncoverage (fraction of all variants of the group with a value)')
    print(f"  {'column':<34}" + ''.join(f'{g:>10}' for g in names))
    print(f"  {'n variants':<34}" + ''.join(f"{coverage[g]['n_variants']:>10}" for g in names))
    for c in value_cols:
        print(f'  {c:<34}' + ''.join(f3(coverage[g][c]['frac']) for g in names))
    print('genes without any value (genes / variants): ' + '; '.join(
        f'{c} {len(g)}/{sum(g.values())}' for c, g in no_value.items() if g))
    print('\nsanity (a) AUROC, oriented, variants covered  [n, AUROC]')
    print(f"  {'predictor':<22}{'direction':<12}" + ''.join(f'{g:>20}' for g in SANITY_PARTITIONS))
    for c in pred_cols:
        cells = ''.join(f"{auc[g][c]['n']:>10}{f3(auc[g][c]['auroc'])}" for g in SANITY_PARTITIONS if g in auc)
        print(f"  {c:<22}{'higher=P' if USE_MAX[c] else 'lower=P':<12}{cells}")
        if any(g in auc and auc[g][c]['auroc'] < 0.6 for g in SANITY_PARTITIONS):
            print(f'    WARNING: AUROC below 0.6 for {c} - check orientation / join')
    print(f'sanity (b) Spearman sc_alphamissense vs sc_revel: rho = {rho:.3f} (n = {n_rho})')
    print('sanity (c) known variants')
    for k in known:
        if not k['in_table']:
            print(f"  {k['variant_id']}: not in the table")
            continue
        print(f"  {k['variant_id']}  {k['hgvs_id']}  found={k['found']}  dbNSFP aa={k['dbnsfp_aa_annot']}  "
              f"aa_match={k['dbnsfp_aa_match']}")
        for cols in (pred_cols, cons_cols, [c for c in af_cols if c != 'af_present']):
            print('     ' + '  '.join(f'{c}={k[c]:.4g}' if pd.notna(k[c]) else f'{c}=NaN' for c in cols))
        print('     ' + '  '.join(f'{c}={k[c]}' for c in ['af_present'] + filt_cols))
    print(f'\nwritten: {_rel(OUT_CSV)} ({len(out)} rows, {len(csv_cols)} columns), '
          f'{_rel(OUT_RULES)}, {_rel(OUT_SUMMARY)}, {_rel(OUT_VERSIONS)}')


if __name__ == '__main__':
    main()
