# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
"""
load_data.py  —  Step 1 of 3 (delta-aware)

Reads Windchill document/context tables, applies the existing business
filters, then narrows the result down to only NEW or CHANGED documents by
comparing a per-document version fingerprint against what is already
recorded in Elasticsearch.

Delta strategy (as agreed):
  - Source of truth for "already indexed" state = Elasticsearch itself.
  - A document is a delta candidate if:
      * its doc_id has no chunks in ES yet ("new"), OR
      * its doc_id has chunks in ES, but the stored version fingerprint
        differs from the freshly computed one ("updated").
  - Documents that no longer match the filters are left as-is in ES
    (no deletion logic) — out of scope for this pipeline by design.
"""
import os
import json
import hashlib
import logging

import dataiku
import pandas as pd
import numpy as np
from dataiku import pandasutils as pdu

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — all secrets come from env vars / Dataiku project variables.
# Set these under Project > Variables (or your platform's secret store),
# never hardcode credentials in source.
# ---------------------------------------------------------------------------
ES_CONFIG = {
    "host": os.environ["ES_HOST"],                 # e.g. https://<host>:9200
    "username": os.environ["ES_USERNAME"],
    "password": os.environ["ES_PASSWORD"],
    "index": os.environ.get("ES_INDEX", "agentic_ai-production-docs"),
}

# Columns whose combination defines "the document changed". Adjust this list
# to match whichever columns in your Windchill export actually change when a
# document is revised (iteration, state, modify date, etc.). Using a hash of
# these — rather than trusting a single field — protects against silent
# upstream changes that don't bump an obvious "version" column.
VERSION_FINGERPRINT_COLUMNS = [
    "id",
    "doc_number",
    "filename",
    "state",
    "latest_iteration",
    "context_id",
]


# ===========================================================================
# Helpers
# ===========================================================================

def compute_version_hash(row: pd.Series, columns: list) -> str:
    """Deterministic fingerprint of the columns that define a document's version."""
    payload = "|".join(str(row.get(c, "")) for c in columns)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def fetch_indexed_versions(es_config: dict) -> dict:
    """
    Query Elasticsearch for the version_hash currently stored per doc_id.

    Uses a terms aggregation grouped by doc id, taking the most recent
    version_hash per doc (top_hits sorted by indexed_at desc). Returns
    {doc_id: version_hash}. Returns {} if the index doesn't exist yet
    (first-ever run) or on any query failure — callers should treat a
    missing index as "nothing indexed yet", not as an error.
    """
    url = f"{es_config['host']}/{es_config['index']}/_search"
    query = {
        "size": 0,
        "aggs": {
            "by_doc": {
                "terms": {"field": "id", "size": 100000},
                "aggs": {
                    "latest": {
                        "top_hits": {
                            "size": 1,
                            "sort": [{"indexed_at": {"order": "desc"}}],
                            "_source": ["version_hash"],
                        }
                    }
                },
            }
        },
    }

    try:
        resp = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            auth=(es_config["username"], es_config["password"]),
            data=json.dumps(query),
            verify=False,
            timeout=60,
        )
        if resp.status_code == 404:
            log.info("ES index '%s' not found — treating as empty (first run).", es_config["index"])
            return {}
        resp.raise_for_status()
        data = resp.json()

        buckets = data.get("aggregations", {}).get("by_doc", {}).get("buckets", [])
        versions = {}
        for bucket in buckets:
            doc_id = bucket["key"]
            hits = bucket.get("latest", {}).get("hits", {}).get("hits", [])
            if hits:
                versions[str(doc_id)] = hits[0]["_source"].get("version_hash")
        log.info("Fetched %d existing doc versions from ES.", len(versions))
        return versions

    except Exception as exc:
        log.error("Failed to fetch existing versions from ES: %s", exc)
        log.error("Proceeding as if no documents are indexed yet (full reprocess). "
                   "Verify ES connectivity if this is unexpected.")
        return {}


def compute_delta(df: pd.DataFrame, indexed_versions: dict) -> pd.DataFrame:
    """
    Add 'version_hash' and 'delta_status' columns, then return only the rows
    that are new or changed relative to what's already in ES.
    """
    df = df.copy()
    df["version_hash"] = df.apply(
        lambda row: compute_version_hash(row, VERSION_FINGERPRINT_COLUMNS), axis=1
    )

    def classify(row):
        doc_id = str(row["id"])
        existing_hash = indexed_versions.get(doc_id)
        if existing_hash is None:
            return "new"
        if existing_hash != row["version_hash"]:
            return "updated"
        return "unchanged"

    df["delta_status"] = df.apply(classify, axis=1)

    n_new = (df["delta_status"] == "new").sum()
    n_updated = (df["delta_status"] == "updated").sum()
    n_unchanged = (df["delta_status"] == "unchanged").sum()
    log.info(
        "Delta classification — new: %d, updated: %d, unchanged (skipped): %d",
        n_new, n_updated, n_unchanged,
    )

    delta_df = df[df["delta_status"].isin(["new", "updated"])].reset_index(drop=True)
    return delta_df


# ===========================================================================
# Main
# ===========================================================================

# Read recipe inputs
bl_rdplm_windchill_biz_dim_wt_document_1 = dataiku.Dataset("bl_rdplm_windchill_biz_dim_wt_document_1")
bl_rdplm_windchill_biz_dim_wt_document_1_df = bl_rdplm_windchill_biz_dim_wt_document_1.get_dataframe()
bl_rdplm_windchill_biz_dim_wt_context_1 = dataiku.Dataset("bl_rdplm_windchill_biz_dim_wt_context_1")
bl_rdplm_windchill_biz_dim_wt_context_1_df = bl_rdplm_windchill_biz_dim_wt_context_1.get_dataframe()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Compute recipe outputs
# ── Rename for brevity ───────────────────────────────────────────────────────
t1 = bl_rdplm_windchill_biz_dim_wt_context_1_df    # context  table
t2 = bl_rdplm_windchill_biz_dim_wt_document_1_df   # document table

# ── Step 1: Define allowed context names ────────────────────────────────────
allowed_context_names = [
    'cDtC_BE',
    'eQM_BE_QM_QMS_MT',
    'cDtC_BE_FMEA_CP_ProcessFlow',
    'cDtC_Bonding_Plan_Melaka'
]

# ── Step 2: RIGHT JOIN t1 → t2 on id = context_id ───────────────────────────
#    RIGHT JOIN = keep ALL rows from t2 (document)
#    match with t1 (context) where available
merged_df = t2.merge(
    t1[['id', 'context_name']],   # only need id + context_name from t1
    left_on='context_id',          # t2.context_id
    right_on='id',                 # t1.id
    how='left',                    # left = RIGHT JOIN (t2 is base)
    suffixes=('', '_context')      # avoid column name collision
)

# ── Step 3: WHERE t1.context_name IN (...) ───────────────────────────────────
condition_context_name = merged_df['context_name'].isin(allowed_context_names)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ── Step 5: AND NOT (t2.infineon_loc = 'Wuxi' ────────────────────────────────
#              AND t1.context_name IN ('cDtC_BE', 'cDtC_BE_FMEA_CP_ProcessFlow'))
exclude_context_names = ['cDtC_BE', 'cDtC_BE_FMEA_CP_ProcessFlow']

condition_exclude = (
    (merged_df['infineon_loc'] == 'Wuxi') &
    (merged_df['context_name'].isin(exclude_context_names))
)

# ── Step 6: Combine all conditions ───────────────────────────────────────────
final_df = merged_df[
    condition_context_name &
    ~condition_exclude          # NOT the exclusion condition
]
final_df = final_df[
    (final_df["state"].astype(str).str.strip() == "Released") &
    (final_df["latest_iteration"] == 1)
]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ── Step 7: Select only t2 columns (SELECT t2.*) ─────────────────────────────
t2_columns = t2.columns.tolist()
final_df   = final_df[t2_columns]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
log.info("Rows after business filters (before delta check): %s", final_df.shape)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ── Step 8: Reset index ───────────────────────────────────────────────────────
filtered_dataset_df = final_df.reset_index(drop=True)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ── Step 9: DELTA — keep only new/changed documents vs what's already in ES ──
indexed_versions = fetch_indexed_versions(ES_CONFIG)
preprocessed_dataset_df = compute_delta(filtered_dataset_df, indexed_versions)

log.info("Rows after delta filtering (to be processed downstream): %s", preprocessed_dataset_df.shape)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Write recipe outputs
preprocessed_dataset = dataiku.Dataset("preprocessed_dataset")
preprocessed_dataset.write_with_schema(preprocessed_dataset_df)
