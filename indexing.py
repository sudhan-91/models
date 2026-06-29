# -*- coding: utf-8 -*-
"""
indexing.py  —  Step 3 of 3 (delta-aware)

Embeds and indexes chunks into Elasticsearch for documents that load_data.py
flagged as new/changed. The only functional additions vs. the original:

  - Credentials read from environment variables instead of hardcoded.
  - For any doc_id whose delta_status is 'updated', existing chunks for that
    doc_id are deleted from ES *before* the fresh chunks are bulk-indexed.
    This matters because a changed document can produce a different number
    of chunks than before — without this step, old chunks past the new
    chunk count would linger in ES forever as stale data.
  - Each indexed chunk stores 'version_hash', which is exactly what
    load_data.py reads back on the next run to decide what's already current.
"""
import dataiku
import numpy as np
from dataiku import pandasutils as pdu
import pandas as pd
import os
import json
import uuid
import logging
import time
import traceback
import base64
from datetime import datetime
from typing import Optional
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True, nb_workers=4)

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
# Configuration — all secrets come from environment variables / Dataiku
# project variables. Set these under Project > Variables (or your platform's
# secret store), never hardcode credentials in source.
# ---------------------------------------------------------------------------
ES_CONFIG = {
    "host": os.environ["ES_HOST"],
    "username": os.environ["ES_USERNAME"],
    "password": os.environ["ES_PASSWORD"],
    "index": os.environ.get("ES_INDEX", "agentic_ai-production-docs"),
}

EMBEDDING_URL = os.environ.get("EMBEDDING_URL", "https://gpt4ifx.icp.infineon.com/embeddings")

EMBEDDING_AUTH_USERNAME = os.environ["EMBEDDING_AUTH_USERNAME"]
EMBEDDING_AUTH_PASSWORD = os.environ["EMBEDDING_AUTH_PASSWORD"]

WAIT_TIME = 65  # seconds to wait on embedding API rate-limit before retry


# ===========================================================================
# Helpers
# ===========================================================================

def get_embedding_vector(text: str, authentication: dict) -> tuple:
    """
    Call the embedding API and return (error, embedding).
    error is None on success, a string on failure.
    """
    encoding_format = "float"
    error = None
    embedding = ""

    try:
        headers = {"Content-Type": "application/json", **authentication}
        payload = {
            "input": text,
            "model": "multilingual-e5-large-instruct",
            "encoding_format": encoding_format,
        }
        resp = requests.post(
            EMBEDDING_URL,
            headers=headers,
            data=json.dumps(payload),
            verify=False,
            timeout=300,
        )
        resp.raise_for_status()
        data = resp.json()
        emb = data.get("data", [])[0].get("embedding")
        if emb is not None:
            return None, emb
        log.warning("Embedding API returned no embedding for a chunk.")
    except Exception as exc:
        error = str(exc)
        log.error("Embedding error: %s", exc)

    return error, embedding


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that converts numpy/pandas scalar types to native Python types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def bulk_index_to_es(documents: list) -> bool:
    """Bulk-index a list of ES documents via the _bulk API."""
    if not documents:
        return True
    try:
        ndjson_lines = []
        for doc in documents:
            action = {"index": {"_index": doc["_index"], "_id": doc["_id"]}}
            ndjson_lines.append(json.dumps(action, cls=NumpyEncoder))
            ndjson_lines.append(json.dumps(doc["_source"], cls=NumpyEncoder))
        body = "\n".join(ndjson_lines) + "\n"

        resp = requests.post(
            f"{ES_CONFIG['host']}/_bulk",
            headers={"Content-Type": "application/x-ndjson", "Accept": "application/json"},
            auth=(ES_CONFIG["username"], ES_CONFIG["password"]),
            data=body,
            verify=False,
            timeout=120,
        )
        resp.raise_for_status()
        result = resp.json()
        if result.get("errors"):
            log.warning("ES bulk index reported errors: %s", result)
        return True
    except Exception as exc:
        log.error("ES bulk index failed: %s", exc)
        return False


def delete_existing_chunks_for_doc(doc_id: str) -> bool:
    """
    Delete all previously-indexed chunks for a doc_id via ES delete_by_query.

    Required before re-indexing an 'updated' document: a changed document
    can produce a different number of chunks than its previous version, so
    a plain overwrite-by-_id would leave stale extra chunks behind if the
    new version has fewer chunks than the old one.
    """
    try:
        query = {"query": {"term": {"id": doc_id}}}
        resp = requests.post(
            f"{ES_CONFIG['host']}/{ES_CONFIG['index']}/_delete_by_query",
            headers={"Content-Type": "application/json"},
            auth=(ES_CONFIG["username"], ES_CONFIG["password"]),
            data=json.dumps(query),
            verify=False,
            timeout=60,
        )
        if resp.status_code == 404:
            # Index doesn't exist yet — nothing to delete, not an error.
            return True
        resp.raise_for_status()
        result = resp.json()
        log.info("Deleted %s existing chunk(s) for doc_id=%s before re-indexing.",
                  result.get("deleted", 0), doc_id)
        return True
    except Exception as exc:
        log.error("Failed to delete existing chunks for doc_id=%s: %s", doc_id, exc)
        return False


def get_auth_headers() -> dict:
    """
    Retrieve the embedding API auth token from environment variables
    (set via Dataiku project variables / secret store, not in source).
    """
    credentials = f"{EMBEDDING_AUTH_USERNAME}:{EMBEDDING_AUTH_PASSWORD}"
    token = base64.b64encode(credentials.encode("ascii")).decode("ascii")
    authorization = f"Basic {token}"
    return {"Token": token, "Authorization": authorization}


# Read recipe inputs
production_split_chunk = dataiku.Dataset("production_split_chunk")
df = production_split_chunk.get_dataframe()

authentication = get_auth_headers()


def process_doc_group(group_df: pd.DataFrame, auth: dict = None) -> dict:
    if auth is None:
        auth = authentication
    """Process all chunks for a single doc_id and return one result dict."""
    doc_id = group_df["doc_id"].iloc[0]
    doc_number = group_df["doc_number"].iloc[0]
    filename = group_df["filename"].iloc[0]
    run_id = group_df["run_id"].iloc[0]
    document_type = group_df["document_type"].iloc[0]
    document_sub_type = group_df["document_sub_type"].iloc[0]
    location = group_df["location"].iloc[0]
    version_hash = group_df["version_hash"].iloc[0] if "version_hash" in group_df.columns else ""
    delta_status = group_df["delta_status"].iloc[0] if "delta_status" in group_df.columns else ""
    chunks_indexed = 0
    error_msg = ""
    status = "success"
    product_data_list = []

    try:
        # If this doc previously existed in ES with a different version,
        # clear its old chunks first so re-indexing can't leave orphans
        # behind when the chunk count changes between versions.
        if delta_status == "updated":
            delete_ok = delete_existing_chunks_for_doc(doc_id)
            if not delete_ok:
                log.warning(
                    "Proceeding with re-index for doc_id=%s despite failed delete "
                    "of old chunks — stale chunks may remain.", doc_id
                )

        for _, chunk_row in group_df.iterrows():
            chunk_seq = chunk_row["chunk_seq"]
            text = chunk_row["chunk_text"]
            err, embedding = get_embedding_vector(text, auth)
            product_data_list.append({
                "_index": ES_CONFIG["index"],
                "_id": f"{doc_id}_{chunk_seq}",
                "_source": {
                    "id": doc_id,
                    "filename": chunk_row["filename"],
                    "filepath": chunk_row["filepath"],
                    "filetype": chunk_row["filetype"],
                    "chunk_seq": chunk_seq,
                    "content": text,
                    "embedding": embedding,
                    "document_type": document_type,
                    "document_sub_type": document_sub_type,
                    "location": location,
                    "version_hash": version_hash,
                    "indexed_at": datetime.utcnow().isoformat(),
                },
            })

        if product_data_list:
            success = bulk_index_to_es(product_data_list)
            if success:
                chunks_indexed = len(product_data_list)
            else:
                status = "failed_indexing"
                error_msg = "ES bulk index returned failure."

    except Exception as exc:
        status = "error"
        error_msg = str(exc)
        log.error("Error processing doc_id=%s: %s\n%s", doc_id, exc, traceback.format_exc())

    return {
        "run_id": run_id,
        "doc_id": doc_id,
        "doc_number": doc_number,
        "filename": filename,
        "status": status,
        "delta_status": delta_status,
        "chunks_indexed": chunks_indexed,
        "images_uploaded": int(group_df["images_count"].iloc[0]),
        "error_msg": error_msg,
        "processed_at": datetime.utcnow().isoformat(),
    }


if df.empty:
    log.info("No chunks to index this run (upstream delta produced nothing new).")
    production_indexing_df = pd.DataFrame(columns=[
        "run_id", "doc_id", "doc_number", "filename", "status", "delta_status",
        "chunks_indexed", "images_uploaded", "error_msg", "processed_at",
    ])
else:
    grouped = df.groupby("doc_id")
    doc_groups_dict = {
        doc_id: group.reset_index(drop=True)
        for doc_id, group in grouped
    }

    # Build a Series of doc_ids to drive parallel_apply
    doc_ids_series = pd.Series(list(doc_groups_dict.keys()), name="doc_id")

    def process_doc_id(doc_id: str) -> dict:
        """Wrapper: looks up the pre-built group DataFrame and processes it."""
        group = doc_groups_dict[doc_id]
        return process_doc_group(group, auth=authentication)

    results_series = doc_ids_series.parallel_apply(process_doc_id)

    results = [r for r in results_series if r is not None]
    production_indexing_df = pd.DataFrame(results)

# Write recipe outputs
production_indexing = dataiku.Dataset("production_indexing")
production_indexing.write_with_schema(production_indexing_df)
