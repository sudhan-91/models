# ----------------------------------------------------------------------------
# Dataiku Python Recipe  –  Recipe 3: EMBED & INDEX
# Name  : recipe_3_embed_and_index
# Input : plm_chunks_raw           (Dataiku dataset — one row per text chunk)
# Output: plm_processing_results   (Dataiku dataset — one row per document)
# ----------------------------------------------------------------------------
# Purpose
# -------
# For each chunk row produced by Recipe 2:
#   1. Calls the embedding API to get a 1024-dim vector for the chunk text
#   2. Bulk-indexes the enriched chunk document to Elasticsearch
#   3. Writes a per-document summary row to "plm_processing_results"
# ----------------------------------------------------------------------------

import dataiku
import pandas as pd
import os
import json
import uuid
import logging
import time
import traceback
from datetime import datetime
from typing import Optional

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
# Dataiku datasets
# ---------------------------------------------------------------------------
input_dataset  = dataiku.Dataset("plm_chunks_raw")
output_dataset = dataiku.Dataset("plm_processing_results")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ES_CONFIG = {
    "host":     "https://muclv0688.muc.infineon.com:9200",
    "username": "SINqmagent",
    "password": "aGentiC4I@25-12",
    "index":    "agentic_ai-product_text",
}

EMBEDDING_URL = "https://gpt4ifx.icp.infineon.com/embeddings"

WAIT_TIME = 65   # seconds to wait on embedding API rate-limit before retry


# ===========================================================================
# Helpers
# ===========================================================================

def get_embedding_vector(text: str, authentication: dict) -> tuple:
    """
    Call the embedding API and return (error, embedding).
    error is None on success, a string on failure.
    """
    encoding_format = "float"
    max_count       = 3
    error           = None
    embedding       = ""

    try:
        count = 0
        while count < max_count:
            headers = {"Content-Type": "application/json", **authentication}
            payload = {
                "input":           text,
                "model":           "multilingual-e5-large-instruct",
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
            emb  = data.get("data", [])[0].get("embedding")
            if emb is not None:
                return None, emb
            count += 1
            log.warning("Embedding retry %d.", count)
    except Exception as exc:
        error = str(exc)
        log.error("Embedding error: %s", exc)

    return error, embedding


def bulk_index_to_es(documents: list) -> bool:
    """Bulk-index a list of ES documents via the _bulk API."""
    try:
        ndjson_lines = []
        for doc in documents:
            action = {"index": {"_index": doc["_index"], "_id": doc["_id"]}}
            ndjson_lines.append(json.dumps(action))
            ndjson_lines.append(json.dumps(doc["_source"]))
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


def get_auth_headers() -> dict:
    """
    Retrieve the embedding API auth token from Dataiku project variables.
    Set EMBEDDING_AUTH_TOKEN under Project > Variables in Dataiku.
    """
    client  = dataiku.api_client()
    project = client.get_default_project()
    token   = project.get_variables()["standard"].get("EMBEDDING_AUTH_TOKEN", "")
    if not token:
        raise ValueError("EMBEDDING_AUTH_TOKEN not set in Dataiku project variables.")
    return {"Authorization": f"Bearer {token}"}


# ===========================================================================
# Main
# ===========================================================================

def main():
    log.info("=" * 60)
    log.info("Recipe 3: Embed & Index PLM Chunks")
    log.info("=" * 60)

    df = input_dataset.get_dataframe()
    log.info("Read %d chunk rows from 'plm_chunks_raw'.", len(df))

    if df.empty:
        log.warning("Input dataset is empty – nothing to embed or index.")
        output_dataset.write_with_schema(pd.DataFrame())
        return

    # Retrieve auth headers once
    authentication = get_auth_headers()

    # Group chunks by doc_id so we can emit one summary row per document
    results = []
    grouped = df.groupby("doc_id")
    total_docs = len(grouped)

    for doc_idx, (doc_id, doc_chunks) in enumerate(grouped):
        doc_number   = doc_chunks["doc_number"].iloc[0]
        filename     = doc_chunks["filename"].iloc[0]
        run_id       = doc_chunks["run_id"].iloc[0]
        chunks_indexed   = 0
        error_msg        = ""
        status           = "success"

        product_data_list = []

        for _, chunk_row in doc_chunks.iterrows():
            chunk_seq = chunk_row["chunk_seq"]
            text      = chunk_row["chunk_text"]

            # Embed with retry loop
            while True:
                err, embedding = get_embedding_vector(text, authentication)
                if err is not None:
                    log.error("Embedding error on doc_id=%s chunk=%d: %s. Waiting %ds …",
                              doc_id, chunk_seq, err, WAIT_TIME)
                    time.sleep(WAIT_TIME)
                else:
                    break

            product_data_list.append({
                "_index": ES_CONFIG["index"],
                "_id":    f"{doc_id}_{chunk_seq}",
                "_source": {
                    "id":         doc_id,
                    "filename":   chunk_row["filename"],
                    "filepath":   chunk_row["filepath"],
                    "filetype":   chunk_row["filetype"],
                    "chunk_seq":  chunk_seq,
                    "content":    text,
                    "embedding":  embedding,
                    "indexed_at": datetime.utcnow().isoformat(),
                },
            })

        # Bulk index all chunks for this document
        if product_data_list:
            success = bulk_index_to_es(product_data_list)
            if success:
                chunks_indexed = len(product_data_list)
            else:
                status    = "failed_indexing"
                error_msg = "ES bulk index returned failure."

        results.append({
            "run_id":          run_id,
            "doc_id":          doc_id,
            "doc_number":      doc_number,
            "filename":        filename,
            "status":          status,
            "chunks_indexed":  chunks_indexed,
            "images_uploaded": int(doc_chunks["images_count"].iloc[0]),
            "error_msg":       error_msg,
            "processed_at":    datetime.utcnow().isoformat(),
        })

        if (doc_idx + 1) % 50 == 0:
            log.info("Progress: %d / %d documents …", doc_idx + 1, total_docs)

    results_df = pd.DataFrame(results)
    output_dataset.write_with_schema(results_df)

    success_count = len(results_df[results_df["status"] == "success"])
    log.info("=" * 60)
    log.info("Recipe 3 complete.  Docs=%d  Success=%d  Other=%d",
             total_docs, success_count, total_docs - success_count)
    log.info("=" * 60)


main()
