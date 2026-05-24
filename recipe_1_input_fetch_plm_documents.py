# ----------------------------------------------------------------------------
# Dataiku Python Recipe  –  Recipe 1: INPUT
# Name  : recipe_1_input_fetch_plm_documents
# Input : (none — reads from external PostgreSQL / DIVE)
# Output: plm_documents_raw   (Dataiku dataset)
# ----------------------------------------------------------------------------
# Purpose
# -------
# Connects to the DIVE PostgreSQL data-warehouse and fetches PLM document
# metadata rows that match the target contexts and document type.
# The resulting DataFrame is written to the Dataiku output dataset
# "plm_documents_raw" for downstream processing.
# ----------------------------------------------------------------------------

import dataiku
import pandas as pd
import logging
from datetime import datetime

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
# Dataiku output dataset
# ---------------------------------------------------------------------------
output_dataset = dataiku.Dataset("plm_documents_raw")

# ---------------------------------------------------------------------------
# Pipeline configuration  (mirrors PIPELINE_CONFIG in the original script)
# ---------------------------------------------------------------------------
TARGET_CONTEXTS = [
    "cDtC_BE",
    "eQM_BE_QM_QMS_MT",
    "cDtC_BE_FMEA_CP_ProcessFlow",
    "cDtC_Bonding_Plan_Melaka",
]
DOCUMENT_TYPE = "Production: Procedure"

# ---------------------------------------------------------------------------
# SQL query  (same query used in run_pipeline())
# ---------------------------------------------------------------------------
FETCH_QUERY = """
    SELECT t2.*
    FROM vdb_bl_rd_product_lifecycle_management.bl_rdplm_windchill_biz_dim_wt_context AS T1
    RIGHT JOIN vdb_bl_rd_product_lifecycle_management.bl_rdplm_windchill_biz_dim_wt_document AS T2
      ON T1.id = T2.context_id
    WHERE T1.context_name IN (
      'cDtC_BE',
      'eQM_BE_QM_QMS_MT',
      'cDtC_BE_FMEA_CP_ProcessFlow',
      'cDtC_Bonding_Plan_Melaka'
    )
    AND t2.type = 'Production: Procedure'
    AND NOT (
      T2.infineon_loc = 'Wuxi'
      AND T1.context_name IN (
        'cDtC_BE',
        'cDtC_BE_FMEA_CP_ProcessFlow'
      )
    )
"""

# ---------------------------------------------------------------------------
# Helper: execute query via Dataiku managed connection
# Dataiku exposes the DIVE connection by name; adjust "DIVE_PostgreSQL" to
# match the actual connection name configured in your Dataiku instance.
# ---------------------------------------------------------------------------
def fetch_plm_documents() -> pd.DataFrame:
    """
    Fetch PLM document metadata from DIVE (PostgreSQL) using the
    Dataiku-managed connection and return a DataFrame.
    """
    try:
        # Use Dataiku's built-in SQL executor on the named connection
        client = dataiku.api_client()
        project = client.get_default_project()

        # Option A – direct SQL via Dataiku SQLExecutor2 (recommended)
        from dataiku.core.sql import SQLExecutor2
        executor = SQLExecutor2(connection="DIVE_PostgreSQL")   # ← adjust name
        df = executor.query_to_df(FETCH_QUERY)
        log.info("Fetched %d rows from DIVE.", len(df))
        return df

    except Exception as exc:
        log.error("Failed to fetch PLM documents from DIVE: %s", exc)
        raise


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log.info("=== Recipe 1: Fetch PLM Documents from DIVE ===")

    df = fetch_plm_documents()

    if df.empty:
        log.warning("No documents returned from DIVE. Writing empty dataset.")

    # Add a pipeline run timestamp for traceability
    df["pipeline_fetched_at"] = datetime.utcnow().isoformat()

    # Write to Dataiku output dataset
    output_dataset.write_with_schema(df)
    log.info("Written %d rows to dataset 'plm_documents_raw'.", len(df))
    log.info("=== Recipe 1 complete ===")


main()
