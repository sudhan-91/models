import cx_Oracle
import pandas as pd
import re
import os

# ════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════

DB_USER     = "dwh_read_qmagent"
DB_PASSWORD = os.environ.get("DWH_PASSWORD", "YOUR_PASSWORD_HERE")  # Use env var, not hardcoded
DB_DSN      = "dwhrbg.rbg.infineon.com"
INSTANT_CLIENT_DIR = r"C:\sudhan\instantclient-basic-windows.x64-23.8.0.25.04\instantclient_23_8"

ALLOWED_EXTENSIONS = {'ppt', 'pptx', 'xls', 'xlsx', 'csv'}

# ── Columns to select from each table ────────────────────────
NERMAL_COLS = [
    'KEY', 'PROCNR', 'TITLE', 'MOTIVATION', 'CURRENT_PROCEDURE',
    'NEWPROCEDURE', 'PROJECT_TYPE', 'REMARK', 'PK_LOCATION',
    'PACKAGE_FAMILY', 'PACKAGE_NAME', 'PROJECT_PHASE', 'LIFECYCLESTATE',
    'CM04_DATE', 'CM08_DATE', 'CM10_DATE', 'LASTUPDATE', 'VERSION',
    'SEQUENCE', 'CREATIONDATE', 'CHANGE_CLASS', 'LOCATION',
    'FROZENINDICATOR', 'SUPERSEDED', 'FACILITY', 'BASIC_TYPE', 'SALES_NAME',
]

TCBU_COLS = [
    'GRADENO',            # join key → dropped after merge
    'CHANGE_ID',
    'AFFECTED_PROCESSES',
    'PL',
    'TEMPCHANGE',
    'MILESTONE',
    'PROJECT_STATE',
    'REASONCANCEL',
    'CHECKERSREJECTED',
]

# Expected 38 final columns:
#   27 from NERMAL (incl. NERMAL_CREATION_DATE, excl. raw CREATIONDATE)
#   +8 from TCBU   (GRADENO dropped after merge, so net 8)
#   +2 from FILES  (ATTACHMENTS, FILE_COUNT)
#   +1 NER_ID
#   = 38

TCFILES_COLS = ['ORIG_OBID', 'SHARELINK', 'FILENAME']


# ════════════════════════════════════════════════════════════
# STEP 0: DB Connection
# ════════════════════════════════════════════════════════════

def create_pool():
    cx_Oracle.init_oracle_client(lib_dir=INSTANT_CLIENT_DIR)
    pool = cx_Oracle.SessionPool(
        DB_USER, DB_PASSWORD, DB_DSN,
        min=1, max=5, increment=1, encoding="UTF-8"
    )
    print("DB connection pool created")
    return pool


def query_table(pool, query, label=""):
    conn = pool.acquire()
    try:
        df = pd.read_sql(query, conn)
        if label:
            print(f"  Loaded {label}: {df.shape[0]:,} rows x {df.shape[1]} cols")
        return df
    finally:
        pool.release(conn)


# ════════════════════════════════════════════════════════════
# STEP 1: Load & Validate Tables
# ════════════════════════════════════════════════════════════

def load_tables(pool):
    print("\n" + "=" * 55)
    print("  STEP 1: Loading Tables")
    print("=" * 55)

    df_nermal  = query_table(pool, "SELECT * FROM LOC_RBL.DWH_NERMAL",           "DWH_NERMAL")
    df_tcbu    = query_table(pool, "SELECT * FROM LOC_RBL.DWH_TEAMCENTER_BU",    "DWH_TEAMCENTER_BU")
    df_tcfiles = query_table(pool, "SELECT * FROM LOC_RBL.DWH_TEAMCENTER_FILES", "DWH_TEAMCENTER_FILES")

    def select_cols(df, cols, name):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            print(f"  WARNING: {name} missing columns: {missing}")
        available = [c for c in cols if c in df.columns]
        return df[available].copy()

    df_nermal  = select_cols(df_nermal,  NERMAL_COLS,  "DWH_NERMAL")
    df_tcbu    = select_cols(df_tcbu,    TCBU_COLS,    "DWH_TEAMCENTER_BU")
    df_tcfiles = select_cols(df_tcfiles, TCFILES_COLS, "DWH_TEAMCENTER_FILES")

    # Rename CREATIONDATE to NERMAL_CREATION_DATE to avoid confusion with other dates
    if 'CREATIONDATE' in df_nermal.columns:
        df_nermal.rename(columns={'CREATIONDATE': 'NERMAL_CREATION_DATE'}, inplace=True)

    return df_nermal, df_tcbu, df_tcfiles


# ════════════════════════════════════════════════════════════
# STEP 2: Filter Files by Extension
# ════════════════════════════════════════════════════════════

def filter_files(df_tcfiles):
    print("\n" + "=" * 55)
    print("  STEP 2: Filter Files by Extension")
    print("=" * 55)

    df_tcfiles = df_tcfiles.copy()
    df_tcfiles['EXT'] = (
        df_tcfiles['FILENAME']
        .str.strip().str.lower()
        .str.split('.').str[-1].str.strip()
    )

    before = len(df_tcfiles)
    df_filtered = df_tcfiles[df_tcfiles['EXT'].isin(ALLOWED_EXTENSIONS)].reset_index(drop=True)
    after = len(df_filtered)

    print(f"  Before : {before:,} files")
    print(f"  After  : {after:,} files  (kept {after/before*100:.1f}%)")
    print(f"  Removed: {before - after:,} non-PPT/Excel files")

    return df_filtered


# ════════════════════════════════════════════════════════════
# STEP 3: Filter NERMAL
#
#   Rule 1 — per KEY, keep only the latest VERSION
#             (VERSION = milestone order; we want the furthest milestone)
#
#   Rule 2 — within that latest VERSION, keep only the highest SEQUENCE
#             (SEQUENCE = revision within a milestone)
#
#   Rule 3 — if two rows share the same KEY + VERSION + SEQUENCE,
#             keep BOTH — they are two different variants of the same record
# ════════════════════════════════════════════════════════════

def filter_nermal_by_sequence(df_nermal):
    print("\n" + "=" * 55)
    print("  STEP 3: Filter NERMAL")
    print("  Rule 1: latest VERSION per KEY (milestone order)")
    print("  Rule 2: highest SEQUENCE within that version")
    print("  Rule 3: ties on SEQUENCE = variants, keep both")
    print("=" * 55)

    before = len(df_nermal)

    # ── Rule 1: latest VERSION per KEY ───────────────────────────
    max_version = (
        df_nermal
        .groupby('KEY')['VERSION']
        .max()
        .reset_index()
        .rename(columns={'VERSION': 'MAX_VERSION'})
    )
    df_step1 = pd.merge(df_nermal, max_version, on='KEY', how='inner')
    df_step1 = df_step1[
        df_step1['VERSION'] == df_step1['MAX_VERSION']
    ].drop(columns=['MAX_VERSION']).reset_index(drop=True)

    after_ver = len(df_step1)
    print(f"  Before                    : {before:,} rows")
    print(f"  After Rule 1 (latest ver) : {after_ver:,} rows  ({before - after_ver:,} old-milestone rows dropped)")

    # ── Rule 2 + 3: highest SEQUENCE per KEY (ties = variants, both kept) ──
    # Group only by KEY — version is already fixed to latest after Rule 1.
    # Rows where SEQUENCE == max(SEQUENCE) for that KEY are kept.
    # If two rows have the same max SEQUENCE, both survive (Rule 3 = variants).
    max_seq = (
        df_step1
        .groupby('KEY')['SEQUENCE']
        .max()
        .reset_index()
        .rename(columns={'SEQUENCE': 'MAX_SEQUENCE'})
    )
    df_step2 = pd.merge(df_step1, max_seq, on='KEY', how='inner')
    df_step2 = df_step2[
        df_step2['SEQUENCE'] == df_step2['MAX_SEQUENCE']
    ].drop(columns=['MAX_SEQUENCE']).reset_index(drop=True)

    after_seq = len(df_step2)
    variants  = after_seq - df_step2['KEY'].nunique()

    print(f"  After Rule 2 (highest seq): {after_seq:,} rows  ({after_ver - after_seq:,} lower-seq rows dropped)")
    print(f"  Rule 3 variants kept      : {variants:,} extra rows (same KEY+VERSION+SEQUENCE, different data)")
    print(f"  Total rows removed        : {before - after_seq:,}")

    return df_step2


# ════════════════════════════════════════════════════════════
# STEP 4: Group Files per KEY
#
#   Files are joined on KEY (= ORIG_OBID in the files table).
#   KEY already points to the latest-version+highest-sequence
#   record after Step 3, so files naturally map to that record.
#
#   Multiple files with the same category for one KEY are rare
#   but valid — ALL are stored in the ATTACHMENTS list (no dedup).
# ════════════════════════════════════════════════════════════

def group_files(df_tcfiles_filtered):
    print("\n" + "=" * 55)
    print("  STEP 4: Group Attachments per KEY")
    print("  All files per KEY collected into a list (including same-category duplicates)")
    print("=" * 55)

    df_grouped = (
        df_tcfiles_filtered
        .groupby('ORIG_OBID')
        .apply(lambda x: [
            {
                'filename' : row['FILENAME'],
                'filepath' : row['SHARELINK'],
                'filetype' : str(row['FILENAME']).split('.')[-1].lower()
                             if pd.notna(row['FILENAME']) else 'unknown',
            }
            for _, row in x.iterrows()
        ])
        .reset_index()
        .rename(columns={'ORIG_OBID': 'KEY', 0: 'ATTACHMENTS'})
    )

    df_grouped['FILE_COUNT'] = df_grouped['ATTACHMENTS'].apply(len)

    print(f"  KEYs with files          : {len(df_grouped):,}")
    print(f"  Max files per KEY        : {df_grouped['FILE_COUNT'].max()}")
    print(f"  KEYs with multiple files : {(df_grouped['FILE_COUNT'] > 1).sum():,}")

    return df_grouped


# ════════════════════════════════════════════════════════════
# STEP 5: Merge All Tables
# ════════════════════════════════════════════════════════════

def merge_tables(df_nermal_filtered, df_tcbu, df_files_grouped):
    print("\n" + "=" * 55)
    print("  STEP 5: Merge All Tables")
    print("=" * 55)

    # Merge 1: NERMAL + TCBU (left join on KEY = GRADENO)
    df_merge1 = pd.merge(
        df_nermal_filtered,
        df_tcbu,
        left_on='KEY', right_on='GRADENO',
        how='left', suffixes=('_NERMAL', '_TCBU')
    )
    df_merge1.drop(columns=['GRADENO'], inplace=True, errors='ignore')
    print(f"  Merge 1 (NERMAL + TCBU) : {df_merge1.shape}")

    # Merge 2: + grouped files (left join on KEY)
    # Files mapped to the KEY that survived Step 3 (latest version + highest sequence)
    df_merge2 = pd.merge(
        df_merge1,
        df_files_grouped[['KEY', 'ATTACHMENTS', 'FILE_COUNT']],
        on='KEY', how='left'
    )
    df_merge2['ATTACHMENTS'] = df_merge2['ATTACHMENTS'].apply(
        lambda x: x if isinstance(x, list) else []
    )
    df_merge2['FILE_COUNT'] = df_merge2['FILE_COUNT'].fillna(0).astype(int)
    print(f"  Merge 2 (+ Files)       : {df_merge2.shape}")

    return df_merge2.reset_index(drop=True)


# ════════════════════════════════════════════════════════════
# STEP 6: Extract NER_ID
#   Only extracts the NER_ID from PROCNR.
#   No re-deduplication here — Step 3 already applied all rules.
# ════════════════════════════════════════════════════════════

def extract_ner_id(df):
    print("\n" + "=" * 55)
    print("  STEP 6: Extract NER_ID from PROCNR")
    print("=" * 55)

    df = df.copy()
    # Extract pattern like NER_1234/2023_01_ABC from PROCNR string
    df['NER_ID'] = df['PROCNR'].str.extract(r'(NER_\d+/\d+_\d+_\w+)')

    print(f"  Total rows      : {len(df):,}")
    print(f"  Unique NER_IDs  : {df['NER_ID'].nunique():,}")
    print(f"  Variant rows    : {len(df) - df['NER_ID'].nunique():,}  (same NER_ID, different variant data)")

    return df


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════

def run_pipeline(export_csv=True, csv_path="NER_ID_GROUPED.csv"):
    print("\n" + "=" * 55)
    print("  NER PIPELINE - Starting")
    print("=" * 55)

    pool = create_pool()

    try:
        # Load
        df_nermal, df_tcbu, df_tcfiles = load_tables(pool)

        # Filter files to PPT/Excel only
        df_tcfiles_filtered = filter_files(df_tcfiles)

        # Filter NERMAL: latest version -> highest sequence -> keep variants
        df_nermal_filtered = filter_nermal_by_sequence(df_nermal)

        # Group files per KEY (all files including same-category duplicates)
        df_files_grouped = group_files(df_tcfiles_filtered)

        # Merge all three sources
        df_merged = merge_tables(df_nermal_filtered, df_tcbu, df_files_grouped)

        # Extract NER_ID (no re-dedup, Step 3 already handled all rules)
        df_final = extract_ner_id(df_merged)

        # ── Enforce final 38-column order ─────────────────────
        FINAL_COLS = [
            # NERMAL (27)
            'KEY', 'PROCNR', 'NER_ID', 'TITLE', 'MOTIVATION',
            'CURRENT_PROCEDURE', 'NEWPROCEDURE', 'PROJECT_TYPE', 'REMARK',
            'PK_LOCATION', 'PACKAGE_FAMILY', 'PACKAGE_NAME', 'PROJECT_PHASE',
            'LIFECYCLESTATE', 'CM04_DATE', 'CM08_DATE', 'CM10_DATE',
            'LASTUPDATE', 'VERSION', 'SEQUENCE', 'NERMAL_CREATION_DATE',
            'CHANGE_CLASS', 'LOCATION', 'FROZENINDICATOR', 'SUPERSEDED',
            'FACILITY', 'BASIC_TYPE', 'SALES_NAME',
            # TCBU (8)
            'CHANGE_ID', 'AFFECTED_PROCESSES', 'PL', 'TEMPCHANGE',
            'MILESTONE', 'PROJECT_STATE', 'REASONCANCEL', 'CHECKERSREJECTED',
            # FILES (2)
            'ATTACHMENTS', 'FILE_COUNT',
        ]
        available = [c for c in FINAL_COLS if c in df_final.columns]
        df_final  = df_final[available].reset_index(drop=True)

        assert df_final.shape[1] == 38, (
            f"Expected 38 columns, got {df_final.shape[1]}. "
            f"Missing: {[c for c in FINAL_COLS if c not in df_final.columns]}"
        )
        print(f"\n  Column count verified: {df_final.shape[1]} columns")

        if export_csv:
            df_final.to_csv(csv_path, index=False)
            print(f"  Exported to: {csv_path}")

        print("\n" + "=" * 55)
        print(f"  PIPELINE COMPLETE")
        print(f"  Final shape    : {df_final.shape[0]:,} rows x {df_final.shape[1]} cols")
        print(f"  Unique NER_IDs : {df_final['NER_ID'].nunique():,}")
        print("=" * 55)

        return df_final

    finally:
        pool.close()


if __name__ == "__main__":
    df_result = run_pipeline(export_csv=True)
    print("\nSample Output:")
    print(df_result[['NER_ID', 'VERSION', 'SEQUENCE', 'LIFECYCLESTATE', 'FILE_COUNT']].head(10))
