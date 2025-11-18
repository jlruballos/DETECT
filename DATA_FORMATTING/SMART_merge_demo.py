#!/usr/bin/env python3
"""
SMART + Yearly Demographics Merge Pipeline (Year-Aligned with Fallback)

This script takes:
- Processed SMART data (per-subject repeated cognitive test data)
- Yearly clinical/demographic data (yearly_recoded.csv)

and merges YEAR-SPECIFIC demographic features onto every SMART test row
for each participant (subid). Primary merge is done by (subid, smart_year)
<-> (subid, visityr_cr). If no demographic row exists for that year, we
fall back to subid-only: we use the latest available yearly record for
that subject to fill in demographics.

Main steps:
- Set up paths and logging
- Load processed SMART data (smart_cleaned.csv)
- Derive SMART year from the SMART test date
- Load recoded yearly demographic data (yearly_recoded.csv)
- Select desired demographic columns
- Merge on (subid, smart_year) <-> (subid, visityr_cr)
- Fallback: for rows with no year-aligned demographics, merge by subid only
- Save merged dataset and summary diagnostics

Outputs:
- Merged dataset (`smart_with_yearly_demographics.csv`)
- Row count summaries before/after merge
- Missingness summaries before/after merge
- Logged pipeline steps with timestamps
"""

__author__ = "Jorge Ruballos"
__email__ = "ruballoj@oregonstate.edu"
__date__ = "2025-11-17"
__version__ = "1.1.0"

import os
from datetime import datetime

import pandas as pd

program_name = "SMART_merge_demographics_yearly"

# ---------------------------------------------------------
# Set up paths
# ---------------------------------------------------------
base_path = "/mnt/d/DETECT"

# Processed SMART data from your SMART_processing script
SMART_PATH = os.path.join(
    base_path, "OUTPUT", "SMART_processing", "smart_cleaned.csv"
)

# Recoded yearly clinical/demographic data
DEMO_PATH = os.path.join(
    base_path, "OUTPUT", "yearly_visit_processing", "yearly_recoded.csv"
)

# Output folder for this merge step
output_path = os.path.join(base_path, "OUTPUT", program_name)
os.makedirs(output_path, exist_ok=True)

# ---------------------------------------------------------
# Logging helper
# ---------------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logfile = os.path.join(output_path, f"pipeline_log_{timestamp}.txt")


def log_step(message: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{ts}] {message}"
    print(full_msg)
    with open(logfile, "a") as f:
        f.write(full_msg + "\n")


# =========================================================
# Step 0: Load processed SMART data and derive SMART year
# =========================================================
log_step(f"Step 0: Loading processed SMART data from {SMART_PATH}")

if not os.path.exists(SMART_PATH):
    raise FileNotFoundError(
        f"Processed SMART data not found at {SMART_PATH}. "
        "Run SMART_processing first or update SMART_PATH."
    )

smart_df = pd.read_csv(SMART_PATH)

if "subid" not in smart_df.columns:
    raise KeyError("Column 'subid' not found in SMART data.")

if "date" not in smart_df.columns:
    raise KeyError(
        "Column 'date' not found in SMART data. "
        "Check that SMART_processing renamed RecordedDate to 'date'."
    )

# Ensure date is datetime and create SMART year
smart_df["date"] = pd.to_datetime(smart_df["date"], errors="coerce")
smart_df["smart_year"] = smart_df["date"].dt.year

num_subids_0 = smart_df["subid"].nunique()
total_rows_0 = len(smart_df)
log_step(
    f"Step 0: SMART data has {num_subids_0} unique subids and {total_rows_0} total rows."
)

row_counts_0 = smart_df["subid"].value_counts().sort_index()
row_counts_0.to_csv(os.path.join(output_path, "row_counts_step_0_smart.csv"))

missing_0 = smart_df.isnull().sum()
missing_0[missing_0 > 0].to_csv(
    os.path.join(output_path, "missing_counts_step_0_smart.csv")
)

# =========================================================
# Step 1: Load and subset yearly demographic data
# =========================================================
log_step(f"Step 1: Loading yearly demographic data from {DEMO_PATH}")

if not os.path.exists(DEMO_PATH):
    raise FileNotFoundError(
        f"Demographic file not found at {DEMO_PATH}. "
        "Check yearly_visit_processing output or update DEMO_PATH."
    )

demo_df = pd.read_csv(DEMO_PATH)

# Required keys
if "subid" not in demo_df.columns:
    raise KeyError(
        "Column 'subid' not found in yearly_recoded demographic data."
    )
if "visityr_cr" not in demo_df.columns:
    raise KeyError(
        "Column 'visityr_cr' (visit year) not found in yearly_recoded demographic data."
    )

# Convert visityr_cr to integer year if possible
demo_df["visityr_cr"] = pd.to_numeric(
    demo_df["visityr_cr"], errors="coerce"
).astype("Int64")

# ---------------------------------------------------------
# Choose which demographic columns you want to keep.
# (Your specified columns.)
# ---------------------------------------------------------
DEMO_COLS = [
    "subid",
    "visityr_cr",       # visit year (used to align with SMART year)
    "sex",              # sex variable
    "educ_group",       # recoded education category
    "age_bucket",       # age bucket
    "moca_category",    # MoCA category for that visit year
    "race_group",       # 3-level race grouping
    "maristat_recoded", # marital status recoded
    "livsitua_recoded", # living situation recoded
]

available_demo_cols = [c for c in DEMO_COLS if c in demo_df.columns]

if len(available_demo_cols) < 2:
    log_step(
        "Warning: Very few of the requested demographic columns are present. "
        f"Available columns: {available_demo_cols}"
    )

log_step(f"Step 1: Keeping demographic columns: {available_demo_cols}")
demo_sub = demo_df[available_demo_cols].drop_duplicates(
    subset=["subid", "visityr_cr"]
)

num_subids_demo = demo_sub["subid"].nunique()
total_rows_demo = len(demo_sub)
log_step(
    f"Step 1: Demographic subset has {num_subids_demo} unique subids and "
    f"{total_rows_demo} rows after selecting columns and dropping duplicates."
)

row_counts_demo = demo_sub["subid"].value_counts().sort_index()
row_counts_demo.to_csv(os.path.join(output_path, "row_counts_step_1_demo.csv"))

missing_demo = demo_sub.isnull().sum()
missing_demo[missing_demo > 0].to_csv(
    os.path.join(output_path, "missing_counts_step_1_demo.csv")
)

# Identify the value columns (everything except keys)
demo_value_cols = [
    c for c in available_demo_cols if c not in ("subid", "visityr_cr")
]

# =========================================================
# Step 2: Primary merge by (subid, smart_year) <-> (subid, visityr_cr)
# =========================================================
log_step(
    "Step 2: Primary merge of SMART data with yearly demographics "
    "on (subid, smart_year) <-> (subid, visityr_cr)."
)

merged_df = smart_df.merge(
    demo_sub,
    left_on=["subid", "smart_year"],
    right_on=["subid", "visityr_cr"],
    how="left",
)

num_subids_2 = merged_df["subid"].nunique()
total_rows_2 = len(merged_df)
log_step(
    f"Step 2: After primary merge, dataset has {num_subids_2} unique subids "
    f"and {total_rows_2} total rows."
)

row_counts_2 = merged_df["subid"].value_counts().sort_index()
row_counts_2.to_csv(os.path.join(output_path, "row_counts_step_2_merged_year.csv"))

missing_2_before_fb = merged_df.isnull().sum()
missing_2_before_fb[missing_2_before_fb > 0].to_csv(
    os.path.join(output_path, "missing_counts_step_2_before_fallback.csv")
)

# =========================================================
# Step 2b: Fallback merge by subid only for rows with no year-aligned demo
# =========================================================
if demo_value_cols:
    # Rows where ALL demographic value columns are NaN -> no year-aligned match
    mask_missing_demo = merged_df[demo_value_cols].isna().all(axis=1)
    n_missing_demo = int(mask_missing_demo.sum())
    log_step(
        f"Step 2b: Found {n_missing_demo} SMART rows with no year-aligned "
        "demographics. Attempting fallback merge by subid."
    )

    if n_missing_demo > 0:
        # For fallback, choose one demographic row per subid.
        # Here we take the LATEST visit year (max visityr_cr) for each subid.
        demo_fallback = (
            demo_sub.sort_values(["subid", "visityr_cr"])
            .dropna(subset=["visityr_cr"])
        )
        demo_fallback = demo_fallback.groupby("subid", as_index=False).last()

        # Merge only the rows that need fallback
        fb = merged_df.loc[mask_missing_demo].merge(
            demo_fallback,
            on="subid",
            how="left",
            suffixes=("", "_fb"),
        )

        # Fill missing demographic values from fallback columns
        for col in demo_value_cols:
            fb[col] = fb[col].fillna(fb.get(f"{col}_fb"))

        # Optional: if you want a fallback visityr, you could also
        # fill visityr_cr where missing using visityr_cr_fb.
        if "visityr_cr" in fb.columns and "visityr_cr_fb" in fb.columns:
            fb["visityr_cr"] = fb["visityr_cr"].fillna(fb["visityr_cr_fb"])

        # Drop the temporary *_fb columns
        drop_cols = [c for c in fb.columns if c.endswith("_fb")]
        fb = fb.drop(columns=drop_cols, errors="ignore")

        # Put the fallback rows back into merged_df
        merged_df.loc[mask_missing_demo, :] = fb

else:
    log_step(
        "Step 2b: No demographic value columns identified, "
        "skipping fallback merge by subid."
    )

# Recompute missingness after fallback
missing_2_after_fb = merged_df.isnull().sum()
missing_2_after_fb[missing_2_after_fb > 0].to_csv(
    os.path.join(output_path, "missing_counts_step_2_after_fallback.csv")
)

# =========================================================
# Step 3: Save final merged dataset & summaries
# =========================================================
output_final = os.path.join(output_path, "smart_with_yearly_demographics.csv")
merged_df.to_csv(output_final, index=False)
log_step(f"Step 3: Final merged data saved to: {output_final}")

# Summary table of row counts across main steps
summary = pd.DataFrame(
    {
        "unique_subids": [num_subids_0, num_subids_demo, num_subids_2],
        "n_rows": [total_rows_0, total_rows_demo, total_rows_2],
    },
    index=["step_0_smart_only", "step_1_demo_subset", "step_2_after_merge"],
)
summary.to_csv(os.path.join(output_path, "summary_pipeline_overview.csv"))
log_step("Summary of row counts saved to summary_pipeline_overview.csv")

# Missingness summary across key steps
missing_all = pd.DataFrame(
    {
        "step_0_smart": missing_0,
        "step_1_demo": missing_demo,
        "step_2_before_fallback": missing_2_before_fb,
        "step_2_after_fallback": missing_2_after_fb,
    }
).fillna(0).astype(int)

missing_all.to_csv(os.path.join(output_path, "missing_counts_summary.csv"))
log_step("Missing value summary saved to missing_counts_summary.csv")

log_step("SMART + yearly demographics (year-aligned with fallback) pipeline completed successfully.")
