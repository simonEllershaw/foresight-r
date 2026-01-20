#!/usr/bin/env python
# Copied from https://github.com/ipolharvard/ethos-ares/blob/2d54383997318eb52f3d47b5969a66fc166b71ff/scripts/meds/mimic/pre_MEDS.py

"""Performs pre-MEDS data wrangling for MIMIC-IV."""

from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

import hydra
import polars as pl
from loguru import logger
from MEDS_transforms.extract.utils import get_supported_fp
from MEDS_transforms.utils import get_shard_prefix, write_lazyframe
from omegaconf import DictConfig

END_OF_DAY = pl.time(23, 59, 59, 999999)


def join_time_column(
    target_df: pl.LazyFrame,
    time_source_df: pl.LazyFrame,
    on: str,
    time_column_name: str,
) -> pl.LazyFrame:
    """Join time column from source dataframe to target dataframe by specified key."""
    time_source_df = time_source_df.select(on, time_column_name)
    return target_df.join(time_source_df, on=on, how="left")


# Helper partials for common time joins
add_dischtime = partial(join_time_column, on="hadm_id", time_column_name="dischtime")
add_outtime = partial(join_time_column, on="stay_id", time_column_name="outtime")
add_intime = partial(join_time_column, on="stay_id", time_column_name="intime")


def convert_date_to_end_of_day_timestamp(
    df: pl.LazyFrame, date_column: str = "chartdate"
) -> pl.LazyFrame:
    """Converts a date-only column to a timestamp set at the end of the day (23:59:59.999999)."""
    return df.with_columns(
        pl.col(date_column).str.to_date().dt.combine(END_OF_DAY).alias(date_column)
    )


def process_lab_events_df(
    lab_events_df: pl.LazyFrame, d_labitems_df: pl.LazyFrame
) -> pl.LazyFrame:
    """Filters lab events to rows with non-null numeric values, joins with `d_labitems` to add item labels and fluid types, and fills null `valueuom` with empty strings."""

    lab_events_df = lab_events_df.filter(pl.col("valuenum").is_not_null())

    d_labitems_df = d_labitems_df.select("itemid", "fluid", "label")
    result = lab_events_df.join(d_labitems_df, on="itemid", how="left")

    # Convert null values in valueuom column to empty strings
    result = result.with_columns(pl.col("valueuom").fill_null(""))
    return result


def process_drgcodes_df(
    drgcodes_df: pl.LazyFrame, admissions_df: pl.LazyFrame
) -> pl.LazyFrame:
    """Filters for 'HCFA' DRG codes (as in ETHOS-ARES) and adds the discharge time from admissions."""
    drgcodes_df = drgcodes_df.filter(pl.col("drg_type") == "HCFA")
    return add_dischtime(drgcodes_df, admissions_df)


def process_hosp_transfers_df(hosp_transfers_df: pl.LazyFrame) -> pl.LazyFrame:
    """Maps raw `eventtype` values ('discharge', 'admit', 'transfer') to descriptive text ('Discharge from', 'Admit to', 'Transfer to')."""
    return hosp_transfers_df.with_columns(
        pl.col("eventtype").replace_strict(
            {
                "discharge": "Discharge from",
                "admit": "Admit to",
                "transfer": "Transfer to",
                # Care unit is always Emergency Department for "ED" eventtypes
                # So no need to repeat it here
                "ED": "",
            },
            default=pl.col("eventtype"),
        )
    )


def process_icd_df(
    icd_df: pl.LazyFrame, admissions_df: pl.LazyFrame, d_icd_df: pl.LazyFrame
) -> pl.LazyFrame:
    """Adds discharge time from admissions, joins with `d_icd` for descriptive titles, ensures codes are strings, and converts `chartdate` to end-of-day timestamps."""
    icd_df = add_dischtime(icd_df, admissions_df)
    # Cast to string to match d_icd schema
    icd_df = icd_df.with_columns(
        pl.col("icd_code").cast(pl.String),
        pl.col("icd_version").cast(pl.String),
    )
    icd_df = icd_df.join(
        d_icd_df.select("icd_code", "icd_version", "long_title"),
        on=["icd_code", "icd_version"],
        how="left",
    )
    # Convert chartdate to timestamp at 23:59:59 (applies to procedures_icd)
    if "chartdate" in icd_df.schema:
        icd_df = convert_date_to_end_of_day_timestamp(icd_df)
    return icd_df


def process_ed_diagnosis_df(
    ed_diagnosis_df: pl.LazyFrame, edstays_df: pl.LazyFrame
) -> pl.LazyFrame:
    """Adds the ED departure time (`outtime`) from `edstays`."""
    return add_outtime(ed_diagnosis_df, edstays_df)


def _round_to_3sf(expr: pl.Expr) -> pl.Expr:
    """Round a numeric expression to 3 significant figures as float."""
    # Use native Polars operations for better performance
    log_val = expr.abs().log10()
    scale = (2 - log_val.floor()).cast(pl.Int32)
    return (expr * (10**scale)).round(0) / (10**scale)


def _convert_fahrenheit_to_celsius(df: pl.LazyFrame) -> pl.LazyFrame:
    """Convert Fahrenheit temperatures to Celsius (assumes F if > 45)."""
    temp_expr = pl.col("temperature").cast(pl.Float64, strict=False)
    converted_temp = (
        pl.when(temp_expr.is_not_null() & (temp_expr > 45))
        .then((temp_expr - 32) * 5 / 9)
        .otherwise(temp_expr)
    )
    return df.with_columns(_round_to_3sf(converted_temp).alias("temperature"))


def process_ed_vitals_with_time_df(
    ed_vitals_with_time_df: pl.LazyFrame, edstays_df: pl.LazyFrame
) -> pl.LazyFrame:
    """Adds ED registration time (`intime`) and converts temperatures > 45 (assumed Fahrenheit) to Celsius."""
    ed_vitals_with_time_df = add_intime(ed_vitals_with_time_df, edstays_df)
    return _convert_fahrenheit_to_celsius(ed_vitals_with_time_df)


def process_hcpcs_events_df(
    hcpcs_events_df: pl.LazyFrame, d_hcpcs_df: pl.LazyFrame
) -> pl.LazyFrame:
    """Joins with `d_hcpcs` to add procedure descriptions (preferring long, falling back to short) and converts `chartdate` to end-of-day timestamps."""
    d_hcpcs_df = d_hcpcs_df.select(
        "code",
        pl.coalesce("long_description", "short_description").alias("long_description"),
    )
    hcpcs_events_df = hcpcs_events_df.join(
        d_hcpcs_df, left_on="hcpcs_cd", right_on="code", how="left"
    )
    return convert_date_to_end_of_day_timestamp(hcpcs_events_df)


def join_hosp_admissions_with_gender(
    admissions_df: pl.LazyFrame, patients_df: pl.LazyFrame
) -> pl.LazyFrame:
    """Joins the `gender` column from the patients table to the admissions table, standardizing 'F'/'M' to 'Female'/'Male'."""
    return admissions_df.join(
        patients_df.select(
            "subject_id",
            pl.col("gender")
            .replace_strict({"F": "Female", "M": "Male"}, default=pl.col("gender"))
            .alias("gender"),
        ),
        on="subject_id",
        how="left",
    )


def process_patients_df(
    patients_df: pl.LazyFrame, death_times_df: pl.LazyFrame
) -> pl.LazyFrame:
    """Aggregates death times, calculates birth years, standardizes gender, and consolidates death timestamps (preferring admission `deathtime` over `dod`)."""
    # Parse datetime once before aggregation for better performance
    death_times_df = (
        death_times_df.with_columns(pl.col("deathtime").str.strptime(pl.Datetime))
        .group_by("subject_id")
        .agg(pl.col("deathtime").min())
    )

    return patients_df.join(death_times_df, on="subject_id", how="left").select(
        "subject_id",
        # If we have a deathtime from admissions, use it; otherwise use dod converted to timestamp at end of day
        pl.coalesce(
            pl.col("deathtime"),
            # dod is date only, convert to timestamp at end of day
            pl.col("dod").str.strptime(pl.Date).dt.combine(END_OF_DAY),
        ).alias("dod"),
        (pl.col("anchor_year") - pl.col("anchor_age")).cast(str).alias("year_of_birth"),
        pl.col("gender")
        .replace_strict({"F": "Female", "M": "Male"}, default=pl.col("gender"))
        .alias("gender"),
    )


# Processing functions registry for MIMIC-IV data files.
# Format: {
#     "relative/path/to/file": (
#         processing_function,
#         [("dependency/path", ["required", "columns"]), ...] or None
#     )
# }
# - Key: Relative path from MIMIC data root (e.g., "hosp/admissions")
# - Value: Tuple of (function, dependencies)
#   - function: Callable that processes the dataframe
#   - dependencies: List of (path, columns) tuples for required reference tables,
#                   or None if no dependencies needed
FUNCTIONS = {
    "hosp/diagnoses_icd": (
        process_icd_df,
        [
            ("hosp/admissions", ["hadm_id", "dischtime"]),
            ("hosp/d_icd_diagnoses", ["icd_code", "icd_version", "long_title"]),
        ],
    ),
    "hosp/procedures_icd": (
        process_icd_df,
        [
            ("hosp/admissions", ["hadm_id", "dischtime"]),
            ("hosp/d_icd_procedures", ["icd_code", "icd_version", "long_title"]),
        ],
    ),
    "hosp/drgcodes": (
        process_drgcodes_df,
        [("hosp/admissions", ["hadm_id", "dischtime"])],
    ),
    "hosp/patients": (
        process_patients_df,
        [("hosp/admissions", ["subject_id", "deathtime"])],
    ),
    "hosp/transfers": (process_hosp_transfers_df, None),
    "hosp/labevents": (
        process_lab_events_df,
        [("hosp/d_labitems", ["itemid", "label", "fluid"])],
    ),
    "hosp/hcpcsevents": (
        process_hcpcs_events_df,
        [("hosp/d_hcpcs", ["code", "long_description", "short_description"])],
    ),
    "hosp/omr": (convert_date_to_end_of_day_timestamp, None),
    "ed/diagnosis": (process_ed_diagnosis_df, [("ed/edstays", ["stay_id", "outtime"])]),
    "ed/triage": (
        process_ed_vitals_with_time_df,
        [("ed/edstays", ["stay_id", "intime"])],
    ),
    "ed/vitalsign": (
        process_ed_vitals_with_time_df,
        [("ed/edstays", ["stay_id", "intime"])],
    ),
    "hosp/admissions": (
        join_hosp_admissions_with_gender,
        [("hosp/patients", ["subject_id", "gender"])],
    ),
}


@hydra.main(version_base=None, config_path="configs", config_name="pre_MEDS")
def main(cfg: DictConfig):
    """Performs pre-MEDS data wrangling for MIMIC-IV.

    Inputs are the raw MIMIC files, read from the `input_dir` config parameter. Output files are
    either symlinked (if they are not modified) or written in processed form to the `MEDS_input_dir`
    config parameter. Hydra is used to manage configuration parameters and logging.

    Big Picture:
    -----------
    This function transforms raw MIMIC-IV data files into a format suitable for MEDS processing.
    Some files require no transformation (symlinked), others need simple processing, and some
    require joining with lookup tables for enrichment.

    Processing occurs in three phases:

    Phase 1 - Independent File Processing:
        - Discovers all input files recursively
        - For files without dependencies: processes immediately or symlinks if no processing needed
        - For files with dependencies: defers processing and tracks what dependencies are needed

    Phase 2 - Dependency Loading:
        - Loads all required lookup tables and reference dataframes into memory
        - Optimizes memory usage by loading only the columns actually needed
        - Handles special cases like ICD code tables that need string preservation

    Phase 3 - Dependent File Processing:
        - Processes files that require dependencies, now that dependencies are loaded
        - Joins main dataframes with their required lookup tables
        - Ensures all dependencies are available before attempting processing

    This approach minimizes memory usage and processing time by loading dependencies once
    and reusing them across multiple files that need the same reference data.
    """

    # Set up input and output directory paths from configuration
    input_dir = Path(cfg.input_dir)
    MEDS_input_dir = Path(cfg.cohort_dir)

    # Check if processing has already been completed (to avoid re-processing)
    done_fp = MEDS_input_dir / ".done"
    if done_fp.is_file() and not cfg.do_overwrite:
        logger.info(
            f"Pre-MEDS transformation already complete as {done_fp} exists and "
            f"do_overwrite={cfg.do_overwrite}. Returning."
        )
        exit(0)

    # Discover all data files in the input directory (including subdirectories)
    all_fps = input_dir.rglob("*.*")

    # Initialize data structures to track files that need dependencies loaded
    dfs_to_load: dict[str, dict[str, set]] = {}  # Files requiring other dataframes
    seen_fps = {}  # Cache of file paths and their read functions
    created_dirs = set()  # Track created directories to avoid redundant mkdir calls

    # PHASE 1: Process files without dependencies or create symlinks
    for in_fp in all_fps:
        # Get the standardized prefix for this file (e.g., 'hosp/admissions')
        pfx = get_shard_prefix(input_dir, in_fp)

        # Try to find a compatible dataframe file and get its read function
        try:
            fp, read_fn = get_supported_fp(input_dir, pfx)
        except FileNotFoundError:
            logger.info(
                f"Skipping {pfx} @ {str(in_fp.resolve())} as "
                "no compatible dataframe file was found."
            )
            continue

        # Configure CSV reading with extended schema inference for better type detection
        if fp.suffix in (".csv", ".csv.gz"):
            read_fn = partial(read_fn, infer_schema_length=100000)

        # Avoid processing the same file multiple times
        fp_key = str(fp.resolve())
        if fp_key in seen_fps:
            continue
        seen_fps[fp_key] = read_fn

        # Determine output file path, maintaining directory structure
        out_fp = MEDS_input_dir / fp.relative_to(input_dir)

        # Skip if already processed
        if out_fp.is_file():
            logger.debug(f"Already processed {pfx}, skipping")
            continue

        # Create output directory structure (only if not already created)
        out_dir = out_fp.parent
        if out_dir not in created_dirs:
            out_dir.mkdir(parents=True, exist_ok=True)
            created_dirs.add(out_dir)

        if pfx not in FUNCTIONS:
            logger.info(f"No function needed for {pfx}: Symlinking to output")
            relative_in_fp = fp.relative_to(out_fp.resolve().parent, walk_up=True)
            out_fp.symlink_to(relative_in_fp)
            continue

        # File requires processing
        out_fp = MEDS_input_dir / f"{pfx}.parquet"
        fn, need_dfs = FUNCTIONS[pfx]

        # If this file doesn't need dependencies, process it immediately
        if not need_dfs:
            st = datetime.now()
            logger.info(f"Processing {pfx}...")
            df = read_fn(fp)
            logger.info(f"  Loaded in {datetime.now() - st}")
            processed_df = fn(df)  # type: ignore
            write_lazyframe(processed_df, out_fp)
            logger.info(f"  Wrote to {out_fp.name} in {datetime.now() - st}")
            continue

        # File needs dependencies - defer processing and track requirements
        for needed_pfx, needed_cols in need_dfs:
            if needed_pfx not in dfs_to_load:
                dfs_to_load[needed_pfx] = {"fps": set(), "cols": set()}
            dfs_to_load[needed_pfx]["fps"].add(fp)
            dfs_to_load[needed_pfx]["cols"].update(needed_cols)

    # PHASE 2: Load all dependency dataframes (lookup tables, etc.)
    loaded_dfs: dict[str, pl.LazyFrame] = {}
    for df_to_load_pfx, fps_and_cols in dfs_to_load.items():
        # Get only the columns we actually need from this dependency
        cols = list(fps_and_cols["cols"])
        df_to_load_fp, df_to_load_read_fn = get_supported_fp(input_dir, df_to_load_pfx)

        st = datetime.now()
        logger.info(f"Loading dependency {df_to_load_pfx}...")

        # Special handling: ICD tables need string schema to preserve leading zeros
        read_kwargs: dict[str, Any] = {}
        if df_to_load_pfx in ("hosp/d_icd_diagnoses", "hosp/d_icd_procedures"):
            read_kwargs["schema_overrides"] = {
                "icd_code": pl.String,
                "icd_version": pl.String,
            }
        # Load only needed columns for compressed files to save memory
        if df_to_load_fp.suffix == ".csv.gz":
            read_kwargs["columns"] = cols

        loaded_dfs[df_to_load_pfx] = df_to_load_read_fn(df_to_load_fp, **read_kwargs)
        logger.info(f"  Loaded in {datetime.now() - st}")

    # PHASE 3: Process files that have dependencies (now that dependencies are loaded)
    processed_fps: set[str] = set()
    for df_to_load_pfx, fps_and_cols in dfs_to_load.items():
        fps = fps_and_cols["fps"]

        # Process each file that depends on this loaded dataframe
        for fp in fps:
            fp_key = str(fp.resolve())
            if fp_key in processed_fps:
                continue

            pfx = get_shard_prefix(input_dir, fp)
            out_fp = MEDS_input_dir / f"{pfx}.parquet"

            # Skip if already processed
            if out_fp.is_file():
                processed_fps.add(fp_key)
                continue

            fn, need_dfs = FUNCTIONS[pfx]

            # Ensure all required dependencies are available before processing
            if need_dfs is None:
                continue
            dep_pfxs = [dep_pfx for dep_pfx, _ in need_dfs]
            if not all(dep_pfx in loaded_dfs for dep_pfx in dep_pfxs):
                continue

            logger.info(f"  Processing {pfx} with dependencies...")

            # Load the main dataframe and apply transformation with dependencies
            fp_st = datetime.now()
            fp_df = seen_fps[fp_key](fp)
            logger.info(f"    Loaded in {datetime.now() - fp_st}")

            # Pass the main dataframe plus all dependency dataframes to the processing function
            dep_dfs = [loaded_dfs[dep_pfx] for dep_pfx in dep_pfxs]
            processed_df = fn(fp_df, *dep_dfs)  # type: ignore
            write_lazyframe(processed_df, out_fp)
            logger.info(f"    Wrote to {out_fp.name} in {datetime.now() - fp_st}")
            processed_fps.add(fp_key)

    # Mark processing as complete
    done_fp.write_text(f"Finished at {datetime.now()}")
    logger.info(f"Pre-MEDS processing complete. Output in {MEDS_input_dir}")


if __name__ == "__main__":
    main()
