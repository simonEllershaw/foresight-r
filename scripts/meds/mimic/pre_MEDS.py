#!/usr/bin/env python
# Copied from https://github.com/ipolharvard/ethos-ares/blob/2d54383997318eb52f3d47b5969a66fc166b71ff/scripts/meds/mimic/pre_MEDS.py

"""Performs pre-MEDS data wrangling for MIMIC-IV."""

from datetime import datetime
from functools import partial
from pathlib import Path

import hydra
import polars as pl
from loguru import logger
from MEDS_transforms.extract.utils import get_supported_fp
from MEDS_transforms.utils import get_shard_prefix, write_lazyframe
from omegaconf import DictConfig


def add_dot(code: pl.Expr, position: int) -> pl.Expr:
    """Adds a dot to the code expression at the specified position.

    Args:
        code: The code expression.
        position: The position to add the dot.

    Returns:
        The expression which would yield the code string with a dot added at the specified position

    Example:
        >>> pl.select(add_dot(pl.lit("12345"), 3))
        shape: (1, 1)
        ┌─────────┐
        │ literal │
        │ ---     │
        │ str     │
        ╞═════════╡
        │ 123.45  │
        └─────────┘
        >>> pl.select(add_dot(pl.lit("12345"), 1))
        shape: (1, 1)
        ┌─────────┐
        │ literal │
        │ ---     │
        │ str     │
        ╞═════════╡
        │ 1.2345  │
        └─────────┘
        >>> pl.select(add_dot(pl.lit("12345"), 6))
        shape: (1, 1)
        ┌─────────┐
        │ literal │
        │ ---     │
        │ str     │
        ╞═════════╡
        │ 12345   │
        └─────────┘
    """
    return (
        pl.when(code.str.len_chars() > position)
        .then(code.str.slice(0, position) + "." + code.str.slice(position))
        .otherwise(code)
    )


def add_icd_diagnosis_dot(icd_version: pl.Expr, icd_code: pl.Expr) -> pl.Expr:
    """Adds the appropriate dot to the ICD diagnosis codebased on the version.

    Args:
        icd_version: The ICD version.
        icd_code: The ICD code.

    Returns:
        The ICD code with appropriate dot syntax based on the version.

    Examples:
        >>> pl.select(add_icd_diagnosis_dot(pl.lit("9"), pl.lit("12345")))
        shape: (1, 1)
        ┌─────────┐
        │ literal │
        │ ---     │
        │ str     │
        ╞═════════╡
        │ 123.45  │
        └─────────┘
        >>> pl.select(add_icd_diagnosis_dot(pl.lit("9"), pl.lit("E1234")))
        shape: (1, 1)
        ┌─────────┐
        │ literal │
        │ ---     │
        │ str     │
        ╞═════════╡
        │ E123.4  │
        └─────────┘
        >>> pl.select(add_icd_diagnosis_dot(pl.lit("9"), pl.lit("F1234")))
        shape: (1, 1)
        ┌─────────┐
        │ literal │
        │ ---     │
        │ str     │
        ╞═════════╡
        │ F12.34  │
        └─────────┘
        >>> pl.select(add_icd_diagnosis_dot(pl.lit("10"), pl.lit("12345")))
        shape: (1, 1)
        ┌─────────┐
        │ literal │
        │ ---     │
        │ str     │
        ╞═════════╡
        │ 123.45  │
        └─────────┘
        >>> pl.select(add_icd_diagnosis_dot(pl.lit("10"), pl.lit("E1234")))
        shape: (1, 1)
        ┌─────────┐
        │ literal │
        │ ---     │
        │ str     │
        ╞═════════╡
        │ E12.34  │
        └─────────┘
    """

    icd9_code = (
        pl.when(icd_code.str.starts_with("E"))
        .then(add_dot(icd_code, 4))
        .otherwise(add_dot(icd_code, 3))
    )

    icd10_code = add_dot(icd_code, 3)

    return pl.when(icd_version == "9").then(icd9_code).otherwise(icd10_code)


def add_icd_procedure_dot(icd_version: pl.Expr, icd_code: pl.Expr) -> pl.Expr:
    """Adds the appropriate dot to the ICD procedure code based on the version.

    Args:
        icd_version: The ICD version.
        icd_code: The ICD code.

    Returns:
        The ICD code with appropriate dot syntax based on the version.

    Examples:
        >>> pl.select(add_icd_procedure_dot(pl.lit("9"), pl.lit("12345")))
        shape: (1, 1)
        ┌─────────┐
        │ literal │
        │ ---     │
        │ str     │
        ╞═════════╡
        │ 12.345  │
        └─────────┘
        >>> pl.select(add_icd_procedure_dot(pl.lit("10"), pl.lit("12345")))
        shape: (1, 1)
        ┌─────────┐
        │ literal │
        │ ---     │
        │ str     │
        ╞═════════╡
        │ 12345   │
        └─────────┘
    """

    icd9_code = add_dot(icd_code, 2)
    icd10_code = icd_code

    return pl.when(icd_version == "9").then(icd9_code).otherwise(icd10_code)


def add_time_by_id(
    df: pl.LazyFrame, time_source_df: pl.LazyFrame, on: str, time_column_name: str
) -> pl.LazyFrame:
    """Joins the two dataframes by ``on`` and adds the time to the original dataframe."""

    time_source_df = time_source_df.select(on, time_column_name)
    return df.join(time_source_df, on=on, how="left")


add_discharge_time_by_hadm_id = partial(
    add_time_by_id, on="hadm_id", time_column_name="dischtime"
)
add_out_time_by_stay_id = partial(
    add_time_by_id, on="stay_id", time_column_name="outtime"
)
add_reg_time_by_stay_id = partial(
    add_time_by_id, on="stay_id", time_column_name="intime"
)


def format_to_3sf(expr: pl.Expr) -> pl.Expr:
    """Format a numeric expression to 3 significant figures as a string."""
    return expr.map_elements(
        lambda x: f"{x:.3g}" if x is not None else None, return_dtype=pl.String
    )


def round_to_3sf(expr: pl.Expr) -> pl.Expr:
    """Round a numeric expression to 3 significant figures as float."""
    return expr.map_elements(
        lambda x: float(f"{x:.3g}") if x is not None else None,
        return_dtype=pl.Float64,
    )


def add_lab_item_text(df: pl.LazyFrame, d_labitems_df: pl.LazyFrame) -> pl.LazyFrame:
    """Joins labevents with d_labitems to add item_text, fluid, and merged value columns."""
    d_labitems_df = d_labitems_df.select("itemid", "fluid", "label")
    result = df.join(d_labitems_df, on="itemid", how="left")
    # Convert null values in valueuom column to empty strings
    result = result.with_columns(pl.col("valueuom").fill_null(""))
    return result


def normalize_edstays(df: pl.LazyFrame) -> pl.LazyFrame:
    """Convert arrival_transport and disposition columns to title case."""
    return df.with_columns(
        pl.col("arrival_transport").str.to_titlecase(),
        pl.col("disposition").str.to_titlecase(),
    )


def normalize_hosp_admissions(df: pl.LazyFrame) -> pl.LazyFrame:
    """Convert admission_type, admission_location, discharge_location, and race to title case."""
    return df.with_columns(
        pl.col("admission_type").str.to_titlecase(),
        pl.col("admission_location").str.to_titlecase(),
        pl.col("discharge_location").str.to_titlecase(),
        pl.col("race").str.to_titlecase(),
    )


def normalize_hosp_drgcodes_with_time(
    df: pl.LazyFrame, admissions_df: pl.LazyFrame
) -> pl.LazyFrame:
    """Add discharge time and convert description to title case."""
    df = add_discharge_time_by_hadm_id(df, admissions_df)
    return df.with_columns(pl.col("description").str.to_titlecase())


def normalize_hosp_transfers(df: pl.LazyFrame) -> pl.LazyFrame:
    """Map eventtype values to descriptive text."""
    return df.with_columns(
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


def add_icd_info(
    df: pl.LazyFrame, admissions_df: pl.LazyFrame, d_icd_df: pl.LazyFrame
) -> pl.LazyFrame:
    """Add discharge time from admissions and long_title from an ICD reference table."""
    df = add_discharge_time_by_hadm_id(df, admissions_df)
    # Cast to string to match d_icd schema
    df = df.with_columns(
        pl.col("icd_code").cast(pl.String),
        pl.col("icd_version").cast(pl.String),
    )
    return df.join(
        d_icd_df.select("icd_code", "icd_version", "long_title"),
        on=["icd_code", "icd_version"],
        how="left",
    )


def normalize_ed_diagnosis(df: pl.LazyFrame, edstays_df: pl.LazyFrame) -> pl.LazyFrame:
    """Add outtime and convert icd_title to title case."""
    df = add_out_time_by_stay_id(df, edstays_df)
    return df.with_columns(pl.col("icd_title").str.to_titlecase())


def convert_fahrenheit_to_celsius(df: pl.LazyFrame) -> pl.LazyFrame:
    """Convert Fahrenheit temperatures to Celsius (assumes F if > 45)."""
    temp_expr = pl.col("temperature").cast(pl.Float64, strict=False)
    converted_temp = (
        pl.when(temp_expr.is_not_null() & (temp_expr > 45))
        .then((temp_expr - 32) * 5 / 9)
        .otherwise(temp_expr)
    )
    return df.with_columns(round_to_3sf(converted_temp).alias("temperature"))


def normalize_ed_triage(df: pl.LazyFrame, edstays_df: pl.LazyFrame) -> pl.LazyFrame:
    """Add registration time and convert Fahrenheit temperatures to Celsius."""
    df = add_reg_time_by_stay_id(df, edstays_df)
    return convert_fahrenheit_to_celsius(df)


def normalize_ed_vitalsign(df: pl.LazyFrame, edstays_df: pl.LazyFrame) -> pl.LazyFrame:
    """Add registration time and convert Fahrenheit temperatures to Celsius."""
    df = add_reg_time_by_stay_id(df, edstays_df)
    return convert_fahrenheit_to_celsius(df)


def add_hcpcs_description(df: pl.LazyFrame, d_hcpcs_df: pl.LazyFrame) -> pl.LazyFrame:
    """Join hcpcsevents with d_hcpcs to add long_description column.

    Uses long_description from d_hcpcs if not null, otherwise uses short_description
    from d_hcpcs.
    """
    d_hcpcs_df = d_hcpcs_df.select(
        "code",
        pl.coalesce("long_description", "short_description").alias("long_description"),
    )
    return df.join(d_hcpcs_df, left_on="hcpcs_cd", right_on="code", how="left")


def fix_static_data(
    raw_static_df: pl.LazyFrame, death_times_df: pl.LazyFrame
) -> pl.LazyFrame:
    """Fixes the static data by adding the death time to the static data and fixes the DOB nonsense.

    Args:
        raw_static_df: The raw static data.
        death_times_df: The death times data.

    Returns:
        The fixed static data.
    """

    death_times_df = death_times_df.group_by("subject_id").agg(
        pl.col("deathtime").min()
    )

    return raw_static_df.join(death_times_df, on="subject_id", how="left").select(
        "subject_id",
        pl.coalesce(pl.col("deathtime"), pl.col("dod")).alias("dod"),
        (pl.col("anchor_year") - pl.col("anchor_age")).cast(str).alias("year_of_birth"),
        pl.col("gender")
        .replace_strict({"F": "Female", "M": "Male"}, default=pl.col("gender"))
        .alias("gender"),
    )


FUNCTIONS = {
    "hosp/diagnoses_icd": (
        add_icd_info,
        [
            ("hosp/admissions", ["hadm_id", "dischtime"]),
            ("hosp/d_icd_diagnoses", ["icd_code", "icd_version", "long_title"]),
        ],
    ),
    "hosp/procedures_icd": (
        add_icd_info,
        [
            ("hosp/admissions", ["hadm_id", "dischtime"]),
            ("hosp/d_icd_procedures", ["icd_code", "icd_version", "long_title"]),
        ],
    ),
    "hosp/drgcodes": (
        normalize_hosp_drgcodes_with_time,
        [("hosp/admissions", ["hadm_id", "dischtime"])],
    ),
    "hosp/patients": (
        fix_static_data,
        [("hosp/admissions", ["subject_id", "deathtime"])],
    ),
    "hosp/admissions": (normalize_hosp_admissions, None),
    "hosp/transfers": (normalize_hosp_transfers, None),
    "hosp/labevents": (
        add_lab_item_text,
        [("hosp/d_labitems", ["itemid", "label", "fluid"])],
    ),
    "hosp/hcpcsevents": (
        add_hcpcs_description,
        [("hosp/d_hcpcs", ["code", "long_description", "short_description"])],
    ),
    "ed/edstays": (normalize_edstays, None),
    "ed/diagnosis": (normalize_ed_diagnosis, [("ed/edstays", ["stay_id", "outtime"])]),
    "ed/triage": (normalize_ed_triage, [("ed/edstays", ["stay_id", "intime"])]),
    "ed/vitalsign": (normalize_ed_vitalsign, [("ed/edstays", ["stay_id", "intime"])]),
}

ICD_DFS_TO_FIX = [
    ("hosp/d_icd_diagnoses", add_icd_diagnosis_dot),
    ("hosp/d_icd_procedures", add_icd_procedure_dot),
]


@hydra.main(version_base=None, config_path="configs", config_name="pre_MEDS")
def main(cfg: DictConfig):
    """Performs pre-MEDS data wrangling for MIMIC-IV.

    Inputs are the raw MIMIC files, read from the `input_dir` config parameter. Output files are
    either symlinked (if they are not modified) or written in processed form to the `MEDS_input_dir`
    config parameter. Hydra is used to manage configuration parameters and logging.
    """

    input_dir = Path(cfg.input_dir)
    MEDS_input_dir = Path(cfg.cohort_dir)

    done_fp = MEDS_input_dir / ".done"
    if done_fp.is_file() and not cfg.do_overwrite:
        logger.info(
            f"Pre-MEDS transformation already complete as {done_fp} exists and "
            f"do_overwrite={cfg.do_overwrite}. Returning."
        )
        exit(0)

    all_fps = list(input_dir.rglob("*.*")) + list(input_dir.rglob("*/*.*"))

    dfs_to_load: dict[str, dict[str, set | set]] = {}
    seen_fps = {}

    for in_fp in all_fps:
        pfx = get_shard_prefix(input_dir, in_fp)

        try:
            fp, read_fn = get_supported_fp(input_dir, pfx)
        except FileNotFoundError:
            logger.info(
                f"Skipping {pfx} @ {str(in_fp.resolve())} as "
                "no compatible dataframe file was found."
            )
            continue

        if fp.suffix in [".csv", ".csv.gz"]:
            read_fn = partial(read_fn, infer_schema_length=100000)

        if str(fp.resolve()) in seen_fps:
            continue
        else:
            seen_fps[str(fp.resolve())] = read_fn

        out_fp = MEDS_input_dir / fp.relative_to(input_dir)

        if out_fp.is_file():
            print(f"Done with {pfx}. Continuing")
            continue

        out_fp.parent.mkdir(parents=True, exist_ok=True)

        if pfx not in FUNCTIONS and pfx not in [p for p, _ in ICD_DFS_TO_FIX]:
            logger.info(
                f"No function needed for {pfx}: "
                f"Symlinking {str(fp.resolve())} to {str(out_fp.resolve())}"
            )
            relative_in_fp = fp.relative_to(out_fp.resolve().parent, walk_up=True)
            out_fp.symlink_to(relative_in_fp)
            continue
        elif pfx in FUNCTIONS:
            out_fp = MEDS_input_dir / f"{pfx}.parquet"
            if out_fp.is_file():
                print(f"Done with {pfx}. Continuing")
                continue

            fn, need_dfs = FUNCTIONS[pfx]
            if not need_dfs:
                st = datetime.now()
                logger.info(f"Processing {pfx}...")
                df = read_fn(fp)
                logger.info(f"  Loaded raw {fp} in {datetime.now() - st}")
                processed_df = fn(df)  # type: ignore
                write_lazyframe(processed_df, out_fp)
                logger.info(
                    f"  Processed and wrote to {str(out_fp.resolve())} in {datetime.now() - st}"
                )
            else:
                for needed_pfx, needed_cols in need_dfs:
                    if needed_pfx not in dfs_to_load:
                        dfs_to_load[needed_pfx] = {"fps": set(), "cols": set()}

                    dfs_to_load[needed_pfx]["fps"].add(fp)
                    dfs_to_load[needed_pfx]["cols"].update(needed_cols)

    # Load all dependency dataframes
    loaded_dfs: dict[str, pl.LazyFrame] = {}
    for df_to_load_pfx, fps_and_cols in dfs_to_load.items():
        cols = list(fps_and_cols["cols"])
        df_to_load_fp, df_to_load_read_fn = get_supported_fp(input_dir, df_to_load_pfx)

        st = datetime.now()
        logger.info(
            f"Loading {str(df_to_load_fp.resolve())} for manipulating other dataframes..."
        )

        # ICD tables need string schema for icd_code/icd_version
        read_kwargs = {}
        if df_to_load_pfx in ["hosp/d_icd_diagnoses", "hosp/d_icd_procedures"]:
            read_kwargs["schema_overrides"] = {
                "icd_code": pl.String,
                "icd_version": pl.String,
            }

        if df_to_load_fp.suffix in [".csv.gz"]:
            loaded_dfs[df_to_load_pfx] = df_to_load_read_fn(
                df_to_load_fp, columns=cols, **read_kwargs
            )
        else:
            loaded_dfs[df_to_load_pfx] = df_to_load_read_fn(
                df_to_load_fp, **read_kwargs
            )
        logger.info(f"  Loaded in {datetime.now() - st}")

    # Process files that have dependencies
    processed_fps: set[str] = set()
    for df_to_load_pfx, fps_and_cols in dfs_to_load.items():
        fps = fps_and_cols["fps"]

        for fp in fps:
            fp_key = str(fp.resolve())
            if fp_key in processed_fps:
                continue

            pfx = get_shard_prefix(input_dir, fp)
            out_fp = MEDS_input_dir / f"{pfx}.parquet"

            if out_fp.is_file():
                processed_fps.add(fp_key)
                continue

            fn, need_dfs = FUNCTIONS[pfx]

            # Check if all dependencies are loaded
            if need_dfs is None:
                continue
            dep_pfxs = [dep_pfx for dep_pfx, _ in need_dfs]
            if not all(dep_pfx in loaded_dfs for dep_pfx in dep_pfxs):
                continue

            logger.info(f"  Processing dependent df @ {pfx}...")

            fp_st = datetime.now()
            logger.info(f"    Loading {str(fp.resolve())}...")
            fp_df = seen_fps[fp_key](fp)
            logger.info(f"    Loaded in {datetime.now() - fp_st}")

            # Pass all dependency dfs in order
            dep_dfs = [loaded_dfs[dep_pfx] for dep_pfx in dep_pfxs]
            processed_df = fn(fp_df, *dep_dfs)  # type: ignore
            write_lazyframe(processed_df, out_fp)
            logger.info(
                f"    Processed and wrote to {str(out_fp.resolve())} in {datetime.now() - fp_st}"
            )
            processed_fps.add(fp_key)

    for pfx, fn in ICD_DFS_TO_FIX:
        fp, read_fn = get_supported_fp(input_dir, pfx)
        out_fp = MEDS_input_dir / f"{pfx}.parquet"

        if out_fp.is_file():
            print(f"Done with {pfx}. Continuing")
            continue

        if fp.suffix != ".parquet":
            read_fn = partial(read_fn, infer_schema=False)

        st = datetime.now()
        logger.info(f"Processing {pfx}...")
        processed_df = (
            read_fn(fp)
            .collect()
            .with_columns(
                fn(
                    pl.col("icd_version").cast(pl.String),
                    pl.col("icd_code").cast(pl.String),
                ).alias("norm_icd_code")
            )
        )
        processed_df.write_parquet(out_fp, use_pyarrow=True)
        logger.info(
            f"  Processed and wrote to {str(out_fp.resolve())} in {datetime.now() - st}"
        )

    logger.info(
        f"Done! All dataframes processed and written to {str(MEDS_input_dir.resolve())}"
    )
    done_fp.write_text(f"Finished at {datetime.now()}")


if __name__ == "__main__":
    main()
