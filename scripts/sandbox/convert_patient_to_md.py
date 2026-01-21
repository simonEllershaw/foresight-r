"""Convert a single patient's MEDS data to markdown format."""

import polars as pl
from pathlib import Path

END_OF_DAY = pl.time(23, 59, 59, 999999)


def changed(col: str, default: bool = False) -> pl.Expr:
    """Detect when a column value changes from the previous row."""
    return (pl.col(col) != pl.col(col).shift(1)).fill_null(default)


def format_age(age_days: pl.Expr) -> pl.Expr:
    """Format age in days as 'X years Y days' string."""
    return pl.format("{} years {} days", age_days // 365, age_days % 365)


def patient_to_text(patient_df: pl.DataFrame) -> str:
    """Convert a patient's MEDS data to free text EHR format.

    Expects columns: time_str, time_unspecified, prefix, text, age_header_text.
    Data should be sorted by time.
    """
    age_changed = changed("age_year") | changed("age_day")
    time_changed = changed("time_of_day")
    prefix_changed = changed("prefix", default=True)
    section_changed = age_changed | time_changed

    df = patient_df.with_columns(
        [
            pl.concat_str(
                [
                    # Age header (## Age X years Y days)
                    pl.when(age_changed)
                    .then(
                        pl.format(
                            "## Age {} years {} days\n",
                            pl.col("age_year"),
                            pl.col("age_day"),
                        )
                    )
                    .otherwise(pl.lit("")),
                    # Time header (### HH:MM or ### Time Unspecified)
                    pl.when(section_changed)
                    .then(
                        pl.when(pl.col("time_of_day").is_null())
                        .then(pl.lit("### Time Unspecified\n"))
                        .otherwise(pl.format("### {}\n", pl.col("time_of_day")))
                    )
                    .otherwise(pl.lit("")),
                    # Category header (#### Category)
                    pl.when(section_changed | prefix_changed)
                    .then(pl.format("#### {}\n", pl.col("prefix")))
                    .otherwise(pl.lit("")),
                    # Code content
                    pl.format("{}\n", pl.col("code")),
                ]
            ).alias("full_line")
        ]
    )

    return f"# Patient's Electronic Health Record\n\n{''.join(df['full_line'].to_list()).rstrip()}"


def main():
    project_root = Path(__file__).parent.parent.parent
    meds_dir = project_root / "data/mimic-iv-meds/data"
    output_dir = project_root / "data/outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_files = list((meds_dir / "train").glob("*.parquet"))
    if not train_files:
        raise FileNotFoundError(f"No training files found in {meds_dir / 'train'}")

    print(f"Found {len(train_files)} training files")
    print(f"Loading: {train_files[0]}")
    df = pl.read_parquet(train_files[0])

    subject_id = df["subject_id"][0]
    print(f"Filtering to subject_id: {subject_id}")

    patient_data = df.filter(pl.col("subject_id") == subject_id)
    print(df)

    print(f"Found {len(patient_data)} events for patient {subject_id}")

    ehr_markdown = patient_to_text(patient_data)
    output_file = output_dir / f"ehr_{subject_id}.md"
    output_file.write_text(ehr_markdown)
    print(f"Saved EHR to: {output_file}")


if __name__ == "__main__":
    main()
