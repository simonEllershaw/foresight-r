"""Convert a single patient's MEDS data to markdown format."""

import polars as pl
from pathlib import Path

END_OF_DAY = pl.time(23, 59, 59, 999999)


def patient_to_text(patient_df: pl.DataFrame) -> str:
    """Convert a patient's MEDS data to free text EHR format using pure Polars.

    Expects patient_df to already have date, time_str, category, text, and age_header_text columns computed.
    And expects patient_df to be sorted by time.
    """
    date_is_null = pl.col("date").is_null()

    # Fill null logic allows for age/time headers not to show for demographics with null dates but still show category
    df = patient_df.with_columns(
        [
            # Change detection
            (pl.col("age_header_text") != pl.col("age_header_text").shift(1))
            .fill_null(False)
            .alias("age_changed"),
            (pl.col("time_str") != pl.col("time_str").shift(1))
            .fill_null(False)
            .alias("time_changed"),
            (pl.col("category") != pl.col("category").shift(1))
            .fill_null(True)
            .alias("category_changed"),
            date_is_null.alias("date_is_null"),
        ]
    ).with_columns(
        [
            # Build complete markdown line for each row
            pl.concat_str(
                [
                    # Age header
                    pl.when(pl.col("age_changed") & ~pl.col("date_is_null"))
                    .then(pl.format("## Age {}\n", pl.col("age_header_text")))
                    .otherwise(pl.lit("")),
                    # Time header (show when time changes OR age changes)
                    pl.when(
                        (pl.col("time_changed") | pl.col("age_changed"))
                        & ~pl.col("date_is_null")
                    )
                    .then(
                        pl.when(pl.col("time_unspecified"))
                        .then(pl.lit("### Time Unspecified\n"))
                        .otherwise(pl.format("### {}\n", pl.col("time_str")))
                    )
                    .otherwise(pl.lit("")),
                    # Category header (show when category changes OR time changes OR age changes)
                    pl.when(
                        pl.col("category_changed")
                        | pl.col("time_changed")
                        | pl.col("age_changed")
                    )
                    .then(pl.format("#### {}\n", pl.col("category")))
                    .otherwise(pl.lit("")),
                    # Text content
                    pl.format("{}\n", pl.col("text")),
                ]
            ).alias("full_line")
        ]
    )

    # Join all lines together
    markdown_body = "".join(df["full_line"].to_list())
    return f"# Patient's Electronic Health Record\n\n{markdown_body.rstrip()}"


def main():
    # Set paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    MEDS_DIR = PROJECT_ROOT / "data/mimic-iv-meds/data"
    OUTPUT_DIR = PROJECT_ROOT / "data/outputs"

    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load training data
    train_files = list((MEDS_DIR / "train").glob("*.parquet"))

    if not train_files:
        print(f"No training files found in {MEDS_DIR / 'train'}")
        return

    print(f"Found {len(train_files)} training files")

    # Load first file
    sample_file = train_files[0]
    print(f"Loading: {sample_file}")
    df = pl.read_parquet(sample_file)

    # Pre-compute all transformations on full dataframe for efficiency
    # First, clean up the code column
    df = df.with_columns(
        [
            pl.col("code")
            .str.replace_all("//", " ")
            .str.replace_all("___|UNK", "?")
            .str.replace_all(r"(\s){2,}", " "),
        ]
    )

    df = df.with_columns(
        [
            # Add time column HH:MM from timestamp
            pl.col("time").dt.date().alias("date"),
            pl.col("time").dt.time().alias("time_component"),
            pl.col("time").dt.strftime("%H:%M").alias("time_str"),
            # Split code into category and text
            pl.col("code").str.split(" | ").list.first().alias("category"),
            pl.col("code").str.split(" | ").list.get(1).alias("text"),
        ]
    ).with_columns(
        [
            # Check if time is unspecified (end of day timestamp)
            (pl.col("time_component") == END_OF_DAY).alias("time_unspecified"),
        ]
    )

    # Calculate age for all patients
    # Get DOB for each subject from BIRTH | MEDS_BIRTH row
    dob_per_subject = df.filter(
        pl.col("code").str.contains("BIRTH | MEDS_BIRTH")
    ).select(["subject_id", pl.col("date").alias("dob")])

    df = df.join(dob_per_subject, on="subject_id", how="left")

    age_days = (pl.col("date") - pl.col("dob")).dt.total_days()
    df = df.with_columns(
        [
            age_days.alias("age_days_total"),
            pl.when(pl.col("date").is_not_null())
            .then(pl.format("{} years {} days", age_days // 365, age_days % 365))
            .alias("age_header_text"),
        ]
    )

    # Get first subject_id
    subject_id = df["subject_id"][0]
    print(f"Filtering to subject_id: {subject_id}")

    # Filter to one patient and sort by time
    patient_data = df.filter(pl.col("subject_id") == subject_id)

    print(f"Found {len(patient_data)} events for patient {subject_id}")
    # Convert to markdown
    ehr_markdown = patient_to_text(patient_data)

    # Save to file
    output_file = OUTPUT_DIR / f"ehr_{subject_id}.md"
    output_file.write_text(ehr_markdown)

    print(f"Saved EHR to: {output_file}")


if __name__ == "__main__":
    main()
