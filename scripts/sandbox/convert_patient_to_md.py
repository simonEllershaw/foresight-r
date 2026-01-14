"""Convert a single patient's MEDS data to markdown format."""

import polars as pl
from pathlib import Path


def patient_to_text(patient_df: pl.DataFrame) -> str:
    """Convert a patient's MEDS data to free text EHR format using pure Polars.

    Expects patient_df to already have date, time_str, category, and text columns computed.
    And expects patient_df to be sorted by time.
    """
    # Detect changes in date, time, and category for header insertion
    df = patient_df.with_columns(
        [
            # Track when date, time or category changes from previous row
            # Fill null handles first row and rows with null dates
            (pl.col("date") != pl.col("date").shift(1))
            .fill_null(True)
            .alias("date_changed"),
            (pl.col("time_str") != pl.col("time_str").shift(1))
            .fill_null(True)
            .alias("time_changed"),
            (pl.col("category") != pl.col("category").shift(1))
            .fill_null(True)
            .alias("category_changed"),
            # Check if date is null
            pl.col("date").is_null().alias("date_is_null"),
        ]
    )

    # Build markdown lines using vectorized operations
    lines = ["# Patient's Electronic Health Record", ""]

    # Create formatted lines for each row with conditional headers
    df = df.with_columns(
        [
            # Date header (only when date changes)
            pl.when(pl.col("date_changed") & ~pl.col("date_is_null"))
            .then(pl.concat_str([pl.lit("## "), pl.col("date"), pl.lit("\n")]))
            .otherwise(pl.lit(""))
            .alias("date_header"),
            # Time header (only when time changes and date is not null)
            pl.when(pl.col("time_changed") & ~pl.col("date_is_null"))
            .then(
                # Special case for 23:59 as unspecified time
                pl.when(pl.col("time_str") == "23:59")
                .then(pl.lit("### Time Unspecified\n"))
                .otherwise(
                    pl.concat_str([pl.lit("### "), pl.col("time_str"), pl.lit("\n")])
                )
            )
            .otherwise(pl.lit(""))
            .alias("time_header"),
            # Category header (only when category changes)
            pl.when(pl.col("category_changed"))
            .then(pl.concat_str([pl.lit("#### "), pl.col("category"), pl.lit("\n")]))
            .otherwise(pl.lit(""))
            .alias("category_header"),
            # The text content
            pl.concat_str([pl.col("text"), pl.lit("\n")]).alias("text_line"),
        ]
    )

    # Concatenate all parts for each row
    df = df.with_columns(
        [
            pl.concat_str(
                [
                    pl.col("date_header"),
                    pl.col("time_header"),
                    pl.col("category_header"),
                    pl.col("text_line"),
                ]
            ).alias("full_line")
        ]
    )

    # Join all lines together
    markdown_body = "".join(df["full_line"].to_list())

    return "\n".join(lines) + "\n" + markdown_body.rstrip("\n")


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
            pl.col("time").dt.strftime("%H:%M").alias("time_str"),
            # Split code into category and text
            pl.col("code").str.split(" | ").list.first().alias("category"),
            pl.col("code").str.split(" | ").list.get(1).alias("text"),
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
