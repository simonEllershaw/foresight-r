"""Convert a single patient's MEDS data to markdown format."""

import polars as pl
from pathlib import Path


def patient_to_text_v2(patient_df: pl.DataFrame) -> str:
    """Convert a patient's MEDS data to free text EHR format with date/time/category headers.

    Expects patient_df to already have date, time_str, category, and text columns computed.
    And expects patient_df to be sorted by time.
    """
    lines = ["# Patient's Electronic Health Record", ""]

    # Sort by time (nulls first)
    # df = patient_df.sort("time", nulls_last=False)

    prev_date = "INITIAL"  # Sentinel value to trigger first header
    prev_time = None
    prev_category = None

    for row in patient_df.iter_rows(named=True):
        category = row["category"]
        text = row["text"]
        curr_date = row["date"]
        curr_time = row["time_str"]

        # Add date header if date changed
        if curr_date != prev_date:
            if curr_date is None:
                lines.append("## Null")
            else:
                lines.append(f"## {curr_date}")
            prev_date = curr_date
            prev_time = None  # Reset time when date changes
            prev_category = None  # Reset category when date changes

        # Add time header if time changed (skip for null dates)
        if curr_date is not None and curr_time != prev_time:
            lines.append(f"### {curr_time}")
            prev_time = curr_time
            prev_category = None  # Reset category when time changes

        # Add category header if category changed
        if category != prev_category:
            lines.append(f"#### {category}")
            prev_category = category

        # Add the text
        lines.append(text)

    return "\n".join(lines)


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
            .str.replace_all("___", "")
            .str.replace_all("UNK", "")
            .str.replace_all(r"(\s){2,}", " "),
        ]
    )

    # Then split the cleaned code and add time columns
    df = df.with_columns(
        [
            pl.col("time").dt.date().alias("date"),
            pl.col("time").dt.strftime("%H:%M").alias("time_str"),
            pl.col("code")
            .str.split(" | ")
            .list.first()
            .fill_null("")
            .alias("category"),
            pl.col("code")
            .str.split(" | ")
            .list.get(1, null_on_oob=True)
            .fill_null(pl.col("code"))
            .alias("text"),
        ]
    )

    # Get first subject_id
    subject_id = df["subject_id"][0]
    print(f"Filtering to subject_id: {subject_id}")

    # Filter to one patient and sort by time
    patient_data = df.filter(pl.col("subject_id") == subject_id).sort("time")

    print(f"Found {len(patient_data)} events for patient {subject_id}")

    # Convert to markdown
    ehr_markdown = patient_to_text_v2(patient_data)

    # Save to file
    output_file = OUTPUT_DIR / f"ehr_{subject_id}.md"
    output_file.write_text(ehr_markdown)

    print(f"Saved EHR to: {output_file}")


if __name__ == "__main__":
    main()
