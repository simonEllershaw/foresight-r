import polars as pl
from pathlib import Path
from datasets import Dataset
from textwrap import dedent


# Reusing helper functions from convert_patient_to_md.py for consistency
def changed(col: str, default: bool = True) -> pl.Expr:
    """Detect when a column value changes from the previous row."""
    return (pl.col(col) != pl.col(col).shift(1)).fill_null(default)


def patient_to_text(patient_df: pl.DataFrame) -> str:
    """Convert a patient's MEDS data to free text EHR format.

    Expects columns: time_str, time_unspecified, prefix, text, age_header_text.
    Data should be sorted by time.
    """
    age_changed = changed("age_year") | changed("age_day")
    time_changed = changed("time_of_day")
    prefix_changed = changed("prefix")
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
    meds_dir = project_root / "data/mimic-iv-meds/data/train"
    readmission_path = project_root / "data/aces_outputs/readmission.parquet"
    output_dir = project_root / "data/hf_data"

    # Load readmission targets
    print(f"Loading readmission data from {readmission_path}")
    readmission_df = pl.read_parquet(readmission_path)
    print(readmission_df)

    # We need prediction times and outcomes for each subject
    # Group by subject_id to easily access per-subject targets
    # partition_by with as_dict=True returns a dictionary where keys are tuples of values
    targets_by_subject = readmission_df.partition_by("subject_id", as_dict=True)

    # Create a mapping from simple subject_id to the dataframe (handling tuple keys)
    subject_targets_map = {}
    for key, df in targets_by_subject.items():
        # Key is likely a tuple (subject_id,)
        if isinstance(key, tuple):
            subject_targets_map[key[0]] = df
        else:
            subject_targets_map[key] = df

    hf_data = []

    # Iterate over MEDS files
    meds_files = list(meds_dir.glob("*.parquet"))
    print(f"Found {len(meds_files)} MEDS files to process")

    for meds_file in meds_files:
        print(f"Processing {meds_file.name}...")
        try:
            meds_df = pl.read_parquet(meds_file)
            print(meds_df)

            # Find common subjects
            meds_subjects = set(meds_df["subject_id"].unique().to_list())
            target_subjects = set(subject_targets_map.keys())
            common_subjects = meds_subjects.intersection(target_subjects)

            if not common_subjects:
                # print(f"  No common subjects in {meds_file.name}")
                continue

            print(f"Found {len(common_subjects)} matching subjects in {meds_file.name}")

            for subject_id in common_subjects:
                subject_targets = subject_targets_map[subject_id]
                subject_events = meds_df.filter(pl.col("subject_id") == subject_id)

                # Iterate over each prediction time for the subject
                for row in subject_targets.iter_rows(named=True):
                    pred_time = row["prediction_time"]
                    target_val = row["boolean_value"]

                    # Filter events up to prediction time
                    filtered_events = subject_events.filter(pl.col("time") <= pred_time)

                    if len(filtered_events) == 0:
                        continue

                    md_text = patient_to_text(filtered_events)

                    task_text = dedent("""
                        # Task
                        Will the patient be readmitted to the hospital within 30 days of their latest hospital discharge?
                        ## Response Format
                        You must respond with the following format. The final answer is either yes or no.
                        Explanation: Brief explanation of your answer
                        Answer: Yes/No
                    """).strip()

                    # Append custom string
                    full_text = f"{md_text}\n\n{task_text}"

                    hf_data.append(
                        {
                            "subject_id": subject_id,
                            "text": full_text,
                            "label": target_val,
                        }
                    )

        except Exception as e:
            print(f"Error processing {meds_file}: {e}")
            import traceback

            traceback.print_exc()

    # Create HF Dataset
    if not hf_data:
        print("No data collected!")
        return

    print(f"Collected {len(hf_data)} examples.")
    ds = Dataset.from_list(hf_data)

    print(f"Saving dataset to {output_dir}")
    ds.save_to_disk(str(output_dir))

    # Save the first element to a file for inspection
    if len(ds) > 0:
        first_example = ds[0]
        subject_id = first_example["subject_id"]
        text_content = first_example["text"]

        output_sample_dir = project_root / "data/outputs"
        output_sample_dir.mkdir(parents=True, exist_ok=True)

        sample_file = output_sample_dir / f"readmission_{subject_id}.md"
        sample_file.write_text(text_content)
        print(f"Saved first element sample to: {sample_file}")

    print("Done!")


if __name__ == "__main__":
    main()
