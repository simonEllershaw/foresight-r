import numpy as np
import polars as pl

from ...constants import SpecialToken as ST
from ..patterns import MatchAndRevise
from ..utils import apply_vocab_to_multitoken_codes, unify_code_names


class PrefixRowData:
    @staticmethod
    def add_prefix_rows(df: pl.DataFrame) -> pl.DataFrame:
        """
        For any rows with the same subject_id, time, and prefix (before first "//"),
        add a new row directly before these rows with just the prefix as the code.
        Other columns (numeric_value, text_value, etc.) are null in the new row.
        
        Assumes df is already sorted by subject_id and time for efficiency.
        """
        group_cols = ["subject_id", "time"]
        
        # Add row index and extract prefix
        df_indexed = df.with_row_index("_idx").with_columns(
            pl.col("code").str.split("//").list.first().alias("_prefix")
        )
        
        # Get first row index for each (subject_id, time, prefix) group
        # Use index - 0.5 to place prefix row before the group
        prefix_rows = (
            df_indexed.group_by(group_cols + ["_prefix"])
            .agg(pl.col("_idx").min().alias("_idx"))
            .with_columns(
                (pl.col("_idx") - 0.5).alias("_idx"),
                (pl.lit("HEADER//") + pl.col("_prefix")).alias("code"),
            )
            .drop("_prefix")
        )
        
        # Ensure prefix_rows has all columns from df
        for col in df.columns:
            if col not in prefix_rows.columns:
                prefix_rows = prefix_rows.with_columns(
                    pl.lit(None).cast(df.schema[col]).alias(col)
                )
        
        # Concatenate, sort by index, and drop helper columns
        return (
            pl.concat([
                prefix_rows.select(["_idx"] + df.columns), 
                df_indexed.drop("_prefix").with_columns(pl.col("_idx").cast(pl.Float64))
            ])
            .sort("_idx")
            .drop("_idx")
        )

class DeathData:
    @staticmethod
    @MatchAndRevise(prefix=["DEATH//", "HOSPITAL_DISCHARGE//"], needs_resorting=True)
    def place_death_before_dc_if_same_time(df: pl.DataFrame) -> pl.DataFrame:
        gb_cols = MatchAndRevise.sort_cols
        idx_col = MatchAndRevise.index_col
        return (
            df.sort(
                pl.col("code").replace_strict(
                    ST.DEATH, 0, default=1, return_dtype=pl.UInt8
                )
            )
            .group_by(gb_cols, maintain_order=True)
            .agg(pl.col(idx_col).last(), pl.exclude(gb_cols, idx_col))
            .explode(pl.exclude(gb_cols, idx_col))
            .sort(by=idx_col)
            .select(df.columns)
        )


class DemographicData:
    @staticmethod
    @MatchAndRevise(prefix="HOSPITAL_ADMISSION//RACE", apply_vocab=True)
    def process_race(df: pl.DataFrame) -> pl.DataFrame:
        """Changes: Remove logic for handling multiple race entries. Just treat as seperate events"""
        race_unknown = ["UNKNOWN", "UNABLE TO OBTAIN", "PATIENT DECLINED TO ANSWER"]
        race_minor = [
            "NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER",
            "AMERICAN INDIAN/ALASKA NATIVE",
            "MULTIPLE RACE/ETHNICITY",
        ]
        # every patient can have only one race assigned, so we can prioritize which one to keep
        race_priority_mapping = {
            "RACE//OTHER": 1,
            "RACE//UNKNOWN": 2,
        }  # every other will get 0
        return (
            df.with_columns(
                code=pl.when(pl.col("text_value").is_in(race_unknown))
                .then(pl.lit("UNKNOWN"))
                .when(pl.col("text_value").is_in(race_minor))
                .then(pl.lit("OTHER"))
                .when(pl.col("text_value") == "SOUTH AMERICAN")
                .then(pl.lit("HISPANIC"))
                .when(pl.col("text_value") == "PORTUGUESE")
                .then(pl.lit("WHITE"))
                .when(pl.col("text_value").str.contains_any(["/", " "]))
                .then(pl.lit(None))
                .otherwise("text_value")
            )
            .with_columns(
                code=(
                    pl.lit("HOSPITAL_ADMISSION//RACE//")
                    + pl.when(pl.col("code").is_null())
                    .then(
                        pl.col("text_value").str.slice(
                            0, pl.col("text_value").str.find("/| ")
                        )
                    )
                    .otherwise("code")
                )
            )
            .select(df.columns)
        )


class InpatientData:
    @staticmethod
    @MatchAndRevise(prefix="DIAGNOSIS_RELATED_GROUPS//", apply_vocab=True)
    def process_drg_codes(df: pl.DataFrame) -> pl.DataFrame:
        """Changes: Filter to HCFA code done in MIMIC Extract instead of here"""
        return df.with_columns(
            code=pl.lit("DIAGNOSIS_RELATED_GROUPS//")
            + pl.col.code.str.split("//").list[2].cast(int).cast(str)
        )

    @staticmethod
    @MatchAndRevise(prefix="HOSPITAL_ADMISSION//TYPE//")
    def process_hospital_admissions(df: pl.DataFrame) -> pl.DataFrame:
        """
        Changes: Shifts admission_type to index 2. Insurance event dealt as simple text event
        """
        scheduled_admissions = ["ELECTIVE", "SURGICAL SAME DAY ADMISSION"]
        return (
            df.with_columns(
                pl.col.code.str.split("//").list[0].alias("code"),
                pl.col.code.str.split("//").list[2].alias("text_value"),
            )
            .with_columns(
                code=(
                    pl.lit("HOSPITAL_ADMISSION//TYPE//")
                    + pl.when(
                        pl.col("text_value").str.ends_with("EMER.")
                        | (pl.col("text_value") == "URGENT")
                    )
                    .then(pl.lit("EMERGENCY"))
                    .when(pl.col("text_value").is_in(scheduled_admissions))
                    .then(pl.lit("SCHEDULED"))
                    .otherwise(pl.lit("OBSERVATION"))
                )
            )
        )

    @staticmethod
    @MatchAndRevise(prefix=["HOSPITAL_DISCHARGE//", "HOSPITAL_DIAGNOSIS//", "EMERGENCY_DEPARTMENT_DIAGNOSIS//", "DIAGNOSIS_RELATED_GROUPS//"])
    def process_hospital_discharges(df: pl.DataFrame) -> pl.DataFrame:
        """Currently must be run before processing diagnoses."""
        discharge_facilities = [
            "HEALTHCARE FACILITY",
            "SKILLED NURSING FACILITY",
            "REHAB",
            "CHRONIC/LONG TERM ACUTE CARE",
            "OTHER FACILITY",
        ]

        is_diagnosis = pl.col.code.str.starts_with(
            "EMERGENCY_DEPARTMENT_DIAGNOSIS//"
        ) | pl.col.code.str.starts_with("HOSPITAL_DIAGNOSIS//")

        drg_following_diag = is_diagnosis & ~pl.col.code.str.starts_with(
            "DIAGNOSIS_RELATED_GROUPS//"
        ).shift(-1, fill_value=False)
        drg_following_disch = pl.col.code.str.starts_with("HOSPITAL_DISCHARGE//")

        if "stay_id" in df.columns:
            # This means that it is MIMIC with ED extension, and diagnoses in addition come from
            # ED_OUT and in those situations DRG code should not be added
            drg_following_diag &= pl.col.stay_id.is_null() & ~(
                is_diagnosis & pl.col.stay_id.is_null()
            ).shift(-1, fill_value=False)

            drg_following_disch &= pl.col.code.str.starts_with(
                "HOSPITAL_DISCHARGE//"
            ).shift(-1, fill_value=True) | pl.col.stay_id.is_not_null().shift(
                -1, fill_value=True
            )
        else:
            drg_following_diag &= ~is_diagnosis.shift(-1, fill_value=False)
            drg_following_disch &= pl.col.code.str.starts_with(
                "HOSPITAL_DISCHARGE//"
            ).shift(-1, fill_value=True)

        drg_missing_cond = drg_following_diag | drg_following_disch

        return (
            df.with_columns(
                text_value=pl.when(pl.col.code.str.starts_with("HOSPITAL_DISCHARGE//"))
                # NOTE: Not super clear why getting oob here now...
                .then(pl.col.code.str.split("//").list.get(2, null_on_oob=True))
                .otherwise("text_value")
            )
            .with_columns(
                code=pl.when(pl.col.code.str.starts_with("HOSPITAL_DISCHARGE//"))
                .then(
                    pl.concat_list(
                        pl.lit("HOSPITAL_DISCHARGE"),
                        (
                            pl.lit("DISCHARGE_LOCATION//")
                            + pl.when(pl.col("text_value").is_in(discharge_facilities))
                            .then(pl.lit("HEALTHCARE_FACILITY"))
                            .when(pl.col("text_value").is_null())
                            .then(pl.lit("UNKNOWN"))
                            .otherwise(pl.col("text_value").replace(" ", "_"))
                        ),
                    )
                )
                .otherwise(pl.concat_list("code")),
                drg_missing=drg_missing_cond,
            )
            .with_columns(
                code=pl.when("drg_missing")
                .then(
                    pl.concat_list("code", pl.lit("DIAGNOSIS_RELATED_GROUPS//UNKNOWN"))
                )
                .otherwise("code")
            )
            .drop("drg_missing")
            .explode("code")
        )

class TextData:
    @staticmethod
    @MatchAndRevise(
        prefix = [
            "HOSPITAL_ADMISSION//INSURANCE",
            "HOSPITAL_ADMISSION//MARITAL_STATUS",
            "HOSPITAL_ADMISSION//GENDER",
            "EMERGENCY_DEPARTMENT_TRIAGE//GENDER",

        ]
    )
    def process_simple_text_events(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            code=pl.col("text_value")
        )


class MeasurementData:
    @staticmethod
    @MatchAndRevise(
        prefix=[
            "EMERGENCY_DEPARTMENT_VITAL_SIGNS//TEMPERATURE",
            "EMERGENCY_DEPARTMENT_VITAL_SIGNS//HEART_RATE",
            "EMERGENCY_DEPARTMENT_VITAL_SIGNS//RESPIRATORY_RATE",
            "EMERGENCY_DEPARTMENT_VITAL_SIGNS//O2_SATURATION",
            # Include Blood Pressure as simple measurments now seperate events
            "EMERGENCY_DEPARTMENT_VITAL_SIGNS//SYSTOLIC_BLOOD_PRESSURE",
            "EMERGENCY_DEPARTMENT_VITAL_SIGNS//DIASTOLIC_BLOOD_PRESSURE",
            "EMERGENCY_DEPARTMENT_TRIAGE//TEMPERATURE",
            "EMERGENCY_DEPARTMENT_TRIAGE//HEART_RATE",
            "EMERGENCY_DEPARTMENT_TRIAGE//RESPIRATORY_RATE",
            "EMERGENCY_DEPARTMENT_TRIAGE//O2_SATURATION",
            "EMERGENCY_DEPARTMENT_TRIAGE//SEVERITY_SCORE"
            # Include Blood Pressure as simple measurments now seperate events
            "EMERGENCY_DEPARTMENT_TRIAGE//SYSTOLIC_BLOOD_PRESSURE",
            "EMERGENCY_DEPARTMENT_TRIAGE//DIASTOLIC_BLOOD_PRESSURE",
            # Include Observational Medical Records
            "OBSERVATION_MEDICAL_RECORD//",
            # Include Age both at Hospital Admission and ED triage
            "HOSPITAL_ADMISSION//AGE",
            "EMERGENCY_DEPARTMENT_TRIAGE//AGE",
        ]
    )
    def process_simple_measurements(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
                # Extract just the code without prefix (e.g., HOSPITAL_ADMISSION//AGE -> AGE)
                code=pl.col("code").str.split("//").list.last()
            ).with_columns(
                code=pl.concat_list(
                    pl.col("code"),
                    # Add Q// prefix for quantile version
                    pl.lit("Q//") + pl.col("code"),
                )
            ).explode("code")

    @staticmethod
    @MatchAndRevise(prefix=["EMERGENCY_DEPARTMENT_VITAL_SIGNS//PAIN", "EMERGENCY_DEPARTMENT_TRIAGE//PAIN"])
    def process_pain(df: pl.DataFrame) -> pl.DataFrame:
        return (
            df.filter(pl.col.text_value.is_not_null())
            .with_columns(
                pl.col("text_value")
                .str.to_lowercase()
                .str.strip_chars(' +"')
                .str.strip_suffix("/10")
            )
            .with_columns(
                numeric_value=pl.when(pl.col.text_value.str.contains("-", literal=True))
                .then(
                    pl.col.text_value.str.split("-").list.first().str.strip_chars()
                    + pl.lit(".5")
                )
                .when(pl.col.text_value.str.contains("crit|moaning"))
                .then(pl.lit("10"))
                .when(pl.col.text_value.str.contains("lot|all over|hurts|much"))
                .then(pl.lit("8"))
                .when(
                    pl.col.text_value.is_in(
                        ["yes", "mild", "moderate", "y", "pain", "uncomfortable"]
                    )
                )
                .then(pl.lit("5"))
                .when(pl.col.text_value.str.contains("little|not bad|some|discomfort"))
                .then(pl.lit("2"))
                .when(
                    pl.col.text_value.str.contains(r"s[lep]{3,4}|sedat|n\/a|resting")
                    | pl.col.text_value.is_in(
                        ["no", "no pain", "ok", "none", "comfortable"]
                    )
                )
                .then(pl.lit("0"))
                .otherwise(pl.col("text_value").str.replace_all(r"\D", ""))
                .cast(float, strict=False)
            )
            .filter(pl.col.numeric_value.is_between(0, 10))
            .with_columns(
                code=pl.concat_list(
                    pl.lit("PAIN_SCORE"),
                    pl.lit("Q//PAIN_SCORE"),
                )
            )
            .explode("code")
        )


class DiagnosesData:
    @staticmethod
    @MatchAndRevise(prefix=["EMERGENCY_DEPARTMENT_DIAGNOSIS//ICD//", "HOSPITAL_DIAGNOSIS//ICD//"] )
    def prepare_codes_for_processing(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(pl.col.code.str.split_exact("//", 3)).with_columns(
            code=pl.lit("ICD//CM//") + pl.col.code.struct[2],
            text_value=pl.col.code.struct[3],
        )

    @staticmethod
    @MatchAndRevise(prefix="ICD//CM//9")
    def convert_icd_9_to_10(icd9_df: pl.DataFrame) -> pl.DataFrame:
        from ..mappings import get_icd_cm_9_to_10_mapping

        icd_9_to_10 = get_icd_cm_9_to_10_mapping()
        return (
            icd9_df.with_columns(
                pl.lit("ICD//CM//10").alias("code"),
                pl.col("text_value").replace_strict(icd_9_to_10, default=None),
            )
        ).drop_nulls("text_value")

    @staticmethod
    @MatchAndRevise(prefix="ICD//CM//10", needs_vocab=True)
    def process_icd10(
        icd10_df: pl.DataFrame, vocab: list[str] | None = None
    ) -> pl.DataFrame:
        from ..mappings import get_icd_cm_code_to_name_mapping

        code_to_name = get_icd_cm_code_to_name_mapping()
        temp_cols = ["part1", "part2", "part3"]
        code_prefixes = ["", "3-6//", "SFX//"]
        code_slices = [(0, 3), (3, 3), (6,)]

        df = (
            icd10_df.with_columns(
                pl.col("text_value").str.slice(*code_slice).alias(col)
                for col, code_slice in zip(temp_cols, code_slices)
            )
            .with_columns(
                pl.col(temp_cols[0]).replace_strict(code_to_name, default=None)
            )
            .with_columns(
                pl.when(pl.col(col) != "")
                .then(pl.lit(f"ICD//CM//{prefix}") + pl.col(col))
                .alias(col)
                for col, prefix in zip(temp_cols, code_prefixes)
            )
            .with_columns(unify_code_names(pl.col(temp_cols)))
        )

        if vocab is not None:
            df = apply_vocab_to_multitoken_codes(df, temp_cols, vocab)

        return (
            df.with_columns(code=pl.concat_list(temp_cols))
            .drop(temp_cols)
            .explode("code")
            .drop_nulls("code")
        )


class ProcedureData:
    @staticmethod
    @MatchAndRevise(prefix="HOSPITAL_PROCEDURE//")
    def prepare_codes_for_processing(df: pl.DataFrame) -> pl.DataFrame:
        return (
            df.with_columns(pl.col.code.str.split("//"))
            .filter(pl.col.code.list[1] == "ICD")
            .with_columns(
                code=pl.lit("ICD//PCS//") + pl.col.code.list[2],
                text_value=pl.col.code.list[3],
            )
        )

    @staticmethod
    @MatchAndRevise(prefix="ICD//PCS//9")
    def convert_icd_9_to_10(icd9_df: pl.DataFrame) -> pl.DataFrame:
        from ..mappings import get_icd_pcs_9_to_10_mapping

        icd_9_to_10 = get_icd_pcs_9_to_10_mapping()
        return (
            icd9_df.with_columns(
                pl.lit("ICD//PCS//10").alias("code"),
                pl.col("text_value").replace(icd_9_to_10, default=None),
            )
        ).drop_nulls("text_value")

    @staticmethod
    @MatchAndRevise(prefix="ICD//PCS//10", needs_vocab=True)
    def process_icd10(
        icd10_df: pl.DataFrame, vocab: list[str] | None = None
    ) -> pl.DataFrame:
        df = icd10_df.with_columns(
            pl.col("text_value").str.split_exact("", 6).alias("code")
        ).with_columns(
            code=pl.concat_list(
                pl.when(pl.col("code").struct[i] != "").then(
                    pl.lit("ICD//PCS//") + pl.col("code").struct[i]
                )
                for i in range(7)
            ).list.drop_nulls()
        )
        if vocab is not None:
            # all characters have to be in the vocab to keep the code
            df = df.filter(
                pl.col("code").list.eval(pl.element().is_in(vocab)).list.all()
            )
        return df.explode("code").drop_nulls("code")


class MedicationData:
    @staticmethod
    @MatchAndRevise(prefix=["HOSPITAL_MEDICATION//", "EMERGENCY_DEPARTMENT_MEDICATION//"], needs_vocab=True)
    def convert_to_atc(
        df: pl.DataFrame, vocab: list[str] | None = None
    ) -> pl.DataFrame:
        from ..mappings import get_atc_code_to_desc, get_mimic_drug_name_to_atc_mapping

        drug_to_atc = get_mimic_drug_name_to_atc_mapping()
        code_to_desc = get_atc_code_to_desc()
        temp_cols = ["pfx", "4", "sfx"]
        code_prefixes = ["ATC//", "ATC//4//", "ATC//SFX//"]
        code_slices = [(0, 3), (3, 1), (4,)]

        df = (
            df.with_columns(pl.col("code").str.split("//"))
            .with_columns(
                pl.when(pl.col("code").list[2] == "Administered")
                .then(None)
                .when(pl.col("code").list[1] == "START")
                .then(pl.lit("MEDICATION_START"))
                .alias("code"),
                pl.when(pl.col("code").list[2] == "Administered")
                .then(pl.col("code").list[1])
                .when(pl.col("code").list[1] == "START")
                .then(pl.col("code").list[2])
                .alias("text_value"),
            )
            .drop_nulls("text_value")
            .with_columns(
                pl.col("text_value")
                .str.strip_chars(" ")
                .str.to_lowercase()
                .replace_strict(
                    drug_to_atc, default=None, return_dtype=pl.List(pl.String)
                )
            )
            .with_columns(
                pl.concat_list(
                    "code",
                    pl.lit(None)
                    .cast(str)
                    .repeat_by(pl.col("text_value").list.len().cast(int) - 1),
                ).alias("code")
            )
            .drop_nulls("text_value")
            .explode("code", "text_value")
            .with_columns(
                pl.col("text_value").str.slice(*slice).alias(col)
                for col, slice in zip(temp_cols, code_slices)
            )
            .with_columns(
                pl.when(pl.col(col) != "")
                .then(
                    pl.lit(pfx)
                    + pl.col(col)
                    + (
                        pl.lit("//")
                        + pl.col(col).replace_strict(code_to_desc, default=None)
                        if pfx == code_prefixes[0]
                        else pl.lit("")
                    )
                )
                .alias(col)
                for col, pfx in zip(temp_cols, code_prefixes)
            )
            .with_columns(unify_code_names(pl.col(temp_cols)))
        )
        if vocab is not None:
            df = apply_vocab_to_multitoken_codes(df, temp_cols, vocab)

        return (
            df.with_columns(code=pl.concat_list("code", *temp_cols))
            .drop(temp_cols)
            .explode("code")
            .drop_nulls("code")
        )


class ICUStayData:
    @staticmethod
    @MatchAndRevise(prefix=["ICU_ADMISSION//", "ICU_DISCHARGE//"])
    def process(df: pl.DataFrame) -> pl.DataFrame:
        """Changes: Don't include SOFA scores"""
        return (
            df.with_columns(pl.col("code").str.split("//"))
            .with_columns(
                code=pl.col("code").list[0],
                text_value=pl.lit("ICU_TYPE//") + pl.col("code").list[1],
            )
            .with_columns(
                pl.when(code=ST.ICU_ADMISSION)
                .then(
                    pl.concat_list(
                        "code",
                        "text_value",
                    ).alias("code")
                )
                .otherwise(pl.concat_list("code"))
            )
            .explode("code")
        )


class LabData:
    @staticmethod
    @MatchAndRevise(prefix="LABORATORY_RESULT//", needs_counts=True, needs_vocab=True)
    def make_quantiles(
        df: pl.DataFrame,
        counts: dict[str, int] | None = None,
        vocab: list[str] | None = None,
    ) -> pl.DataFrame:
        # TODO: we've run a simple analysis and decided to keep 200 most frequent labs
        # as the cover most of all the labs in the dataset
        known_lab_names = list(counts.keys())[:200] if vocab is None else vocab
        return (
            df.filter(pl.col("code").is_in(known_lab_names))
            .with_columns(
                pl.concat_list("code", pl.lit("Q//LABORATORY_RESULT//") + pl.col("code").str.slice(5))
            )
            .explode("code")
        )
class EdData:
    @staticmethod
    @MatchAndRevise(prefix="EMERGENCY_DEPARTMENT_REGISTRATION//")
    def process_ed_registration(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            code=pl.concat_list(
                "code",
                pl.lit("EMERGENCY_DEPARTMENT_REGISTRATION//")
                + pl.when(pl.col.text_value == "HELICOPTER")
                .then(pl.lit("OTHER"))
                .otherwise("text_value"),
            )
        ).explode("code")
