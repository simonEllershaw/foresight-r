# MIMIC-IV Data Processing Pipeline
# ===================================
#
# Download: MIMIC_USER=user MIMIC_PASSWORD=pass make download-demo-data
# Process:  make meds

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

MIMIC_IV_DEMO_URL    := https://physionet.org/files/mimic-iv-demo/2.2/
MIMIC_ED_DEMO_URL    := https://physionet.org/files/mimic-iv-ed-demo/2.2/

MIMICIV_RAW_DIR      := data/mimic-iv
MIMICIV_ED_RAW_DIR   := data/mimic-iv-ed
MIMICIV_PRE_MEDS_DIR := data/mimic-iv-premeds
MIMICIV_MEDS_DIR     := data/mimic-iv-meds

MIMIC_MEDS_DIR_SRC := foresight_r/meds/mimic
N_WORKERS          := 1

ACES_OUTPUT_DIR := data/mimic-iv-aces-labels
ACES_CONFIG_DIR := config/aces/mimic4ed-benchmark
ACES_COHORTS := $(notdir $(basename $(wildcard $(ACES_CONFIG_DIR)/*.yaml)))

.PHONY: download-mimic-demo download-mimic-ed-demo download-demo-data meds sample_markdown_file test aces_outputs ethos-tokenization nl-dataset

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------

define check_credentials
	@test -n "$(MIMIC_USER)" || (echo "Error: MIMIC_USER required" && exit 1)
	@test -n "$(MIMIC_PASSWORD)" || (echo "Error: MIMIC_PASSWORD required" && exit 1)
endef

# ------------------------------------------------------------------------------
# Data Download Targets
# ------------------------------------------------------------------------------

download-mimic-demo:
	$(call check_credentials)
	@mkdir -p $(MIMICIV_RAW_DIR)
	@cd $(MIMICIV_RAW_DIR) && \
		wget -r -N -c -np --user $(MIMIC_USER) --password $(MIMIC_PASSWORD) \
			--no-check-certificate --accept gz,csv --reject "index.html*" \
			$(MIMIC_IV_DEMO_URL) && \
		[ -d physionet.org ] && mv physionet.org/files/mimic-iv-demo/2.2/* . && \
		rm -rf physionet.org || true
	@echo "Downloaded MIMIC-IV to $(MIMICIV_RAW_DIR)"


download-mimic-ed-demo:
	$(call check_credentials)
	@mkdir -p $(MIMICIV_ED_RAW_DIR)
	@cd $(MIMICIV_ED_RAW_DIR) && \
		wget -r -N -c -np --user $(MIMIC_USER) --password $(MIMIC_PASSWORD) \
			--no-check-certificate --accept gz,csv --reject "index.html*" \
			$(MIMIC_ED_DEMO_URL) && \
		[ -d physionet.org ] && mv physionet.org/files/mimic-iv-ed-demo/2.2/* . && \
		rm -rf physionet.org || true
	@echo "Downloaded MIMIC-IV-ED to $(MIMICIV_ED_RAW_DIR)"

download-demo-data: download-mimic-demo download-mimic-ed-demo
	@mv $(MIMICIV_ED_RAW_DIR)/ed $(MIMICIV_RAW_DIR)/
	@rm -rf $(MIMICIV_ED_RAW_DIR)
	@echo "Complete: MIMIC-IV-ED moved to $(MIMICIV_RAW_DIR)/ed"
	gunzip -r -v $(MIMICIV_RAW_DIR)
	@echo "Unzipped MIMIC-IV to $(MIMICIV_RAW_DIR)"

# ------------------------------------------------------------------------------
# MEDS Processing Targets
# ------------------------------------------------------------------------------

meds:
	@echo "Running pre-MEDS processing..."
	@rm -rf $(MIMICIV_PRE_MEDS_DIR)
	uv run python "$(CURDIR)/$(MIMIC_MEDS_DIR_SRC)/pre_MEDS.py" \
		input_dir="$(CURDIR)/$(MIMICIV_RAW_DIR)" \
		cohort_dir="$(CURDIR)/$(MIMICIV_PRE_MEDS_DIR)"

	@echo "Running MEDS extraction..."
	@rm -rf $(MIMICIV_MEDS_DIR)
	INPUT_DIR="$(CURDIR)/$(MIMICIV_PRE_MEDS_DIR)" \
	COHORT_DIR="$(CURDIR)/$(MIMICIV_MEDS_DIR)" \
	EVENT_CONVERSION_CONFIG_FP="$(CURDIR)/config/meds/mimic/schemas/event_configs-ed-foresight.yaml" \
	N_WORKERS=$(N_WORKERS) \
	uv run MEDS_transform-runner \
		pipeline_config_fp="$(CURDIR)/config/meds/mimic/transforms/extract_MIMIC.yaml" \
		stage_runner_fp="$(CURDIR)/config/meds/mimic/transforms/local_parallelism_runner.yaml"

sample_markdown_file:
	uv run python scripts/sandbox/convert_patient_to_md.py

test:
	uv run pytest

aces-labels:
	@rm -rf $(ACES_OUTPUT_DIR)
	for cohort in $(ACES_COHORTS); do \
		uv run aces-cli -m data=sharded \
			data.root="data/mimic-iv-meds/data" \
			"data.shard=$$(uv run expand_shards train/1 held_out/1 tuning/1)" \
			data.standard="meds" \
			config_path="$(ACES_CONFIG_DIR)/$$cohort.yaml" \
			cohort_dir=$(ACES_OUTPUT_DIR) \
			cohort_name="$$cohort"; \
	done

	@echo "Combining critical_outcome_icu_12h and critical_outcome_hospital_mortality" \
		"into critical_outcome using OR logic."
	uv run combine-critical-outcome \
		$(ACES_OUTPUT_DIR)/critical_outcome_icu_12h \
		$(ACES_OUTPUT_DIR)/critical_outcome_hospital_mortality \
		$(ACES_OUTPUT_DIR)/critical_outcome

	@echo "All ACES task definitions share the same trigger point" \
		"(ED arrival). So check that for each shard, the subject_ids" \
		"and prediction_times are identical across tasks."
	uv run validate-aces-prediction-times $(ACES_OUTPUT_DIR)

ethos-tokenization:
	PYTHONPATH=external/ethos-ares/src \
	uv run python -m ethos.tokenize.run_tokenization \
		input_dir=data/mimic-iv-meds/data/train \
		output_dir=data/mimic-iv-ethos-tokenized \
		out_fn=train \
		overwrite=True


# ------------------------------------------------------------------------------
# Natural Language Dataset Target
# ------------------------------------------------------------------------------

NL_DATASET_DIR := data/mimic-iv-nl-dataset

nl-dataset:
	@rm -rf $(NL_DATASET_DIR)
	@echo "Creating natural language dataset from MEDS with ACES labels..."
	MEDS_DIR="$(CURDIR)/$(MIMICIV_MEDS_DIR)/data" \
	LABELS_DIR="$(CURDIR)/$(ACES_OUTPUT_DIR)" \
	OUTPUT_DIR="$(CURDIR)/$(NL_DATASET_DIR)" \
	uv run create-nl-dataset

download-qwen-8B:
	uvx hf auth login
	uvx hf download Qwen/Qwen3-0.6B --local-dir ./models/Qwen3-0.6B

inference:
	uv run run-llm-inference \
		model_path=./models/Qwen3-0.6B \
		dataset_dir=data/mimic-iv-nl-dataset \
		split=train \
		output_dir=outputs/inference
