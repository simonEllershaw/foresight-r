# MIMIC-IV Data Processing Pipeline
# ===================================
#
# Download: MIMIC_USER=user MIMIC_PASSWORD=pass make download-demo-data
# Process:  make run-meds-extraction

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

MIMIC_IV_DEMO_URL    := https://physionet.org/files/mimic-iv-demo/2.2/
MIMIC_ED_DEMO_URL    := https://physionet.org/files/mimic-iv-ed-demo/2.2/

MIMICIV_RAW_DIR      := data/mimic-iv
MIMICIV_ED_RAW_DIR   := data/mimic-iv-ed
MIMICIV_PRE_MEDS_DIR := data/mimic-iv-premeds
MIMICIV_MEDS_DIR     := data/mimic-iv-meds

MIMIC_MEDS_SCRIPT_DIR := scripts/meds/mimic
N_WORKERS             := 1

.PHONY: download-mimic-demo download-mimic-ed-demo download-demo-data run-meds-extraction

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

# ------------------------------------------------------------------------------
# MEDS Processing Targets
# ------------------------------------------------------------------------------

run-meds-extraction:
	@rm -rf $(MIMICIV_PRE_MEDS_DIR) $(MIMICIV_MEDS_DIR)
	@mkdir -p $(MIMICIV_PRE_MEDS_DIR) $(MIMICIV_MEDS_DIR)
	@echo "Running pre-MEDS processing..."
	uv run python "$(CURDIR)/$(MIMIC_MEDS_SCRIPT_DIR)/pre_MEDS.py" \
		input_dir="$(CURDIR)/$(MIMICIV_RAW_DIR)" \
		cohort_dir="$(CURDIR)/$(MIMICIV_PRE_MEDS_DIR)"
	@echo "Running MEDS transform runner..."
	MIMICIV_PRE_MEDS_DIR="$(CURDIR)/$(MIMICIV_PRE_MEDS_DIR)" \
	MIMICIV_MEDS_COHORT_DIR="$(CURDIR)/$(MIMICIV_MEDS_DIR)" \
	EVENT_CONVERSION_CONFIG_FP="$(CURDIR)/$(MIMIC_MEDS_SCRIPT_DIR)/configs/event_configs-ed-nl.yaml" \
	N_WORKERS=$(N_WORKERS) \
	uv run MEDS_transform-runner \
		pipeline_config_fp="$(CURDIR)/$(MIMIC_MEDS_SCRIPT_DIR)/configs/extract_MIMIC.yaml" \
		stage_runner_fp="$(CURDIR)/scripts/meds/local_parallelism_runner.yaml"
