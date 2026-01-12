# MIMIC-Demo Data Download
# Usage: MIMIC_USER=username MIMIC_PASSWORD=password make download-demo-data

MIMIC_IV_DEMO_URL := https://physionet.org/files/mimic-iv-demo/2.2/
MIMIC_ED__DEMO_URL := https://physionet.org/files/mimic-iv-ed-demo/2.2/

MIMICIV_RAW_DIR := data/mimic-iv
MIMICIV_ED_RAW_DIR:= data/mimic-iv-ed
MIMICIV_PRE_MEDS_DIR= data/mimic-iv-premeds
MIMICIV_MEDS_DIR=data/mimic-iv-meds

MIMIC_MEDS_SCRIPT_DIR := scripts/meds/mimic
N_WORKERS := 1

.PHONY: download-mimic-demo download-mimic-ed-demo download-demo-data run-meds-extraction

download-mimic-demo:
	@test -n "$(MIMIC_USER)" || (echo "Error: MIMIC_USER required. Usage: MIMIC_USER=user MIMIC_PASSWORD=pass make download-demo-data" && exit 1)
	@test -n "$(MIMIC_PASSWORD)" || (echo "Error: MIMIC_PASSWORD required. Usage: MIMIC_USER=user MIMIC_PASSWORD=pass make download-demo-data" && exit 1)
	@mkdir -p $(MIMICIV_RAW_DIR)
	@cd $(MIMICIV_RAW_DIR) && \
		wget -r -N -c -np --user $(MIMIC_USER) --password $(MIMIC_PASSWORD) --no-check-certificate \
			--accept gz,csv --reject "index.html*" $(MIMIC_IV_DEMO_URL) && \
		[ -d physionet.org ] && mv physionet.org/files/mimic-iv-demo/2.2/* . && rm -rf physionet.org || true
	@echo "Downloaded MIMIC-IV to $(MIMICIV_RAW_DIR)"

download-mimic-ed-demo:
	@mkdir -p $(MIMICIV_ED_RAW_DIR)
	@test -n "$(MIMIC_USER)" || (echo "Error: MIMIC_USER required. Usage: MIMIC_USER=user MIMIC_PASSWORD=pass make download-demo-data" && exit 1)
	@test -n "$(MIMIC_PASSWORD)" || (echo "Error: MIMIC_PASSWORD required. Usage: MIMIC_USER=user MIMIC_PASSWORD=pass make download-demo-data" && exit 1)
	@cd $(MIMICIV_ED_RAW_DIR) && \
		wget -r -N -c -np --user $(MIMIC_USER) --password $(MIMIC_PASSWORD) --no-check-certificate \
			--accept gz,csv --reject "index.html*" $(MIMIC_ED__DEMO_URL) && \
		[ -d physionet.org ] && mv physionet.org/files/mimic-iv-ed-demo/2.2/* . && rm -rf physionet.org || true
	@echo "Downloaded MIMIC-IV-ED into $(MIMICIV_ED_RAW_DIR)"

download-demo-data: download-mimic-demo download-mimic-ed-demo
	@mv $(MIMICIV_ED_RAW_DIR)/ed $(MIMICIV_RAW_DIR)/
	@rm -rf $(MIMICIV_ED_RAW_DIR)
	@echo "Complete: MIMIC-IV-ED moved to $(MIMICIV_RAW_DIR)/ed"

run-meds-extraction:
	
	@mkdir -p $(MIMICIV_PRE_MEDS_DIR) $(MIMICIV_MEDS_DIR)
	@echo "Running pre-MEDS processing..."
	uv run python "$(CURDIR)/$(MIMIC_MEDS_SCRIPT_DIR)/pre_MEDS.py" input_dir="$(CURDIR)/$(MIMICIV_RAW_DIR)" cohort_dir="$(CURDIR)/$(MIMICIV_PRE_MEDS_DIR)"
	
	@echo "Running MEDS transform runner..."
	MIMICIV_PRE_MEDS_DIR="$(CURDIR)/$(MIMICIV_PRE_MEDS_DIR)" \
	MIMICIV_MEDS_COHORT_DIR="$(CURDIR)/$(MIMICIV_MEDS_DIR)" \
	EVENT_CONVERSION_CONFIG_FP="$(CURDIR)/$(MIMIC_MEDS_SCRIPT_DIR)/configs/event_configs-ed.yaml" \
	uv run MEDS_transform-runner pipeline_config_fp="$(CURDIR)/$(MIMIC_MEDS_SCRIPT_DIR)/configs/extract_MIMIC.yaml" stage_runner_fp="$(CURDIR)/scripts/meds/local_parallelism_runner.yaml"