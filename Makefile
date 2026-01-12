# MIMIC-Demo Data Download
# Usage: make download-mimic USER=your_physionet_username

MIMIC_DIR := data/raw/mimic-demo
MIMIC_URL := https://physionet.org/files/mimic-iv-demo/2.2/
METADATA_URL := https://raw.githubusercontent.com/MIT-LCP/mimic-code/v2.4.0/mimic-iv/concepts/concept_map

METADATA_FILES := d_labitems_to_loinc inputevents_to_rxnorm lab_itemid_to_loinc \
                  meas_chartevents_main meas_chartevents_value numerics-summary \
                  outputevents_to_loinc proc_datetimeevents proc_itemid waveforms-summary

.PHONY: download-mimic download-metadata dirs

dirs:
	@mkdir -p $(MIMIC_DIR)
	@mkdir -p data/processed/mimic-demo-premeds
	@mkdir -p data/processed/mimic-demo-meds

download-mimic-demo: dirs
	@test -n "$(USER)" || (echo "Error: USER variable required. Use: make download-mimic USER=username" && exit 1)
	cd $(MIMIC_DIR) && \
	wget -r -N -c -np --user $(USER) --ask-password \
		--no-check-certificate --accept gz,csv --reject "index.html*" \
		$(MIMIC_URL) && \
	[ -d physionet.org/files/mimic-iv-demo/2.2 ] && \
		mv physionet.org/files/mimic-iv-demo/2.2/* . && \
		rm -rf physionet.org || true
	@echo "MIMIC-Demo data downloaded to $(MIMIC_DIR)"
