# Foresight-R

## Install
```
uv sync
uv run pre-commit install
```
Download mimic-iv-demo with ed extension (copies ed directory into mimic directory)
```
export MIMIC_USER=your_username
export MIMIC_PASSWORD=your_password
make download-demo-data
```
Run meds pipeline
`make run-meds-extraction`
