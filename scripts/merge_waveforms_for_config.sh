#!/usr/bin/env bash
set -euo pipefail

# Merge public FDSN and authenticated Hi-net waveform-derived CSVs into the
# combined paths used by configs/japan_all_usgs_modern_m2.yml.
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source scripts/workspace_env.sh

mkdir -p data/raw/waveforms/combined

"${CRUST_LITE_VENV}/bin/python" scripts/merge_waveform_csv.py \
  --output data/raw/waveforms/combined/japan_all_2000_2026_m55_spectra.csv \
  --key-columns event_id,station_id,channel,frequency_hz \
  data/raw/waveforms/fdsn/japan_all_2000_2026_m6_spectra.csv \
  data/raw/waveforms/hinet/japan_all_2000_2026_m55_spectra.csv

"${CRUST_LITE_VENV}/bin/python" scripts/merge_waveform_csv.py \
  --output data/raw/waveforms/combined/japan_all_2000_2026_m55_waveform_features.csv \
  --key-columns event_id,station_id,channel \
  data/raw/waveforms/fdsn/japan_all_2000_2026_m6_waveform_features.csv \
  data/raw/waveforms/hinet/japan_all_2000_2026_m55_waveform_features.csv
