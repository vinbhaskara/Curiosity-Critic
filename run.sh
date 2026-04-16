#!/usr/bin/env bash
set -euo pipefail

python curiosity_experiment.py batch \
    --output-dir results/ \
    --total-steps 35000

python plot.py \
    --input-dir results/ \
    --output-dir results/

python analyze_visits.py \
    --input-dir results/ \
    --output-dir results/

python animate.py \
    --results-dir results/ \
    --fps 1200 \
    --smooth-interpol 8
