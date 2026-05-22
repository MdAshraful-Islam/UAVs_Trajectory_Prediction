# Physics-guided residual learning for UAV trajectory prediction

This repository contains the code, final figures, and result tables for a journal submission on multi-step UAV trajectory prediction. The project evaluates a hybrid physics-guided residual-correction model against three baseline variants: a standalone physics-based rollout, an LSTM model, and an MLP fusion model.

The main idea is to use the physics-based prediction as a structured trajectory prior and then learn a data-driven residual correction. This allows the final model to retain useful physical guidance while compensating for drift, phase-transition effects, and unmodeled disturbances in real UAV telemetry.

## What This Repository Contains

This is a lightweight sharing package for journal review. It includes the essential scripts, final paper figures, and reported metric files. Large raw datasets, trained model checkpoints, and large intermediate prediction arrays are intentionally excluded.

```text
.
|-- src/
|   |-- lstm_unified.py
|   |-- Physics_model.py
|   |-- MLP.py
|   `-- PIML.py
|-- figures/
|-- results
|   |-- metrics/
|   |-- run_info_piml_v12.json
|   `-- piml_summary.csv
|-- data/
|   `-- README.md
|-- docs/
|   |-- REPRODUCIBILITY.md
|   `-- latex_tables.md
|-- requirements.txt
|-- CITATION.cff
`-- .gitignore
```

## Model Variants

- `Physics_model.py`: analytical physics-based trajectory rollout.
- `lstm_unified.py`: LSTM-only data-driven sequence prediction model.
- `MLP.py`: learnable MLP fusion baseline combining physics and LSTM predictions.
- `PIML.py`: proposed physics-informed residual-correction model.

## Key Outputs

The `figures/` folder contains the final paper-ready visualizations, including:

- horizon-error comparison over the forecast horizon
- phase-wise real-vs-predicted trajectory plots
- accuracy and physical-feasibility trade-off plot
- residual-correction magnitude plot
- physical-feasibility effect plot
- rolling forecast timeline diagram

The `results/metrics/` folder contains the CSV and JSON files used to report:

- overall RMSE, MAE, ADE, and FDE
- phase-wise prediction performance
- physical-feasibility metrics
- residual correction magnitudes
- accuracy-feasibility trade-off data

## Installation

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```


## Data Availability

The raw UAV telemetry data are not included in this lightweight repository package. To rerun the full training pipeline, place the dataset folders under `data/` as described in `data/README.md`.

The included figures and metric files are sufficient to inspect the reported journal results without rerunning the full pipeline.

## Reproducibility

See `docs/REPRODUCIBILITY.md` for the expected execution order and notes on the final reported result source.

## Suggested GitHub Description

Physics-guided residual learning framework for multi-step UAV trajectory prediction, including LSTM, physics-based, MLP fusion, and PIML/PGRL model variants with final journal figures and evaluation metrics.

