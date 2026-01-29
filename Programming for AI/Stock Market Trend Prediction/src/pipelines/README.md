# Project CA2 - ETL: Dagster Pipeline

## Overview

This project implements a complete ETL pipeline using Dagster for stock market prediction data.

## Pipeline Structure

```
src/pipelines/
├── __init__.py              # Dagster Definitions
├── resources.py             # PostgreSQL & Storage resources
├── jobs.py                  # Job definitions
├── schedules.py            # Daily schedule (2 AM)
└── assets/
    ├── extraction.py       # Load CSV/JSON data
    ├── transformation.py   # Clean &engineer features
    └── loading.py          # Train ML model
```

## Asset Lineage

```
raw_stock_csv_data  ┐
                     ├─→ combined_raw_data → cleaned_stock_data
raw_stock_json_data ┘
                    
cleaned_stock_data → engineered_features → training_dataset → trained_model → model_metrics
```

## Running the Pipeline

### 1. Start Dagster UI

```bash
cd "c:/Users/Kenshin/Documents/Programming for AI/Project CA2 - ETL"
dagster dev
```

Access UI at: http://localhost:3000

### 2. Run Full Pipeline

**Via UI:**
- Navigate to "Jobs" → "etl_full_pipeline"
- Click "Launchpad" → "Launch Run"

**Via CLI:**
```bash
dagster job execute -f src/pipelines/__init__.py -j etl_full_pipeline
```

### 3. Run Modular Jobs

```bash
# Extract only
dagster job execute -f src/pipelines/__init__.py -j extraction_only

# Transform only
dagster job execute -f src/pipelines/__init__.py -j transformation_only

# Train model only
dagster job execute -f src/pipelines/__init__.py -j model_training_only
```

## Data Quality Checks

The pipeline includes 4 asset checks:

1. **Combined Data Quality** - Validates raw data integrity
2. **Feature Quality** - Checks engineered features
3. **Model Performance** - Ensures model meets thresholds (accuracy ≥ 70%)

## Schedule

- **Daily ETL**: Runs every day at 2 AM
- Mode: Full refresh (not incremental)

## Configuration

Edit `src/pipelines/resources.py` to change:
- PostgreSQL credentials
- File paths
- Model parameters

## Outputs

- **Processed Data**: `data/processed/training_data.csv`
- **Model**: `models/random_forest_model.pkl`
- **Metrics**: `models/model_metrics.json`
