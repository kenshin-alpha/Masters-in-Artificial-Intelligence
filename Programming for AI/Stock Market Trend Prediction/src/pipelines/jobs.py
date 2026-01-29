from dagster import AssetSelection, define_asset_job


# Full ETL job: Execute all assets
etl_job = define_asset_job(
    name="etl_full_pipeline",
    description="Run the complete ETL pipeline: extract, transform, and load",
    selection=AssetSelection.all(),
)


# Extraction only job
extraction_job = define_asset_job(
    name="extraction_only",
    description="Run only the data extraction assets",
    selection=AssetSelection.groups("extraction"),
)


# Transformation only job  
transformation_job = define_asset_job(
    name="transformation_only",
    description="Run only the data transformation assets",
    selection=AssetSelection.groups("transformation"),
)


# Model training only job
model_training_job = define_asset_job(
    name="model_training_only",
    description="Run only the model training assets",
    selection=AssetSelection.groups("loading"),
)
