"""
Dagster pipeline definitions for Project CA2 - ETL.

This module defines the assets, jobs, schedules, and sensors for the ETL pipeline.
"""

from dagster import Definitions, load_assets_from_modules

from . import assets
from .jobs import etl_job
from .schedules import daily_etl_schedule
from .resources import postgres_connection, mongodb_connection, data_storage

# Load all assets from the assets module
all_assets = load_assets_from_modules([assets])

# Define the Dagster project
defs = Definitions(
    assets=all_assets,
    jobs=[etl_job],
    schedules=[daily_etl_schedule],
    resources={
        "postgres": postgres_connection,
        "mongodb": mongodb_connection,
        "storage": data_storage,
    },
)
