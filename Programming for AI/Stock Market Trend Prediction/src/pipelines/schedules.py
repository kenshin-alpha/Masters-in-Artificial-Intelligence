from dagster import ScheduleDefinition
from .jobs import etl_job


# Daily ETL schedule - runs every day at 2 AM
daily_etl_schedule = ScheduleDefinition(
    name="daily_etl",
    job=etl_job,
    cron_schedule="0 2 * * *",  # 2 AM every day
    description="Run the full ETL pipeline daily at 2 AM"
)
