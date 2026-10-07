# Data exports

Business workspaces can schedule daily exports of raw events to Amazon S3 or Google Cloud Storage.
Exports are written as Parquet files partitioned by event date.
A failed export is retried three times over six hours before the workspace admin is notified.
Manual CSV exports from the UI are capped at one million rows.
