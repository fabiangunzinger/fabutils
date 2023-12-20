import os
import pandas as pd

def _query_data(
    query,
    project_id="just-data-expenab-dev",
    cache_path=None,
    convert_dates=None,
):
    """Query data from GBC or read from disk (for testing).

    Arguments:
    ----------
    query : string
        A valid BigQuery query.
    project_id : string, default "just-data-bq-users"
        Project ID to be used for query.
    cache_path: string, filepath of parquet file, default None
        Filepath to write query to (if file doesn't exist)
        or read query from (if file does exist).
    convert_dates : list of strings, default None
        List of columns to convert to datetime.
    """
    if cache_path is not None:
        if not cache_path.endswith('.parquet'):
            raise ValueError("Filepath must end with '.parquet'.")
        if os.path.isfile(cache_path):
            print("Reading data from cache...")
            result = pd.read_parquet(cache_path)
            return result

    print("Querying data from BigQuery...")
    result = pd.read_gbq(query, project_id=project_id)
    if cache_path is not None:
        print("Writing data to cache...")
        if convert_dates is not None:
            # Convert dates to datetime so that they can be written to parquet
            result[convert_dates] = result[convert_dates].apply(pd.to_datetime)
        result.to_parquet(cache_path, index=False)

    return result
