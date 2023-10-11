import os

import pandas as pd
import s3fs


def read_csv(path, profile=None, **kwargs):
    """Reads csv files from local directory or AWS bucket."""
    if path.startswith("s3://"):
        storage_options = dict(storage_options=dict(profile=profile))
        kwargs.update(storage_options)
    return pd.read_csv(path, **kwargs)


def write_csv(df, path, profile=None, verbose=True, **kwargs):
    """Writes csv to local directory or to AWS bucket."""
    if path.startswith("s3://"):
        options = dict(storage_options=dict(profile=profile))
        df.to_csv(path, index=False, **options, **kwargs)
    else:
        df.to_csv(path, index=False, **kwargs)
    if verbose:
        print(f"{path} (of shape {df.shape}) written.")


def read_parquet(path, profile=None, **kwargs):
    """Reads parquet file from local directory or AWS bucket."""
    if path.startswith("s3://"):
        options = dict(storage_options=dict(profile=profile))
        return pd.read_parquet(path, **options, **kwargs)
    return pd.read_parquet(path, **kwargs)


def write_parquet(df, path, profile=None, index=False, verbose=True, **kwargs):
    """Writes parquet to local directory or to AWS bucket."""
    if path.startswith("s3://"):
        options = dict(storage_options=dict(profile=profile))
        df.to_parquet(path, index=index, **options, **kwargs)
    else:
        df.to_parquet(path, index=index, **kwargs)
    if verbose:
        print(f"{path} (of shape {df.shape}) written.")
