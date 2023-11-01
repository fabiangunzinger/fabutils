def select_sample(df, num_units=None, num_periods=None, seed=2312):
    """Select sample from dataframe.

    Arguments:
    ----------
    df : pd.DataFrame
        Dataframe to sample from.
    num_units : int, default None
        Number of units to sample. If None, all units are included.
    num_periods : int, default None
        Number of periods to sample. If None, all periods are included.
    seed : int, default 2312
        Random seed to use for sampling.

    Returns:
    --------
    df : pd.DataFrame
        Sampled dataframe.
    """
    if num_units is None:
        units = df["id"].unique()
    else:
        rng = np.random.default_rng(seed)
        units = rng.choice(df["id"].unique(), size=num_units, replace=False)

    if num_periods is None:
        periods = sorted(df["timeframe"].unique())
    else:
        periods = periods[:num_periods]

    return df[df["id"].isin(units) & df["timeframe"].isin(periods)]
