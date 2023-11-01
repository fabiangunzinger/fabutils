"""
Title:  dgp.py
Author: Fabian Gunzinger
Date:   11 Oct 2023

This module contains data generating processes to easily generate data in sample projects. I borrow liberally from a similar module by Matteo Courthoud, which can be found here: https://github.com/matteocourthoud/Blog-Posts/blob/main/notebooks/src/dgp.py.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class dgp_cuped:
    """Data generating process for CUPED examples"""

    mu: float = 20
    sigma: float = 4
    beta: float = 2

    def make_data(self, n=1000, p=0.5, seed=2312):
        rng = np.random.default_rng(seed)

        # Ids and treatment assignment vector
        ids = np.arange(n)
        d = np.random.choice([0, 1], n, p=[1 - p, p])

        # Pre-exepriment and experiment data
        x = rng.normal(self.mu, self.sigma, n)
        y = x + self.beta * d + rng.normal(2, 2, n)

        df = pd.DataFrame({"id": ids, "y": y, "x": x, "d": d})
        return df


# todo turn into class for simple cross section and panel data creation
def make_sample_data(
    units=100_000,
    periods=50,
    max_unit_effect=0.1,
    metric_name="y",
    add_assignment=False,
    cross_section=False,
    default_panel=False,
    default_cross_section=False,
):
    """Create dummy data for testing.

    Arguments:
    ----------
    units : int
        Number of units to simulate.
    periods : int
        Number of periods to simulate.
    max_unit_effect : float
        Size of unit effect to simulate. This is useful, for instance, to ensure that
        CUPED has an effect.
    metric_name : str
        Name of metric column.
    add_assignment : bool
        Whether to add assignment columns.
    cross_section : bool
        Whether to return cross-sectional data.

    Returns:
    --------
    df : pd.DataFrame
        Simulated data.
    """
    n = units * periods
    date_range = list(pd.date_range("2023-01-01", periods=periods, freq="D"))
    unit_effect = np.random.uniform(0, max_unit_effect, size=n)

    df = pd.DataFrame(
        {
            "id": sorted([f"unit_{id}" for id in range(units)] * periods),
            "timeframe": date_range * units,
            metric_name: np.random.uniform(0, 1, size=n) + unit_effect,
        }
    )

    if add_assignment:
        labs = {False: "control", True: "treatment"}
        df["is_treated"] = np.random.choice([0, 1], size=n).astype(bool)
        df["assignments"] = df["is_treated"].map(labs)
        df["assignments_freq"] = 1

    if cross_section:
        df = df.drop(columns="timeframe").groupby("id").mean().reset_index()

    return df
