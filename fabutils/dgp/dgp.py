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
        d = np.random.choice([0, 1], n, p=[1-p, p])

        # Pre-exepriment and experiment data
        x = rng.normal(self.mu, self.sigma, n)
        y = x + self.beta * d + rng.normal(2, 2, n)

        df = pd.DataFrame({'id': ids, 'y': y, 'x': x, 'd': d})  
        return df
