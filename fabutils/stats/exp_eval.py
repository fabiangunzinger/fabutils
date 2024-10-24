import statsmodels.api as sm
from statsmodels.stats.weightstats import ttest_ind


def welch_t_test(df, metric):
    """Return p-value of Welch's t-test."""
    control_sample = df[df["assignments"] == "control"][metric]
    variant_sample = df[df["assignments"] == "treatment"][metric]
    _, p, _ = ttest_ind(control_sample, variant_sample, usevar="unequal")
    return p


def wls(df, metric):
    """Return p-value of weighted least squares regression."""
    y = df[metric]
    x = sm.add_constant(df["is_treated"].astype(float))
    w = df["assignments_freq"]
    model = sm.WLS(endog=y, exog=x, weights=w)
    results = model.fit()
    return results.pvalues["is_treated"]

def traditional_cuped(df, metric):
    """Run traditional CUPED and return p-value."""
    
    def _cuped_adjusted_metric(df, metric, metric_pre):
        y = df[metric].values
        x = df[metric_pre].values
        valid_indices = (~np.isnan(y)) & (~np.isnan(x))
        y_valid, x_valid = y[valid_indices], x[valid_indices]
        m = np.cov(y_valid, x_valid)
        theta = m[0, 1] / m[1, 1]
        return (y - (x - np.nanmean(x)) * theta)

    # Perform experiment evaluation and return p-value
    # (Use WLS to be consistent with CausalJet)
    y = _cuped_adjusted_metric(df, metric, f"{metric}_pre")
    x = sm.add_constant(df["is_treated"].astype(float))
    w = df["assignments_freq"]
    model = sm.WLS(endog=y, exog=x, weights=w)
    results = model.fit()
    return results.pvalues["is_treated"]


if __name__ == "__main__":
    print('exp_eval.py is being run directly')
