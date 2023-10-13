import torch


def _r_squared_linreg(y_true, y_pred):
    """Compute the R-squared value via linear regression.

    Parameters
    ----------
    y_true
        Matrix of the true gene expression values for each sample and gene.
    y_pred
        Matrix of the predicted gene expression values for each sample and gene.

    Returns
    -------
    R-squared value.
    """
    x = y_true
    y = y_pred
    # means
    xmean = torch.mean(x)
    ymean = torch.mean(y)
    # covariance
    ssxm = torch.mean(torch.square(x - xmean))
    ssym = torch.mean(torch.square(y - ymean))
    ssxym = torch.mean((x - xmean) * (y - ymean))
    xmym = ssxm * ssym

    # Helper functions for tf.cond
    def f0():
        return torch.zeros(size=(), dtype=torch.float32)

    def f1():
        return torch.ones(size=(), dtype=torch.float32)

    def r():
        return ssxym / torch.sqrt(xmym)  # formula for r

    def r2():
        return r**2  # formula for r_squared

    # R-value
    # If the denominator was going to be 0, r = 0.0
    if torch.is_nonzero(xmym):
        r = r()
    else:
        r = f0()
    # Test for numerical error propagation (make sure -1 < r < 1)
    if torch.gt(torch.abs(r), torch.ones(size=(), dtype=torch.float32)):
        r_squared = f1()
    else:
        r_squared = r2()
    return r_squared
