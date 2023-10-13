import torch
import torch.nn as nn


class NegLogNegBinLoss(nn.Module):
    """Negative Log Negative Binomial Loss."""

    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true, var):
        """Forward pass of the loss function.

        Parameters
        ----------
         y_pred
             Matrix of the predicted gene expression values for each sample and gene.
        y_true
             Matrix of the true gene expression values for each sample and gene.
         var
             Predicted variance for each gene.


        Returns
        -------
         Total loss.
        """
        eps = 1e-8  # Small constant to avoid taking log(0)
        mu = torch.log1p(torch.exp(y_pred))  # Log1p transformation for stability
        theta = torch.log1p(torch.exp(var))  # Log1p transformation for stability

        # Compute the Negative Binomial Loss.
        t1 = torch.lgamma(y_true + 1 / theta + eps) - torch.lgamma(y_true + 1 + eps) - torch.lgamma(1 / theta + eps)
        t2 = -(1 / theta) * torch.log1p(mu * theta + eps)
        t3 = y_true * torch.log(mu * theta / (1 + mu * theta + eps))

        loss = -(t1 + t2 + t3)
        loss = torch.clamp(loss, min=-300, max=300)
        nll = loss.mean()

        return nll
