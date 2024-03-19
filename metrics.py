# anything in pytorch forecasting metrics should go in here so its all local
# from pytorch_forecasting.metrics import MAE, MAPE, MASE, RMSE, SMAPE, MultiHorizonMetric, MultiLoss, QuantileLoss
from typing import Any, List, Optional
from sklearn.base import BaseEstimator
import torch
from pytorch_forecasting.metrics.base_metrics import MultiHorizonMetric
from torchmetrics import Metric as LightningMetric


class MyMetric(LightningMetric):
    full_state_update = False
    higher_is_better = False
    is_differentiable = False

    def __init__(
        self,
        name: Optional[str] = None,
        quantiles: Optional[List[float]] = None,
        reduction: str = "mean",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.quantiles = quantiles
        self.reduction = reduction

        if name is None:
            name = (
                self.__class__.__name__
            )  # interesting, make the name of the instance the name of the class if none passed
        self.name = name

    def update(self, y_pred: torch.Tensor, y_actual: torch.Tensor):
        raise NotImplementedError()

    def compute(self) -> torch.Tensor:
        raise NotImplementedError()

    def rescale_parameters(
        self, parameters: torch.Tensor, target_scale: torch.Tensor, encoder: BaseEstimator
    ) -> torch.Tensor:
        return encoder(dict(prediction=parameters, target_scale=target_scale))

    def to_prediction(self, y_pred: torch.Tensor) -> torch.Tensor:
        """
        network preds to point predictions
        """
        if y_pred.ndim == 3:
            if self.quantiles is None:
                assert y_pred.size(-1) == 1, "Prediction should only have one extra dimension"
                y_pred = y_pred[..., 0]
            else:
                y_pred = y_pred.mean(
                    0
                )  # if it has quantiles it just gets the mean of all of them, reducing dimensions by 1
        return y_pred

    def to_quantiles(self, y_pred: torch.Tensor, quantiles: List[float] = None) -> torch.Tensor:
        """
        network preds to quantile preds
        """
        if quantiles is None:
            quantiles = self.quantiles

        if y_pred.ndim == 2:
            return y_pred.unsqueeze(-1)
        elif y_pred.ndim == 3:
            if y_pred.size(2) > 1:
                assert quantiles is not None, "quantiles are not defined"
                y_pred = torch.quantile(y_pred, torch.tensor(quantiles, device=y_pred.device), dim=2).permute(1, 2, 0)
            return y_pred  # ^ look up what this does
        else:
            raise ValueError(f"pred has 1 or more than 3 dimenstions: {y_pred.ndims}")


class MyMultiHorizonMetric(MyMetric):
    """
    class for defining metric in a multi-horizon forecast
    """

    def __init__(self, reduction: str = "mean", **kwargs: Any) -> None:
        super().__init__(reduction=reduction, **kwargs)
        # add state comes from torchmetrics.Metric
        self.add_state("losses", default=torch.tensor(0.0), dist_reduce_fx="sum" if reduction != "none" else "cat")
        self.add_state("lengths", default=torch.tensor(0), dist_reduce_fx="sum" if reduction != "none" else "mean")


class MAE(MyMultiHorizonMetric):
    """
    Mean average absolute error.

    Defined as ``(y_pred - target).abs()``
    """

    def loss(self, y_pred, target):
        loss = (self.to_prediction(y_pred) - target).abs()
        return loss
