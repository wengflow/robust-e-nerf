import easydict
import torch
from ..utils import modules


class Loss(torch.nn.Module):
    LOSS_NAMES = [
        "log_intensity_grad",
        "log_intensity_diff"
    ]

    def __init__(self, loss_weight, loss_error_fn):
        super().__init__()

        # assert loss names & weights
        assert set(self.LOSS_NAMES) <= set(loss_weight.keys())
        for loss_weight_value in loss_weight.values():
            assert isinstance(loss_weight_value, (int, float)) \
                   and loss_weight_value >= 0
        assert sum(loss_weight.values()) > 0

        # save some hyperparameters as attributes
        self.loss_weight = loss_weight
        self.error_fn = easydict.EasyDict()
        for key in self.LOSS_NAMES:
            self.error_fn[key] = {
                "l1": torch.nn.L1Loss(reduction="none"),
                "mse": torch.nn.MSELoss(reduction="none"),
                "mape": modules.MAPELoss(reduction="none")
            }[loss_error_fn[key]]

    def compute(self, batch_event, batch_grad=None, batch_diff=None):
        """
        NOTE:
            Care must be taken to handle the computation of losses with
            zero-shaped inputs.
        """
        batch_mean_loss = easydict.EasyDict({})
        batch_event.log_intensity_grad = (                                      # (batch.size)
            batch_event.log_intensity_diff
            / (batch_event.end_ts - batch_event.start_ts)
        )

        if self.loss_weight.log_intensity_grad > 0:
            batch_mean_loss.log_intensity_grad = self.log_intensity_grad(
                batch_event, batch_grad
            )
        if self.loss_weight.log_intensity_diff > 0:
            batch_mean_loss.log_intensity_diff = self.log_intensity_diff(
                batch_event, batch_diff
            )
        return batch_mean_loss

    def log_intensity_grad(self, batch_event, batch_grad):
        log_intensity_grad_err = self.error_fn.log_intensity_grad(              # (batch.size)
            input=batch_grad.log_intensity_grad,
            target=batch_event.log_intensity_grad
        )
        mean_log_intensity_grad_err = (                                         # ()
            log_intensity_grad_err[batch_grad.is_valid].mean()
        )
        return mean_log_intensity_grad_err

    def log_intensity_diff(self, batch_event, batch_diff):
        log_intensity_diff_err = self.error_fn.log_intensity_diff(              # (batch.size)
            input=batch_diff.log_intensity_diff,
            target=(batch_diff.ts_diff * batch_event.log_intensity_grad).to(
                        batch_diff.log_intensity_diff.dtype
                   )
        )
        mean_log_intensity_diff_err = (                                         # ()
            log_intensity_diff_err[batch_diff.is_valid].mean()
        )
        return mean_log_intensity_diff_err
