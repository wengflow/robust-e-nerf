import easydict
import torch
import torchmetrics
import lpips
from ..utils import modules


class Metric(torch.nn.Module):
    METRIC_NAMES = [
        "l1",
        "psnr",
        "ssim",
        "lpips"
    ]

    def __init__(self, metric_lpips_net):
        super().__init__()
        self.lpips = lpips.LPIPS(net=metric_lpips_net)
        modules.freeze(self.lpips)

    def init_batch_metric(self):
        batch_metric = easydict.EasyDict({
            metric_name: [ ]
            for metric_name in self.METRIC_NAMES
        })
        return batch_metric

    def compute(
        self,
        pred_img,                                                               # ([[batch_size,] 1/3,] img_height, img_width)
        target_img,                                                             # ([[batch_size,] 1/3,] img_height, img_width)
        min_target_val,
        max_target_val
    ):
        # verify inputs
        assert pred_img.shape == target_img.shape
        assert 2 <= target_img.dim() <= 4
        if target_img.dim() > 2: 
            assert target_img.shape[-3] in (1, 3)
        assert 0 <= min_target_val < max_target_val
        assert torch.all(min_target_val <= target_img) \
               and torch.all(target_img <= max_target_val)
        
        # normalize image dimensions
        if target_img.dim() < 4:
            new_shape = (4 - target_img.dim()) * ( 1, ) + target_img.shape      # ie. (1, 1/3, img_height, img_width)
            pred_img = pred_img.view(*new_shape)                                # (1, 1/3, img_height, img_width)
            target_img = target_img.view(*new_shape)                            # (1, 1/3, img_height, img_width)
        
        # compute metrics
        metric = easydict.EasyDict({})

        """
        NOTE:
            For monochrome images, `metric.l1` gives the mean L1 distance/loss
            across pixels (& batches). For RGB images, `metric.l1` gives
            1/3 * mean L1 distance/loss across pixels (& batches), which yields
            a comparable value compared to monochrome images.
        """
        metric.l1 = torch.nn.functional.l1_loss(
            input=pred_img, target=target_img
        )

        """
        NOTE:
            We set `data_range=target_val_range` as PSNR is a relative metric.
        """
        target_val_range = max_target_val - min_target_val
        metric.psnr = torchmetrics.functional.psnr(
            preds=pred_img, target=target_img, data_range=target_val_range,
            reduction="elementwise_mean", dim=(1, 2, 3)
        )

        """
        NOTE:
            We set `data_range=max_target_val` as SSIM is an absolute metric.
        """
        metric.ssim = torchmetrics.functional.ssim(
            preds=pred_img, target=target_img, data_range=max_target_val,
            reduction="elementwise_mean"
        )

        # normalize both predicted & target image values equally so that target
        # image values are in [-1, 1] & convert monochrome images to RGB, if
        # necessary, for LPIPS computation
        pred_img = 2 * (pred_img - min_target_val) / target_val_range - 1       # (batch_size, 1/3, img_height, img_width)
        target_img = 2 * (target_img - min_target_val) / target_val_range - 1   # (batch_size, 1/3, img_height, img_width)
        pred_img = pred_img.expand(-1, 3, -1, -1)                               # (batch_size,   3, img_height, img_width)
        target_img = target_img.expand(-1, 3, -1, -1)                           # (batch_size,   3, img_height, img_width)
        metric.lpips = self.lpips(in0=pred_img, in1=target_img).mean()

        return metric
