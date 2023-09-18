import collections
import numpy as np
import torch


def freeze(module):
    module.requires_grad_(False)
    module.eval()


def unfreeze(module):
    module.requires_grad_(True)
    module.train()


def extract_descendent_state_dict(state_dict, descendent_name):
    descendent_name_prefix = descendent_name + "."
    descendent_state_dict_items = [
        ( key[len(descendent_name_prefix):], value )
        for key, value in state_dict.items()
        if key.startswith(descendent_name_prefix)
    ]
    descendent_state_dict_metadata_items = [
        ( key[len(descendent_name):].lstrip("."), value )
        for key, value in state_dict._metadata.items()
        if key.startswith(descendent_name)
    ]

    descendent_state_dict = collections.OrderedDict(
        descendent_state_dict_items
    )
    descendent_state_dict._metadata = collections.OrderedDict(
        descendent_state_dict_metadata_items
    )
    return descendent_state_dict


class Softplus(torch.nn.Module):
    def __init__(self, beta=1, threshold=20):
        super().__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, input):
        return torch.nn.functional.softplus(input, self.beta, self.threshold)

    def right_inverse(self, input):
        softplus_inverse = torch.log(torch.exp(self.beta * input) - 1) \
                           / self.beta
        linear_inverse = input
        return torch.where(
            input * self.beta > self.threshold,
            linear_inverse,
            softplus_inverse
        )


class ScaledShiftedSigmoid(torch.nn.Module):
    """
    NOTE:
        `ScaledShiftedSigmoid` preserves the gradient profile of `Sigmoid`.
        Specifically, the gradient of `ScaledShiftedSigmoid` with scale k at
        any scalar input x is the gradient of `Sigmoid` at input x / k.
    """
    def __init__(self, low=0, high=1):
        super().__init__()
        self.low = low
        self.scale = high - low

    def forward(self, input):
        return self.scale * torch.sigmoid(input / self.scale) + self.low

    def right_inverse(self, input):
        return self.scale * torch.logit((input - self.low) / self.scale)


class MAPELoss(torch.nn.Module):
    """
    Implementation References:
        1. https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html#torch.nn.L1Loss
        2. https://github.com/scikit-learn/scikit-learn/blob/15a949460/sklearn/metrics/_regression.py#L260
    """
    def __init__(
        self,
        reduction="mean",
        eps=np.finfo(np.float64).eps.item()     # approximately 2.22e-16
    ):
        super().__init__()
        assert reduction in ( "none", "mean", "sum" )

        self.l1loss = torch.nn.L1Loss(reduction="none")
        self.reduction = reduction
        self.eps = eps

    def forward(self, input, target):
        output = self.l1loss(input, target) / target.abs().clamp(min=self.eps)
        if self.reduction != "none":
            numel = output.numel()
            output = output.sum()
        if self.reduction == "mean":
            output = output / numel
        return output
