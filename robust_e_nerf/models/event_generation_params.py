import warnings
import easydict
import torch
from ..utils import modules
from ..data import datasets


class ContrastThreshold(torch.nn.Module):
    POS_CONTRAST_THRESHOLD_KEY = "pos_contrast_threshold"
    NEG_CONTRAST_THRESHOLD_KEY = "neg_contrast_threshold"

    def __init__(self, dataset_directory):
        super().__init__()
        camera_calibration = datasets.Event.load_camera_calibration(
            dataset_directory
        )
        calibrated_pos_contrast_threshold = torch.from_numpy(
            camera_calibration[self.POS_CONTRAST_THRESHOLD_KEY]
        )
        calibrated_neg_contrast_threshold = torch.from_numpy(
            camera_calibration[self.NEG_CONTRAST_THRESHOLD_KEY]
        )
        calibrated_p2n_contrast_threshold_ratio = (
            calibrated_pos_contrast_threshold
            / calibrated_neg_contrast_threshold
        )
        assert calibrated_p2n_contrast_threshold_ratio > 0

        # define calibrated positive-to-negative contrast threshold ratio &
        # negative contrast threshold as buffer
        """
        NOTE:
            A value of 1 is an appropriate initialization for the positive-to
            -negative contrast threshold ratio, if uncalibrated. It can also be
            estimated by empirically calculating the no. of positive-to
            -negative events within a sufficiently long time interval.
        """
        self.register_buffer(
            "init_p2n_contrast_threshold_ratio",
            calibrated_p2n_contrast_threshold_ratio,
            persistent=False
        )
        self.register_buffer(
            "neg_contrast_threshold",
            calibrated_neg_contrast_threshold,
            persistent=False
        )

        # define positive-to-negative contrast threshold ratio parameter
        # (must be > 0)
        softplus = modules.Softplus(beta=1)
        self.p2n_contrast_threshold_ratio = torch.nn.parameter.Parameter(
            calibrated_p2n_contrast_threshold_ratio
        )
        torch.nn.utils.parametrize.register_parametrization(
            self, "p2n_contrast_threshold_ratio", softplus
        )

    @property
    def ref_p2n_contrast_threshold_ratio(self):
        return self.p2n_contrast_threshold_ratio \
               / self.init_p2n_contrast_threshold_ratio

    @property
    def pos_contrast_threshold(self):
        return self.p2n_contrast_threshold_ratio * self.neg_contrast_threshold

    @property
    def mean_contrast_threshold(self):
        return (self.pos_contrast_threshold + self.neg_contrast_threshold) / 2

    def forward(self, input_event):
        """
        Derive the effective difference in log-intensity from the number of
        positive & negative events with the pos. & neg. constrast thresholds
        """
        output_event = easydict.EasyDict(input_event)       
        output_event.log_intensity_diff = (
            output_event.num_pos * self.pos_contrast_threshold
            - output_event.num_neg * self.neg_contrast_threshold
        )
        output_event.pop("num_pos")
        output_event.pop("num_neg")
        return output_event


class RefractoryPeriod(torch.nn.Module):
    REFRACTORY_PERIOD_KEY = "refractory_period"
    REDEFINED_CALIBRATED_REFRACTORY_PERIOD_FACTOR = 0.999
    MIN_SCALED_SHIFTED_SIGMOID_GRAD_MAGNITUDE = 0.0001

    def __init__(self, dataset_directory):
        super().__init__()
        camera_calibration = datasets.Event.load_camera_calibration(
            dataset_directory
        )
        calibrated_refractory_period = torch.from_numpy(
            camera_calibration[self.REFRACTORY_PERIOD_KEY]
        )

        # load the maximum refractory period, if it has been cached
        max_refractory_period = datasets.Event.load_max_refractory_period(
            dataset_directory
        )
        # else, extract the maximum refractory period & cache it
        if max_refractory_period is None:
            raw_events = datasets.Event.load_raw_events(dataset_directory)
            max_refractory_period = (
                datasets.Event.extract_max_refractory_period(
                    raw_events, camera_calibration
                )
            )
            datasets.Event.save_max_refractory_period(
                max_refractory_period, dataset_directory
            )
        if not (0 <= calibrated_refractory_period < max_refractory_period):
            warnings.warn(("Calibrated refractory period ({})"
                           " >= Max. possible refractory period ({}).").format(
                           calibrated_refractory_period, max_refractory_period
                          ))
            calibrated_refractory_period = (
                self.REDEFINED_CALIBRATED_REFRACTORY_PERIOD_FACTOR
                * max_refractory_period
            )
            warnings.warn(("Redefining calibrated refractory period to {} of"
                           " max. possible refractory period ({}).").format(
                           self.REDEFINED_CALIBRATED_REFRACTORY_PERIOD_FACTOR,
                           calibrated_refractory_period
                          ))

        # define calibrated refractory period, maximum refractory_period &
        # maximum scaled logit magnitude (associated to the scaled & shifted
        # sigmoid refractory period parameterization) as buffer
        """
        NOTE:
            A value of `self.max_refractory_period` is an appropriate
            initialization, if uncalibrated.
        """
        self.register_buffer(
            "init_refractory_period",
            calibrated_refractory_period,
            persistent=False
        )
        self.register_buffer(
            "max_refractory_period",
            max_refractory_period,
            persistent=False
        )
        self.register_buffer(
            "max_scaled_logit_magnitude",
            torch.tensor(
                self.MIN_SCALED_SHIFTED_SIGMOID_GRAD_MAGNITUDE
            ).logit().abs(),
            persistent=False
        )

        # define refractory period parameter
        # (must be >= 0 and < `max_refractory_period`)
        scaled_shifted_sigmoid = modules.ScaledShiftedSigmoid(
            low=0, high=max_refractory_period
        )
        self._refractory_period = torch.nn.parameter.Parameter(
            calibrated_refractory_period.to(torch.float64)
        )
        torch.nn.utils.parametrize.register_parametrization(
            self, "_refractory_period", scaled_shifted_sigmoid
        )
        self.clamp_refractory_period()

    @torch.no_grad()
    def clamp_refractory_period(self):
        """
        Clamp the (scaled logits of the) refractory period such that its
        corresponding scaled & shifted sigmoid gradient does not vanish.
        """
        scaled_logit = (
            self.parametrizations._refractory_period.original
            / self.max_refractory_period
        )
        clamped_scaled_logit = scaled_logit.clamp(
            min=-self.max_scaled_logit_magnitude,
            max=self.max_scaled_logit_magnitude
        )
        clamped_logit = self.max_refractory_period * clamped_scaled_logit
        self.parametrizations._refractory_period.original.copy_(clamped_logit)

    @property
    def refractory_period(self):
        self.clamp_refractory_period()
        return self._refractory_period

    @property
    def delta_refractory_period(self):
        return self.refractory_period - self.init_refractory_period

    def forward(self, input_event):
        """
        Delay the initial timestamps of the event intervals by the refractory
        period
        """
        output_event = easydict.EasyDict(input_event)
        output_event.start_ts = output_event.start_ts + self.refractory_period
        return output_event
