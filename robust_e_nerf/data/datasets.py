import os
import glob
import json
import collections
import math
import easydict
import tqdm
import numpy as np
import torch
import cv2
from ..utils import tensor_ops


class Event(torch.utils.data.Dataset):
    RAW_EVENTS_FILENAME = "raw_events.npz"
    TF_EVENTS_FILENAME = "events.pt"
    CAMERA_CALIBRATION_FILENAME = "camera_calibration.npz"
    MAX_REFRACTORY_PERIOD_FILENAME = "max_refractory_period.pt"
    RAW_EVENT_POSITION_KEY = "position"
    RAW_EVENT_TIMESTAMP_KEY = "timestamp"
    RAW_EVENT_POLARITY_KEY = "polarity"
    IMG_HEIGHT_KEY = "img_height"
    IMG_WIDTH_KEY = "img_width"
    DISTORTION_MODEL_KEY = "distortion_model"
    DISTORTION_PARAMS_KEY = "distortion_params"
    INTRINSICS_KEY = "intrinsics"
    BAYER_PATTERN_KEY = "bayer_pattern"
    NULL_BAYER_PATTERN = ""     # ie. monochrome camera
    BAYER_PATTERN_LEN = 4
    COLOR_CHANNEL_NAME_TO_INDEX = {
        "R": 0,
        "G": 1,
        "B": 2
    }

    def __init__(
        self,
        root_directory,
        permutation_seed
    ):
        super().__init__()

        # load transformed events, if it has been cached
        self.events = self.load_transformed_events(root_directory)
        # else, transform raw events and cache it
        if self.events is None:
            camera_calibration = self.load_camera_calibration(root_directory)
            self.events = self.queue_raw_events(
                root_directory, camera_calibration
            )
            self.events = self.colorize_events(self.events, camera_calibration)
            self.events = self.undistort_events(
                self.events, camera_calibration
            )
            self.save_transformed_events(self.events, root_directory)

        # randomly permutate the transformed events to enable the emulation of
        # sparse events, due to a large refractory period, with
        # `train_dataset_ratio`, if necessary
        if permutation_seed is not None:
            perm_indices = tensor_ops.randperm_manual_seed(
                len(self.events.position), permutation_seed
            )
            for key, value in self.events.items():
                self.events[key] = value[perm_indices]

    @classmethod
    def load_transformed_events(
        cls,
        root_directory
    ):
        transformed_events_filepath = os.path.join(
            root_directory, cls.TF_EVENTS_FILENAME
        )

        if os.path.isfile(transformed_events_filepath):
            transformed_events = easydict.EasyDict(
                torch.load(transformed_events_filepath)
            )
        else:
            transformed_events = None
        return transformed_events

    @classmethod
    def save_transformed_events(
        cls,
        transformed_events,
        root_directory
    ):
        print("Saving transformed events...")
        transformed_events_filepath = os.path.join(
            root_directory, cls.TF_EVENTS_FILENAME
        )
        torch.save(dict(transformed_events), transformed_events_filepath)
        print("Done!")

    @classmethod
    def load_raw_events(cls, root_directory):
        raw_events_path = os.path.join(root_directory, cls.RAW_EVENTS_FILENAME)
        raw_events = np.load(raw_events_path)

        return raw_events

    @classmethod
    def load_camera_calibration(cls, root_directory):
        camera_calibration_path = os.path.join(
            root_directory, cls.CAMERA_CALIBRATION_FILENAME
        )
        camera_calibration = np.load(camera_calibration_path)

        return camera_calibration

    @classmethod
    def load_max_refractory_period(cls, root_directory):
        max_refractory_period_path = os.path.join(
            root_directory, cls.MAX_REFRACTORY_PERIOD_FILENAME
        )
        if os.path.isfile(max_refractory_period_path):
            max_refractory_period = torch.load(max_refractory_period_path)
        else:
            max_refractory_period = None

        return max_refractory_period

    @classmethod
    def save_max_refractory_period(cls, max_refractory_period, root_directory):
        max_refractory_period_path = os.path.join(
            root_directory, cls.MAX_REFRACTORY_PERIOD_FILENAME
        )
        torch.save(max_refractory_period, max_refractory_period_path)

    @classmethod
    def extract_max_refractory_period(cls, raw_events, camera_calibration):
        # extract & verify components of raw events & camera calibration
        raw_event_positions = raw_events[cls.RAW_EVENT_POSITION_KEY]
        raw_event_timestamps = raw_events[cls.RAW_EVENT_TIMESTAMP_KEY]
        raw_event_polarities = raw_events[cls.RAW_EVENT_POLARITY_KEY]
        img_height = camera_calibration[cls.IMG_HEIGHT_KEY]
        img_width = camera_calibration[cls.IMG_WIDTH_KEY]
        assert len(raw_event_positions) == len(raw_event_timestamps) \
               == len(raw_event_polarities)

        # initialize maximum refractory period extraction
        raw_event_ts_sliding_windows = [                                        # (img_height, img_width) list of (2) deques
            [ collections.deque(maxlen=2)
              for x in range(img_width) ]
            for y in range(img_height)
        ]
        max_refractory_period = np.array(float("inf"))

        # extract the maximum refractory period as the minimum event time
        # interval across all event substreams of each pixel
        print("Extracting maximum refractory period...")
        for raw_event_position, raw_event_timestamp \
        in zip(tqdm.tqdm(raw_event_positions), raw_event_timestamps):
            # extract other components of this event
            raw_event_x, raw_event_y = raw_event_position
            raw_event_ts_sliding_window = (
                raw_event_ts_sliding_windows[raw_event_y][raw_event_x]
            )

            # skip this event, if its timestamp coincides with the previous
            # event at this pixel location / event position
            if (
                len(raw_event_ts_sliding_window) > 0
                and raw_event_timestamp == raw_event_ts_sliding_window[-1]
            ):
                continue

            # update the event timestamp sliding window at this pixel
            raw_event_ts_sliding_window.append(raw_event_timestamp)

            # update the maximum refractory period at this pixel, if applicable
            if len(raw_event_ts_sliding_window) < 2:
                continue

            raw_event_interval = raw_event_ts_sliding_window[1] \
                                 - raw_event_ts_sliding_window[0]
            max_refractory_period = min(
                max_refractory_period, raw_event_interval
            )
        print("Done!")

        # cast maximum refractory period extraction to `torch.Tensor`
        max_refractory_period = torch.tensor(max_refractory_period)

        return max_refractory_period

    @classmethod
    def queue_raw_events(
        cls,
        root_directory,
        camera_calibration
    ):
        # extract & verify components of raw events & camera calibration
        raw_events = cls.load_raw_events(root_directory)
        raw_event_positions = raw_events[cls.RAW_EVENT_POSITION_KEY]
        raw_event_timestamps = raw_events[cls.RAW_EVENT_TIMESTAMP_KEY]
        raw_event_polarities = raw_events[cls.RAW_EVENT_POLARITY_KEY]
        img_height = camera_calibration[cls.IMG_HEIGHT_KEY]
        img_width = camera_calibration[cls.IMG_WIDTH_KEY]
        assert len(raw_event_positions) == len(raw_event_timestamps) \
               == len(raw_event_polarities)

        # cast event positions from `np.uint16` to `np.int64` & convert event
        # polarities from {True, False} to {1, 0}
        raw_event_positions = raw_event_positions.astype(np.int64)
        raw_event_polarities = raw_event_polarities.astype(np.int64)

        # initialize
        raw_event_ts_sliding_windows = [                                        # (img_height, img_width) list of (2) deques
            [ collections.deque(maxlen=2)                                       # `maxlen=2` to compute the duration for subsequent event
              for x in range(img_width) ]
            for y in range(img_height)
        ]
        raw_event_pol_sliding_windows = [                                       # (img_height, img_width) list of (2) deques
            [ collections.deque(maxlen=2)                                       # `maxlen=2` to compute the duration for subsequent event
              for x in range(img_width) ]
            for y in range(img_height)
        ]
        num_events = len(raw_event_positions)
        queued_events = easydict.EasyDict({
            "position": raw_event_positions,
            "start_ts": np.empty_like(raw_event_timestamps),
            "end_ts": raw_event_timestamps,
            "num_pos": np.empty_like(raw_event_timestamps),
            "num_neg": np.empty_like(raw_event_timestamps)
        })
        is_valid = np.ones(num_events, dtype=bool)

        print("Defining `t_prev` for each event...")
        for event_index in tqdm.tqdm(range(num_events)):
            # extract this event
            raw_event_position = raw_event_positions[event_index]
            raw_event_timestamp = raw_event_timestamps[event_index]
            raw_event_polarity = raw_event_polarities[event_index]

            # extract other components of this event
            raw_event_x, raw_event_y = raw_event_position
            raw_event_ts_sliding_window = (
                raw_event_ts_sliding_windows[raw_event_y][raw_event_x]
            )
            raw_event_pol_sliding_window = (
                raw_event_pol_sliding_windows[raw_event_y][raw_event_x]
            )

            # update the raw event sliding window at this pixel
            raw_event_ts_sliding_window.append(raw_event_timestamp)
            raw_event_pol_sliding_window.append(raw_event_polarity)

            # update the queued events, if the sliding window is full &
            # the first & last timestamp in the sliding window are distinct
            if (
                (len(raw_event_ts_sliding_window)
                 < raw_event_ts_sliding_window.maxlen)
                or (raw_event_ts_sliding_window[0]
                    == raw_event_ts_sliding_window[-1])
            ):
                is_valid[event_index] = False
                continue

            queued_events.start_ts[event_index] = (
                raw_event_ts_sliding_window[0]
            )
            # the earliest event is only used to compute the duration for
            # subsequent event
            queued_events.num_pos[event_index] = (
                sum(raw_event_pol_sliding_window)
                - raw_event_pol_sliding_window[0]
            )
            queued_events.num_neg[event_index] = (
                (raw_event_pol_sliding_window.maxlen - 1)
                - queued_events.num_pos[event_index]
            )
        print("Done!")
        del raw_events, raw_event_positions, \
            raw_event_timestamps, raw_event_polarities, \
            raw_event_ts_sliding_windows, raw_event_pol_sliding_window

        # cast valid queued event components to `torch.Tensor`
        for key, value in queued_events.items():
            queued_events[key] = torch.tensor(value[is_valid, ...])
        
        return queued_events

    @classmethod
    def colorize_events(cls, events, camera_calibration):
        # extract & verify bayer pattern
        bayer_pattern = str(camera_calibration[cls.BAYER_PATTERN_KEY])
        assert len(bayer_pattern) in (
            len(cls.NULL_BAYER_PATTERN), cls.BAYER_PATTERN_LEN
        )

        # skip event colorization, if the event camera is monochrome
        if bayer_pattern == cls.NULL_BAYER_PATTERN:
            return events
        assert set(cls.COLOR_CHANNEL_NAME_TO_INDEX.keys()) \
               == set(bayer_pattern)

        # convert the bayer pattern string to channel indices
        bayer_pattern = list(bayer_pattern)
        bayer_pattern_channel_indices = [
            cls.COLOR_CHANNEL_NAME_TO_INDEX[color_channel_name]
            for color_channel_name in bayer_pattern
        ]

        # derive the bayer pattern pos. indices for the events based on its pos
        is_position_even = (events.position % 2 == 0)
        is_x_even = is_position_even[:, 0]
        is_y_even = is_position_even[:, 1]
        is_bayer_index = cls.BAYER_PATTERN_LEN * [ None ]
        is_bayer_index[0] =   is_x_even  &   is_y_even      # ie. top-left
        is_bayer_index[1] = (~is_x_even) &   is_y_even      # ie. top-right
        is_bayer_index[2] =   is_x_even  & (~is_y_even)     # ie. bottom-left
        is_bayer_index[3] = (~is_x_even) & (~is_y_even)     # ie. bottom-right

        # initialize event colorization
        colorized_events = events
        colorized_events.channel_idx = torch.empty(
            len(colorized_events.position), dtype=torch.uint8
        )

        # append color channel indices to events
        for bayer_index in range(cls.BAYER_PATTERN_LEN):
            colorized_events.channel_idx[is_bayer_index[bayer_index]] = (
                bayer_pattern_channel_indices[bayer_index]
            )
        return colorized_events

    @classmethod
    def undistort_events(cls, events, camera_calibration):
        # extract & verify components of camera calibration
        distortion_model = camera_calibration[cls.DISTORTION_MODEL_KEY]
        distortion_params = camera_calibration[cls.DISTORTION_PARAMS_KEY]
        intrinsics = camera_calibration[cls.INTRINSICS_KEY]
        assert len(distortion_params) in (0, 4)

        # undistort events according to the distortion model & parameters
        undistorted_events = events
        undistorted_events.position = events.position.to(
            torch.get_default_dtype()
        )
        if len(distortion_params) == 0:
            return undistorted_events

        if distortion_model == "plumb_bob":
            undistorted_events.position = torch.from_numpy(
                cv2.undistortPoints(
                    events.position.numpy(), intrinsics,
                    distortion_params, P=intrinsics
                ).squeeze(axis=1)
            )
        elif distortion_model == "equidistant":
            undistorted_events.position = torch.from_numpy(
                cv2.fisheye.undistortPoints(
                    events.position.unsqueeze(dim=1).numpy(), intrinsics,
                    distortion_params, P=intrinsics
                ).squeeze(axis=1)
            )
        elif distortion_model == "fov":
            raise NotImplementedError       # TODO
        else:
            raise NotImplementedError
        return undistorted_events

    def __getitem__(self, index):
        return {
            key: value[index]
            for key, value in self.events.items()
        }

    def __len__(self):
        return len(self.events.position)


class PosedImage(torch.utils.data.Dataset):
    STAGES = ( "train", "val", "test" )
    NORMALIZED_SAMPLE_ID_CHAR_LEN = 16
    ACCEPTED_NUM_IMG_CHANNELS = ( 1, 3, 4 ) # ie. Gray, BGR, BGRA format
    T_COPENGL_CCOMMON_ORIENTATION = np.array([[1,  0,  0],
                                              [0, -1,  0],
                                              [0,  0, -1]])

    POSED_IMG_FOLDER_NAME = "views"
    STAGE_TRANSFORMS_FILENAME_FORMAT_STR = "transforms_{}.json"
    HORIZONTAL_FOV_KEY = "camera_angle_x"
    INTRINSICS_KEY = "intrinsics"
    BIT_DEPTH_KEY = "bit_depth"
    IMG_METADATA_KEY = "frames"
    IMG_PATH_KEY = "file_path"
    IMG_POSE_KEY = "transform_matrix"

    RENDERER_PARAMS_FILENAME = "renderer_params.npz"
    INTERM_COLOR_SPACE_KEY = "interm_color_space"
    LOG_EPS_KEY = "log_eps"

    BAYER_PATTERN_KEY = "bayer_pattern"
    NULL_BAYER_PATTERN = ""                 # ie. monochrome camera

    def __init__(
        self,
        root_directory,
        stage,
        permutation_seed,
        alpha_over_white_bg=False
    ):
        super().__init__()        
        assert stage in self.STAGES

        stage_transforms = self.load_stage_transforms(root_directory, stage)
        renderer_params = self.load_renderer_params(root_directory)
        camera_calibration = Event.load_camera_calibration(root_directory)

        self.posed_imgs = self.load_posed_imgs(
            root_directory, stage_transforms
        )
        self.posed_imgs = self.transform_img(
            self.posed_imgs, alpha_over_white_bg,
            stage_transforms, renderer_params, camera_calibration
        )
        self.posed_imgs = self.transform_pose(self.posed_imgs)

        # randomly permutate the images & their poses
        if permutation_seed is None:
            return
        perm_indices = tensor_ops.randperm_manual_seed(
            len(self.posed_imgs.img), permutation_seed
        )
        for key, value in self.posed_imgs.items():
            if key != "intrinsics":
                self.posed_imgs[key] = value[perm_indices]

    @classmethod
    def posed_img_folder_path(cls, root_directory):
        # the posed image folder is either in the root dir. or 1 level above it
        posed_img_folder_paths = [
            os.path.join(root_directory, cls.POSED_IMG_FOLDER_NAME),
            os.path.join(root_directory, "..", cls.POSED_IMG_FOLDER_NAME)
        ]
        for path in posed_img_folder_paths:
            if os.path.isdir(path):
                return path

    @classmethod
    def load_stage_transforms(cls, root_directory, stage):
        stage_transforms_path = os.path.join(
            cls.posed_img_folder_path(root_directory),
            cls.STAGE_TRANSFORMS_FILENAME_FORMAT_STR.format(stage)
        )
        with open(stage_transforms_path) as stage_transforms_file:
            stage_transforms = json.load(stage_transforms_file)

        return stage_transforms

    @classmethod
    def load_renderer_params(cls, root_directory):
        renderer_params_path = os.path.join(
            root_directory, cls.RENDERER_PARAMS_FILENAME
        )
        if os.path.isfile(renderer_params_path):
            renderer_params = np.load(renderer_params_path)
        else:
            renderer_params = None

        return renderer_params

    @classmethod
    def load_posed_imgs(cls, root_directory, stage_transforms):
        # load the images & extract the camera extrinsics / poses of each image
        posed_imgs = easydict.EasyDict({
            "sample_id": [],
            "img": [],
            "T_wc_position": [],
            "T_wc_orientation": [],
            "intrinsics": None
        })
        for img_metadata in stage_transforms[cls.IMG_METADATA_KEY]:
            sample_id = os.path.basename(img_metadata[cls.IMG_PATH_KEY])

            # normalize the sample id to `cls.NORMALIZED_SAMPLE_ID_CHAR_LEN`
            # characters by padding it with a trailing space whenever necessary
            normalized_sample_id = sample_id.ljust(
                cls.NORMALIZED_SAMPLE_ID_CHAR_LEN
            )

            # convert the `cls.NORMALIZED_SAMPLE_ID_CHAR_LEN`-character
            # normalized sample id to an array of Unicode code point integers
            # with shape (cls.NORMALIZED_SAMPLE_ID_CHAR_LEN)
            normalized_sample_id = np.asarray(                                  # (cls.NORMALIZED_SAMPLE_ID_CHAR_LEN)
                list(map(ord, normalized_sample_id))
            )
            posed_imgs.sample_id.append(normalized_sample_id)

            img_path = glob.glob(os.path.join(
                cls.posed_img_folder_path(root_directory),
                img_metadata[cls.IMG_PATH_KEY] + ".*"
            ))[0]
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)                    # (H, W [, 3/4]), in Gray/BGR/BGRA format 
            posed_imgs.img.append(img)

            T_wc = np.array(img_metadata[cls.IMG_POSE_KEY])                     # (4, 4)
            posed_imgs.T_wc_position.append(T_wc[:3, 3])                        # (3)
            posed_imgs.T_wc_orientation.append(T_wc[:3, :3])                    # (3, 3)
            
        # cast the relevant posed image components to `numpy.ndarray`
        for key, value in posed_imgs.items():
            if key != "intrinsics":
                posed_imgs[key] = np.stack(value, axis=0)

        # cast the sample IDs to `torch.Tensor`
        posed_imgs.sample_id = torch.tensor(posed_imgs.sample_id)

        # extract or derive the camera intrinsics
        assert (cls.HORIZONTAL_FOV_KEY in stage_transforms.keys()) \
               or (cls.INTRINSICS_KEY in stage_transforms.keys())
        if cls.HORIZONTAL_FOV_KEY in stage_transforms.keys():
            H, W = posed_imgs.img.shape[1:3]
            horizontal_fov = stage_transforms[cls.HORIZONTAL_FOV_KEY]
            focal_len = (W / 2) / math.tan(horizontal_fov / 2)
            posed_imgs.intrinsics = np.array([                                  # (3, 3)
                [ focal_len, 0,         W/2 - 0.5 ],
                [ 0,         focal_len, H/2 - 0.5 ],
                [ 0,         0,         1         ]
            ])
        else:   # elif cls.INTRINSICS_KEY in stage_transforms.keys():
            posed_imgs.intrinsics = np.array(
                stage_transforms[cls.INTRINSICS_KEY]
            )

        return posed_imgs

    def transform_img(
        self,
        posed_imgs,
        alpha_over_white_bg,
        stage_transforms,
        renderer_params,
        camera_calibration
    ):
        # deduce some image properties
        is_quantized = np.issubdtype(posed_imgs.img.dtype, np.unsignedinteger)
        is_synthetic = (renderer_params != None)
        if posed_imgs.img.ndim == 3:
            num_img_channels = 1
        elif posed_imgs.img.ndim == 4:
            num_img_channels = posed_imgs.img.shape[3]
        bayer_pattern = camera_calibration[self.BAYER_PATTERN_KEY]

        if is_quantized:
            has_non_default_bit_depth = (self.BIT_DEPTH_KEY
                                         in stage_transforms.keys())
            if has_non_default_bit_depth:
                num_quantization_levels = (
                    2 ** stage_transforms[self.BIT_DEPTH_KEY]
                )
            else:
                num_quantization_levels = (
                    np.iinfo(posed_imgs.img.dtype).max + 1
                )
        if is_synthetic:
            interm_color_space = renderer_params[self.INTERM_COLOR_SPACE_KEY]

        # verify the validity of images
        """
        NOTE:
            1. Images should only be arrays of either:
               a. Unsigned integers, given by real captures from image sensors
                  with Analog-to-Digital Converter (ADC) quantization
                  circuitry, or quantized synthetic display color space renders
               b. Positive floats, given by unquantized synthetic linear color
                  space renders
            2. Each image is an array of shape (H, W [, 3/4]) in Gray/BGR/BGRA
               format, where only synthetic renders can have the alpha channel
        """
        assert np.issubdtype(posed_imgs.img.dtype, np.unsignedinteger) \
               or np.issubdtype(posed_imgs.img.dtype, np.floating)
        assert np.all(posed_imgs.img >= 0)
        if is_synthetic:
            if is_quantized:
                assert interm_color_space == "display"
            else:
                assert interm_color_space == "linear"
        else:
            assert is_quantized

        assert num_img_channels in self.ACCEPTED_NUM_IMG_CHANNELS
        if num_img_channels == 4:
            assert is_synthetic
        
        # alpha-composite the image over a white background (in the output
        # color space), if the images are synthetically rendered BGRA images
        # & necessary
        if alpha_over_white_bg:
            if interm_color_space == "display":
                alpha = posed_imgs.img[..., 3] / (num_quantization_levels - 1)  # (N, H, W)
                alpha = alpha[..., np.newaxis]                                  # (N, H, W, 1)
                
                # BGR components are represented as straight alpha
                posed_imgs.img = alpha * posed_imgs.img[..., :3] \
                                 + (1 - alpha) * (num_quantization_levels - 1)  # (N, H, W, 3) in BGR format
            elif interm_color_space == "linear":
                alpha = posed_imgs.img[..., 3]                                  # (N, H, W)
                alpha = alpha[..., np.newaxis]                                  # (N, H, W, 1)

                # BGR components are represented as premultiplied alpha
                posed_imgs.img = posed_imgs.img[..., :3] + (1 - alpha)          # (N, H, W, 3) in BGR format
        elif num_img_channels == 4:
            posed_imgs.img = posed_imgs.img[..., :3]                            # (N, H, W, 3) in BGR format

        # cast the images to `np.float32` to support `cv2.cvtColor()`
        posed_imgs.img = posed_imgs.img.astype(np.float32)
        
        # convert the images to (N, 3, H, W) shape in RGB format, if the
        # synthetic/real event camera sensor has a bayer filter
        if bayer_pattern != self.NULL_BAYER_PATTERN:
            posed_imgs.img = np.stack([                                         # (N, H, W, 3) in RGB format
                cv2.cvtColor(img_sample, cv2.COLOR_BGR2RGB)                     # (H, W, 3) in RGB format
                for img_sample in posed_imgs.img                                # (H, W, 3) in BGR format, (N, H, W, 3) in BGR format
            ], axis=0)
            posed_imgs.img = posed_imgs.img.transpose(0, 3, 1, 2)               # (N, 3, H, W) in RGB format

        # else, convert the images to grayscale, if necessary
        elif num_img_channels == 3: # and bayer_pattern == self.NULL_BAYER_PATTERN
            posed_imgs.img = np.stack([                                         # (N, H, W) in Gray format
                cv2.cvtColor(img_sample, cv2.COLOR_BGR2GRAY)                    # (H, W) in Gray format
                for img_sample in posed_imgs.img                                # (H, W, 3) in BGR format, (N, H, W, 3) in BGR format
            ], axis=0)

        # normalize pixel values from [0, `num_quantization_levels`-1] to 
        # [0.5/`num_quantization_levels`, (`num_quantization_levels`-0.5)
        # /`num_quantization_levels`], if the pixel values are quantized
        """
        NOTE:
            The Analog-to-Digital Converter (ADC) of an D-bit image sensor
            yields a quantized raw intensity value of x, for any true analog
            raw (scaled) intensity value in the range of [x, x+1), where x in
            { 0, 1, ..., 2**D-1 }. Thus, it is more accurate to represent the
            quantized intensity values as y = x + 0.5, where y in { 0.5, 1.5, 
            ..., 2**D-0.5 } & normalized values as y' = y / 2**D.
        """
        if is_quantized:
            self.min_normalized_pixel_value = 0.5 / num_quantization_levels
            posed_imgs.img = posed_imgs.img / num_quantization_levels \
                            + self.min_normalized_pixel_value
            self.max_normalized_pixel_value = (
                1 - self.min_normalized_pixel_value
            )

        # else, add a small epsilon to the pixel values
        else:
            self.min_normalized_pixel_value = renderer_params[self.LOG_EPS_KEY]
            posed_imgs.img = posed_imgs.img + self.min_normalized_pixel_value
            self.max_normalized_pixel_value = posed_imgs.img.max()

        # cast the images to `torch.Tensor`
        posed_imgs.img = torch.tensor(posed_imgs.img,
                                      dtype=torch.get_default_dtype())

        return posed_imgs

    @classmethod
    def transform_pose(cls, posed_imgs):
        # convert the loaded camera pose in OpenGL convention, where it gives
        # the rigid body transformation of the OpenGL camera frame (x-axis
        # points to the right, y-axis points upwards & z-axis points to the
        # back/outwards, of an image respectively) wrt. the world frame, to a
        # more common convention, where it gives the transformation of a more
        # common camera frame (x-axis points to the right, y-axis points
        # downwards & z-axis points to the front/inwards, of an image
        # respectively) wrt. the world frame
        T_w_copengl_orientation = posed_imgs.T_wc_orientation
        posed_imgs.T_wc_orientation = T_w_copengl_orientation \
                                      @ cls.T_COPENGL_CCOMMON_ORIENTATION
        
        # cast the camera poses to `torch.Tensor`
        for key in ( "T_wc_position", "T_wc_orientation", "intrinsics" ):
            posed_imgs[key] = torch.tensor(posed_imgs[key],
                                           dtype=torch.get_default_dtype())

        return posed_imgs

    def __getitem__(self, index):
        return {
            key: value[index]
            for key, value in self.posed_imgs.items()
            if key != "intrinsics"
        }

    def __len__(self):
        return len(self.posed_imgs.img)


class CameraPose(torch.utils.data.Dataset):
    CAMERA_POSES_FILENAME = "camera_poses.npz"
    CAMERA_POSES_KEYS = set(
        [ "T_wc_position", "T_wc_orientation", "T_wc_timestamp" ]
    )

    def __init__(
        self,
        root_directory,
        permutation_seed
    ):
        super().__init__()
        self.camera_poses = self.load_camera_poses(root_directory)

        # randomly permute camera poses, if necessary
        if permutation_seed is None:
            return
        perm_indices = tensor_ops.randperm_manual_seed(
            len(self.camera_poses.T_wc_position), permutation_seed
        )
        for key, value in self.camera_poses.items():
            self.camera_poses[key] = value[perm_indices]

    @classmethod
    def load_camera_poses(cls, root_directory):
        camera_poses_path = os.path.join(
            root_directory, cls.CAMERA_POSES_FILENAME
        )
        camera_poses = easydict.EasyDict(np.load(camera_poses_path))
        assert set(camera_poses.keys()) == cls.CAMERA_POSES_KEYS
        
        # cast camera poses components to `torch.Tensor`
        for key, value in camera_poses.items():
            camera_poses[key] = torch.tensor(value)

        return camera_poses

    def __getitem__(self, index):
        return {
            key: value[index] for key, value in self.camera_poses.items()
        }

    def __len__(self):
        return len(self.camera_poses.T_wc_position)
