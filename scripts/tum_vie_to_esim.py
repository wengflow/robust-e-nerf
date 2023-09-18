import os
import sys
import argparse
import json

import easydict
import hdf5plugin
import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import cv2

# insert the project / script parent directory into the module search path
PROJECT_DIR = os.path.join(sys.path[0], '..')
sys.path.insert(1, PROJECT_DIR)
import robust_e_nerf as ren


T_CCOMMON_COPENGL = np.array([[1,  0,  0, 0],
                              [0, -1,  0, 0],
                              [0,  0, -1, 0],
                              [0,  0,  0, 1]], dtype=np.float32)
US_TO_NS = int(1e+3)

# calibration constants
CAMERA_CALIBRATION_CONFIG_FILENAME_FORMAT_STR = "camera-calibration{}.json"
MOCAP_IMU_CALIBRATION_CONFIG_FILENAME_FORMAT_STR = (
    "mocap-imu-calibration{}.json"
)
SEQUENCE_NAMES_WITH_CONFIG_ID_A = [
    "loop-floor0", "loop-floor1", "loop-floor2", "loop-floor3",
    "mocap-desk", "mocap-desk2", "skate-easy"
]

CAMERA_POSITIONS = [ "left", "right" ]
CAMERA_INDICES = easydict.EasyDict({
    "rgb": { "left": 0,
             "right": 1 },
    "event": { "left": 2,
               "right": 3 }
})
TRIM_INITIAL_NUM_IMAGES = 80

# raw dataset file/folder names
RAW_EVENTS_FILENAME_FORMAT_STR = "{}-events_{}.h5"
NON_RAW_EVENTS_FOLDER_NAME_FORMAT_STR = "{}-vi_gt_data"
MARKER_POSES_FILENAME = "mocap_data.txt"
DISTORTED_IMAGES_FOLDER_NAME_FORMAT_STR = "{}_images"
IMAGE_TIMESTAMPS_FILENAME_FORMAT_STR = "image_timestamps_{}.txt"
IMAGE_FILENAME_FORMAT_STR = "{:05d}.jpg"

# preprocessed dataset file/folder names
PREPROCESSED_EVENTS_FILENAME = "raw_events.npz"
PREPROCESSED_EVENT_CAMERA_POSES_FILENAME = "camera_poses.npz"
PREPROCESSED_EVENT_CAMERA_CALIBRATION_FILENAME = "camera_calibration.npz"
POSED_UNDISTORTED_IMAGES_FOLDER_NAME = "views"
STAGE = "val"
STAGE_TRANSFORMS_FILENAME_FORMAT_STR = "transforms_{}.json"

# assumed / estimated event camera parameters
ESTIMATED_REFRACTORY_PERIOD = 1375
ASSUMED_NEGATIVE_CONTRAST_THRESHOLD = 0.25
ESTIMATED_POSITIVE_TO_NEGATIVE_CONTRAST_THRESHOLD_RATIO = 1.458
NULL_BAYER_PATTERN = ""     # ie. monochrome camera


def main(args):
    # derive config ID, non-raw events path & camera indices from user input
    if args.sequence_name in SEQUENCE_NAMES_WITH_CONFIG_ID_A:
        config_id = "A"
    else:
        config_id = "B"
    non_raw_events_path = os.path.join(
        args.raw_dataset_path,
        NON_RAW_EVENTS_FOLDER_NAME_FORMAT_STR.format(args.sequence_name)
    )
    rgb_cam_idx = CAMERA_INDICES.rgb[args.camera_position]
    event_cam_idx = CAMERA_INDICES.event[args.camera_position]

    # create the preprocessed dataset directory, if necessary
    os.makedirs(args.preprocessed_dataset_path, exist_ok=True)

    # load the TUM-VIE calibration results from the config file
    camera_calibration_path = os.path.join(
        args.raw_dataset_path,
        CAMERA_CALIBRATION_CONFIG_FILENAME_FORMAT_STR.format(config_id)
    )
    with open(camera_calibration_path) as f:
        camera_calibration = easydict.EasyDict(json.load(f))
    camera_calibration = camera_calibration.value0

    mocap_imu_calibration_path = os.path.join(
        args.raw_dataset_path,
        MOCAP_IMU_CALIBRATION_CONFIG_FILENAME_FORMAT_STR.format(config_id)
    )
    with open(mocap_imu_calibration_path) as f:
        mocap_imu_calibration = easydict.EasyDict(json.load(f))
    mocap_imu_calibration = mocap_imu_calibration.value0

    # derive & save the event camera calibration parameters into an npz file
    preprocessed_event_calibration_path = os.path.join(
        args.preprocessed_dataset_path,
        PREPROCESSED_EVENT_CAMERA_CALIBRATION_FILENAME
    )
    ev_intr_dist = camera_calibration.intrinsics[event_cam_idx].intrinsics

    event_intrinsics = np.array(                                                # (3, 3)
        [[ ev_intr_dist.fx, 0,               ev_intr_dist.cx ],
         [ 0,               ev_intr_dist.fy, ev_intr_dist.cy ],
         [ 0,               0,               1               ]],
        dtype=np.float32
    )
    event_distortion_params = np.array(                                         # (4)
        [ ev_intr_dist.k1, ev_intr_dist.k2, ev_intr_dist.k3, ev_intr_dist.k4 ],
        dtype=np.float32
    )
    event_distortion_model = np.array({
        "kb4": "equidistant"
    }[camera_calibration.intrinsics[event_cam_idx].camera_type])

    event_img_width, event_img_height = (
        camera_calibration.resolution[event_cam_idx]
    )
    event_img_width = np.array(event_img_width, dtype=np.uint16)
    event_img_height = np.array(event_img_height, dtype=np.uint16)

    neg_contrast_threshold = np.array(
        ASSUMED_NEGATIVE_CONTRAST_THRESHOLD, dtype=np.float32
    )
    pos_contrast_threshold = (
        ESTIMATED_POSITIVE_TO_NEGATIVE_CONTRAST_THRESHOLD_RATIO
        * neg_contrast_threshold
    )
    refractory_period = np.array(ESTIMATED_REFRACTORY_PERIOD, dtype=np.float32)
    bayer_pattern = NULL_BAYER_PATTERN

    np.savez(
        preprocessed_event_calibration_path,
        intrinsics=event_intrinsics,
        distortion_params=event_distortion_params,
        distortion_model=event_distortion_model,
        img_height=event_img_height,
        img_width=event_img_width,
        pos_contrast_threshold=pos_contrast_threshold,
        neg_contrast_threshold=neg_contrast_threshold,
        refractory_period=refractory_period,
        bayer_pattern=bayer_pattern
    )

    # convert & save event camera poses into an npz file
    marker_poses_path = os.path.join(
        non_raw_events_path, MARKER_POSES_FILENAME
    )
    preprocessed_event_poses_path = os.path.join(
        args.preprocessed_dataset_path,
        PREPROCESSED_EVENT_CAMERA_POSES_FILENAME
    )
    marker_poses = np.loadtxt(marker_poses_path)                                # (P, 8)

    T_wm_timestamp = US_TO_NS * marker_poses[:, 0]                              # (P)
    T_wm_timestamp = T_wm_timestamp.astype(np.int64)
    T_wm = marker_poses[:, 1:].astype(np.float32)                               # (P, 7)
    T_wm = se3_vec_to_mat(T_wm)                                                 # (P, 4, 4)

    # trim the sequence to the interval-of-interest
    is_valid_timestamp = (args.start_timestamp <= T_wm_timestamp) \
                         & (T_wm_timestamp < args.end_timestamp)                # (P)
    T_wm_timestamp = T_wm_timestamp[is_valid_timestamp]                         # (P')
    init_T_wm_timestamp = T_wm_timestamp[0]
    T_wm_timestamp = T_wm_timestamp - init_T_wm_timestamp
    T_wm = T_wm[is_valid_timestamp, :, :]                                       # (P', 4, 4)

    # transform marker poses & timestamps to event camera poses & timestamps
    init_T_wc_timestamp = init_T_wm_timestamp
    T_wc_timestamp = T_wm_timestamp

    T_imu_marker = mocap_imu_calibration.T_imu_marker
    T_imu_marker = se3_json_to_mat(T_imu_marker)                                # (4, 4)

    T_imu_event = camera_calibration.T_imu_cam[event_cam_idx]
    T_imu_event = se3_json_to_mat(T_imu_event)                                  # (4, 4)

    T_marker_event = np.linalg.inv(T_imu_marker) @ T_imu_event                  # (4, 4)
    T_wc = T_wm @ T_marker_event                                                # (P', 4, 4)
    T_wc = se3_mat_to_vec(T_wc)                                                 # (P', 7)
    T_wc_position = T_wc[:, :3]                                                 # (P', 3)
    T_wc_orientation = T_wc[:, 3:]                                              # (P', 4)

    np.savez(
        preprocessed_event_poses_path,
        T_wc_position=T_wc_position,
        T_wc_orientation=T_wc_orientation,
        T_wc_timestamp=T_wc_timestamp
    )

    # convert & save events into an npz file
    raw_events_path = os.path.join(
        args.raw_dataset_path,
        RAW_EVENTS_FILENAME_FORMAT_STR.format(
            args.sequence_name, args.camera_position
        )
    )
    preprocessed_events_path = os.path.join(
        args.preprocessed_dataset_path, PREPROCESSED_EVENTS_FILENAME
    )
    with h5py.File(raw_events_path, "r") as f:
        event_position = np.stack((f['events']['x'], f['events']['y']), axis=1) # (N, 2)
        event_timestamp = US_TO_NS * np.array(f['events']['t'])                 # (N)
        event_timestamp = event_timestamp - init_T_wc_timestamp
        event_polarity = np.array(f['events']['p'], dtype=bool)                 # (N)

    # filter out events that occur outside of the T_wc relative pose timestamps
    event_position, event_timestamp, event_polarity = filter_event(
        event_position, event_timestamp, event_polarity, T_wc_timestamp
    )
    np.savez(
        preprocessed_events_path,
        position=event_position,
        timestamp=event_timestamp,
        polarity=event_polarity,
    )

    # derive & save RGB camera intrinsics & poses into a json file
    assert camera_calibration.intrinsics[rgb_cam_idx].camera_type == "kb4"
    rgb_intr_dist = camera_calibration.intrinsics[rgb_cam_idx].intrinsics

    rgb_intrinsics = np.array(                                                  # (3, 3)
        [[ rgb_intr_dist.fx, 0,                rgb_intr_dist.cx ],
         [ 0,                rgb_intr_dist.fy, rgb_intr_dist.cy ],
         [ 0,                0,                1                ]],
        dtype=np.float32
    )
    rgb_distortion_params = np.array(                                           # (4)
        [ rgb_intr_dist.k1, rgb_intr_dist.k2,
          rgb_intr_dist.k3, rgb_intr_dist.k4 ],
        dtype=np.float32
    )
    rgb_img_width, rgb_img_height = camera_calibration.resolution[rgb_cam_idx]
    new_rgb_intrinsics = (                                                      # (3, 3)
        cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            rgb_intrinsics, rgb_distortion_params,
            (rgb_img_width, rgb_img_height), R=np.eye(3, dtype=np.float32),
            balance=0
        )
    )

    # linearly interpolate event camera poses at the image timestamps
    distorted_images_path = os.path.join(
        non_raw_events_path,
        DISTORTED_IMAGES_FOLDER_NAME_FORMAT_STR.format(args.camera_position)
    )
    image_timestamps_path = os.path.join(
        distorted_images_path,
        IMAGE_TIMESTAMPS_FILENAME_FORMAT_STR.format(args.camera_position)
    )
    image_timestamp = np.loadtxt(image_timestamps_path)                         # (I)
    image_timestamp = US_TO_NS * image_timestamp                                # (I)
    image_timestamp = image_timestamp.astype(np.int64)
    image_timestamp = image_timestamp - init_T_wc_timestamp

    is_valid_image = (0 <= image_timestamp) \
                     & (image_timestamp <= T_wc_timestamp[-1])                  # [I]
    is_valid_image[:TRIM_INITIAL_NUM_IMAGES] = False
    image_timestamp = image_timestamp[is_valid_image]                           # [I']

    event_trajectory = ren.models.trajectories.LinearTrajectory(
        easydict.EasyDict({
            "camera_poses": {
                "T_wc_position": torch.from_numpy(T_wc_position),
                "T_wc_orientation": torch.from_numpy(T_wc_orientation),
                "T_wc_timestamp": torch.from_numpy(T_wc_timestamp)
            }
        })
    )
    with torch.no_grad():
        T_w_event_position, T_w_event_orientation = event_trajectory(           # (I', 3), (I', 3, 3)
            torch.from_numpy(image_timestamp)
        )

    # derive the RGB camera poses from the event camera poses
    T_w_event = np.zeros((len(T_w_event_position), 4, 4), dtype=np.float32)     # (I', 4, 4)
    T_w_event[:, :3, 3] = T_w_event_position.numpy()
    T_w_event[:, :3, :3] = T_w_event_orientation.numpy()
    T_w_event[:, 3, 3] = 1

    T_imu_rgb = camera_calibration.T_imu_cam[rgb_cam_idx]
    T_imu_rgb = se3_json_to_mat(T_imu_rgb)                                      # (4, 4)
    T_event_rgb = np.linalg.inv(T_imu_event) @ T_imu_rgb                        # (4, 4)
    T_w_rgb = T_w_event @ T_event_rgb                                           # (I', 4, 4)

    # convert the RGB camera poses from a common to the OpenGL convention
    T_w_rgb = T_w_rgb @ T_CCOMMON_COPENGL                                       # (I', 4, 4)


    posed_undistorted_images_path = os.path.join(
        args.preprocessed_dataset_path, POSED_UNDISTORTED_IMAGES_FOLDER_NAME
    )
    transforms_path = os.path.join(
        posed_undistorted_images_path,
        STAGE_TRANSFORMS_FILENAME_FORMAT_STR.format(STAGE)
    )
    valid_image_index = is_valid_image.nonzero()[0]                             # (I')
    image_filename = list(                                                      # (I')
        map(IMAGE_FILENAME_FORMAT_STR.format, valid_image_index)
    )
    transforms = {
        "intrinsics": new_rgb_intrinsics.tolist(),
        "frames": [ { "file_path": os.path.join(
                                    ".", STAGE, os.path.splitext(filename)[0]
                                   ),
                      "transform_matrix": tf_matrix.tolist() }
                    for filename, tf_matrix in zip(image_filename, T_w_rgb) ]
    }
    os.mkdir(posed_undistorted_images_path)
    with open(transforms_path, "w") as f:
        json.dump(transforms, f, indent=4)

    # undistort the RGB images & save them
    stage_undistorted_images_path = os.path.join(
        posed_undistorted_images_path, STAGE
    )
    os.mkdir(stage_undistorted_images_path)
    for filename in image_filename:
        distorted_image_path = os.path.join(distorted_images_path, filename)
        distorted_image = cv2.imread(
            distorted_image_path, cv2.IMREAD_UNCHANGED
        )
        undistorted_image = cv2.fisheye.undistortImage(
            distorted_image, rgb_intrinsics, rgb_distortion_params,
            Knew=new_rgb_intrinsics
        )
        undistorted_image_path = os.path.join(
            stage_undistorted_images_path, filename
        )
        cv2.imwrite(undistorted_image_path, undistorted_image)


def se3_json_to_mat(se3_json):
    se3_vector = np.array(                                                      # (7)
        [ se3_json.px, se3_json.py, se3_json.pz,
          se3_json.qx, se3_json.qy, se3_json.qz, se3_json.qw ],
        dtype=np.float32
    )
    se3_matrix = se3_vec_to_mat(se3_vector)                                     # (4, 4)
    return se3_matrix


def se3_vec_to_mat(se3_vector):                                                 # ([N,] 7)
    assert se3_vector.ndim in (1, 2)
    assert se3_vector.shape[-1] == 7

    position = se3_vector[..., :3]                                              # ([N,] 3)
    orientation = se3_vector[..., 3:]                                           # ([N,] 4)

    if se3_vector.ndim == 1:
        se3_matrix = np.zeros((4, 4), dtype=position.dtype)                     # (4, 4)
    else:   # elif position.ndim == 2:
        N = se3_vector.shape[0]
        se3_matrix = np.zeros((N, 4, 4), dtype=position.dtype)                  # (N, 4, 4)

    se3_matrix[..., :3, 3] = position
    orientation = Rotation.from_quat(orientation)
    se3_matrix[..., :3, :3] = orientation.as_matrix()                           
    se3_matrix[..., 3, 3] = 1

    return se3_matrix


def se3_mat_to_vec(se3_matrix):                                                 # ([N,] 4, 4)
    assert se3_matrix.ndim in (2, 3)
    assert se3_matrix.shape[-2:] == (4, 4)
    assert np.all(se3_matrix[..., 3, :] == np.array([0, 0, 0, 1]))

    position = se3_matrix[..., :3, 3]                                           # ([N,] 3)
    orientation = Rotation.from_matrix(se3_matrix[..., :3, :3])
    orientation = orientation.as_quat().astype(np.float32)                      # ([N,] 4)
    se3_vector = np.concatenate(( position, orientation ), axis=-1)             # ([N,] 7)

    return se3_vector


def filter_event(
    event_position,
    event_timestamp,
    event_polarity,
    T_wc_timestamp
):
    valid_indices = (T_wc_timestamp[0] <= event_timestamp) \
                    & (event_timestamp <= T_wc_timestamp[-1])
    event_position = event_position[valid_indices, :].copy(order="C")
    event_timestamp = event_timestamp[valid_indices].copy(order="C")
    event_polarity = event_polarity[valid_indices].copy(order="C")

    return event_position, event_timestamp, event_polarity


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Script for converting TUM-VIE datasets to"
                     " pre-processed ESIM format.")
    )
    parser.add_argument(
        "sequence_name", type=str,
        help=("Desired TUM-VIE dataset sequence for conversion.")
    )
    parser.add_argument(
        "raw_dataset_path", type=str,
        help="Path to the raw TUM-VIE datasets folder."
    )
    parser.add_argument(
        "preprocessed_dataset_path", type=str,
        help="Desired path to the pre-processed TUM-VIE dataset."
    )
    parser.add_argument(
        "--camera_position", type=str, choices=CAMERA_POSITIONS,
        default="left", help="Left or right event & RGB camera to convert."
    )
    parser.add_argument(
        "--start_timestamp", type=int, default=0,
        help="Trim the sequence to start at the given timestamp (inclusive)."
    )
    parser.add_argument(
        "--end_timestamp", type=int, default=float("inf"),
        help="Trim the sequence to end at the given timestamp (exclusive)."
    )
    args = parser.parse_args()

    main(args)
