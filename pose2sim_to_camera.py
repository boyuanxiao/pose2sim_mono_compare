"""Transform Pose2Sim world-space 3D keypoints into camera space."""

import argparse
import csv
import os
import re

import cv2
import numpy as np
import toml


# COCO index -> HALPE_26 TRC column name (only the 13 that Pose2Sim typically triangulates)
KEYPOINT_MAPPING = {
    0: "Nose",
    5: "LShoulder",
    6: "RShoulder",
    7: "LElbow",
    8: "RElbow",
    9: "LWrist",
    10: "RWrist",
    11: "LHip",
    12: "RHip",
    13: "LKnee",
    14: "RKnee",
    15: "LAnkle",
    16: "RAnkle",
}


def find_trc_file(pose3d_dir):
    """Find the non-filtered TRC file in the pose-3d directory.

    Matches pattern like 'name_0-295.trc' but excludes files containing 'filt'.
    """
    pattern = re.compile(r".+_\d+-\d+\.trc$")
    candidates = []
    for f in os.listdir(pose3d_dir):
        if pattern.match(f) and "filt" not in f:
            candidates.append(f)
    if len(candidates) == 0:
        raise FileNotFoundError(f"No non-filtered TRC file found in {pose3d_dir}")
    if len(candidates) > 1:
        raise ValueError(f"Multiple TRC candidates found: {candidates}")
    return os.path.join(pose3d_dir, candidates[0])


def load_trc(trc_path):
    """Parse a TRC file and return frame numbers and marker data.

    Returns:
        frames: 1D array of frame numbers (int)
        times: 1D array of timestamps (float)
        marker_data: dict of {marker_name: Nx3 ndarray} with world coords in metres
    """
    with open(trc_path, "r") as f:
        lines = f.readlines()

    # Line 4 (index 3): marker names, tab-separated
    # Format: Frame#\tTime\tMarker1\t\t\tMarker2\t\t\t...
    header_line = lines[3].strip().split("\t")
    marker_names = [name for name in header_line[2:] if name]  # skip Frame#, Time, empty tabs

    # Data starts at line 7 (index 6) after the blank line
    # But some TRC files have data starting right at line 6 (index 5) with no blank line
    # Find first data line by looking for a line starting with a digit
    data_start = 5
    for i in range(5, len(lines)):
        line = lines[i].strip()
        if line and line[0].isdigit():
            data_start = i
            break

    frames = []
    times = []
    raw_data = []

    for line in lines[data_start:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        frames.append(int(parts[0]))
        times.append(float(parts[1]))
        coords = [float(x) for x in parts[2:]]
        raw_data.append(coords)

    frames = np.array(frames, dtype=int)
    times = np.array(times, dtype=float)
    raw_data = np.array(raw_data)  # shape (N_frames, N_markers * 3)

    # Build marker dict: each marker has 3 consecutive columns (X, Y, Z)
    marker_data = {}
    for i, name in enumerate(marker_names):
        col_start = i * 3
        marker_data[name] = raw_data[:, col_start : col_start + 3]

    return frames, times, marker_data


def load_calibration(toml_path, camera_name):
    """Load camera calibration from a Pose2Sim TOML file.

    Returns:
        R: 3x3 rotation matrix (world to camera)
        t: translation vector (3,)
        intrinsic_matrix: 3x3 camera intrinsic matrix
    """
    calib = toml.load(toml_path)
    if camera_name not in calib:
        available = [k for k in calib if k != "metadata"]
        raise KeyError(
            f"Camera '{camera_name}' not found in calibration. Available: {available}"
        )
    cam = calib[camera_name]

    rodrigues_vec = np.array(cam["rotation"], dtype=np.float64)
    R, _ = cv2.Rodrigues(rodrigues_vec)

    t = np.array(cam["translation"], dtype=np.float64)
    intrinsic_matrix = np.array(cam["matrix"], dtype=np.float64)

    return R, t, intrinsic_matrix


def world_to_camera(points_world, R, t):
    """Transform Nx3 world-space points to camera space.

    P_camera = R @ P_world + t
    """
    return (R @ points_world.T).T + t


def extract_shared_keypoints(marker_data):
    """Extract the 13 shared COCO keypoints from TRC marker data.

    Returns:
        keypoint_info: list of (coco_index, name) tuples, sorted by coco_index
        keypoints: ndarray of shape (N_frames, 13, 3)
    """
    keypoint_info = []
    arrays = []

    for coco_idx in sorted(KEYPOINT_MAPPING.keys()):
        name = KEYPOINT_MAPPING[coco_idx]
        if name not in marker_data:
            print(f"Warning: keypoint '{name}' (COCO {coco_idx}) not found in TRC, skipping")
            continue
        keypoint_info.append((coco_idx, name))
        arrays.append(marker_data[name])

    # Stack: each array is (N_frames, 3) -> result is (N_frames, K, 3)
    keypoints = np.stack(arrays, axis=1)
    return keypoint_info, keypoints


def main():
    parser = argparse.ArgumentParser(
        description="Transform Pose2Sim 3D keypoints into camera space"
    )
    parser.add_argument("--dir", required=True, help="Pose2Sim result directory")
    parser.add_argument("--camera", required=True, help="Camera name (e.g. int_cam01_img)")
    parser.add_argument("--num-frames", type=int, default=None,
                        help="Sample N frames evenly spaced across the full range")
    parser.add_argument("--output", default=None, help="Output CSV path (default: auto)")
    args = parser.parse_args()

    pose3d_dir = os.path.join(args.dir, "pose-3d")
    calib_path = os.path.join(args.dir, "calibration", "Calib_scene.toml")

    # Find and load TRC
    trc_path = find_trc_file(pose3d_dir)
    print(f"TRC file: {trc_path}")
    frames, times, marker_data = load_trc(trc_path)
    print(f"Loaded {len(frames)} frames, {len(marker_data)} markers: {list(marker_data.keys())}")

    # Subsample frames if requested
    if args.num_frames is not None and args.num_frames < len(frames):
        indices = np.linspace(0, len(frames) - 1, args.num_frames, dtype=int)
        frames = frames[indices]
        times = times[indices]
        marker_data = {name: data[indices] for name, data in marker_data.items()}
        print(f"Sampled {len(frames)} frames: {frames.tolist()}")

    # Load calibration
    R, t, intrinsic_matrix = load_calibration(calib_path, args.camera)
    print(f"Camera '{args.camera}': R shape {R.shape}, t = {t}")

    # Extract shared keypoints
    keypoint_info, keypoints_world = extract_shared_keypoints(marker_data)
    print(f"Extracted {len(keypoint_info)} shared keypoints, shape {keypoints_world.shape}")

    # Transform each frame to camera space
    n_frames, n_kpts, _ = keypoints_world.shape
    keypoints_cam = np.zeros_like(keypoints_world)
    for i in range(n_frames):
        keypoints_cam[i] = world_to_camera(keypoints_world[i], R, t)

    # Sanity check: Z should be positive (in front of camera)
    z_vals = keypoints_cam[..., 2]
    n_negative = np.sum(z_vals < 0)
    if n_negative > 0:
        print(f"Warning: {n_negative} keypoints have negative Z (behind camera)")
    print(f"Z range: [{z_vals.min():.3f}, {z_vals.max():.3f}] m")

    # Save to CSV
    output_path = args.output or os.path.join(
        os.path.dirname(trc_path),
        f"pose2sim_camera_{args.camera}.csv",
    )
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "keypoint_index", "keypoint_name", "X", "Y", "Z"])
        for i in range(n_frames):
            for j, (coco_idx, name) in enumerate(keypoint_info):
                x, y, z = keypoints_cam[i, j]
                writer.writerow([frames[i], coco_idx, name, f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"])

    print(f"Saved {n_frames * len(keypoint_info)} rows to {output_path}")


if __name__ == "__main__":
    main()