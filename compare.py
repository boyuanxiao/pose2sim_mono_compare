"""Compare Pose2Sim and RTMPose3D 3D keypoints after root-centering and scale fitting."""

import argparse
import csv
import os
from collections import defaultdict

import numpy as np

from pose2sim_to_camera import KEYPOINT_MAPPING

# Sorted COCO indices of shared keypoints: [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
SHARED_COCO_INDICES = sorted(KEYPOINT_MAPPING.keys())

# Position of LHip (COCO 11) and RHip (COCO 12) within the 13-keypoint array
LHIP_POS = SHARED_COCO_INDICES.index(11)  # 7
RHIP_POS = SHARED_COCO_INDICES.index(12)  # 8


def load_keypoints_csv(csv_path):
    """Load keypoints CSV into a per-frame dict.

    Returns:
        dict: {frame_num (int): ndarray of shape (K, 3)} where K is the
              number of keypoints per frame, ordered by keypoint_index.
    """
    frames = defaultdict(dict)
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row["frame"])
            kpt_idx = int(row["keypoint_index"])
            coords = np.array([float(row["X"]), float(row["Y"]), float(row["Z"])])
            frames[frame][kpt_idx] = coords

    # Convert to ordered arrays
    result = {}
    for frame, kpts in frames.items():
        ordered = [kpts[idx] for idx in SHARED_COCO_INDICES if idx in kpts]
        if len(ordered) == len(SHARED_COCO_INDICES):
            result[frame] = np.array(ordered)  # (13, 3)
    return result


def apply_axis_swap(kpts):
    """Apply the RTMPose3D axis convention swap to camera-space coordinates.

    Standard camera: X=right, Y=down, Z=forward
    RTMPose3D convention: X=horizontal, Y=depth, Z=vertical
    Transform: new = -old[:, [0, 2, 1]]
    """
    return -kpts[:, [0, 2, 1]]


def align_and_compare(p2s_kpts, rtm_kpts):
    """Root-centre, scale-fit, and compute per-joint discrepancy for one frame.

    Args:
        p2s_kpts: (13, 3) Pose2Sim keypoints (already axis-swapped).
        rtm_kpts: (13, 3) RTMPose3D keypoints.

    Returns:
        discrepancies: (13,) per-joint Euclidean distance in metres.
        scale: best-fit scale factor.
    """
    # Hip centre
    p2s_hip = (p2s_kpts[LHIP_POS] + p2s_kpts[RHIP_POS]) / 2
    rtm_hip = (rtm_kpts[LHIP_POS] + rtm_kpts[RHIP_POS]) / 2

    # Root-centre
    p2s_centered = p2s_kpts - p2s_hip
    rtm_centered = rtm_kpts - rtm_hip

    # Best-fit scale: s = sum(rtm * p2s) / sum(rtm^2)
    numerator = np.sum(rtm_centered * p2s_centered)
    denominator = np.sum(rtm_centered ** 2)
    if denominator < 1e-12:
        return np.full(len(SHARED_COCO_INDICES), np.nan), np.nan
    scale = numerator / denominator

    # Apply scale
    rtm_scaled = rtm_centered * scale

    # Per-joint discrepancy
    discrepancies = np.linalg.norm(p2s_centered - rtm_scaled, axis=1)

    return discrepancies, scale


def main():
    parser = argparse.ArgumentParser(
        description="Compare Pose2Sim and RTMPose3D 3D keypoints"
    )
    parser.add_argument("--pose2sim-csv", required=True,
                        help="CSV from pose2sim_to_camera.py")
    parser.add_argument("--rtmpose3d-csv", required=True,
                        help="CSV from rtmpose3d_extract.py")
    parser.add_argument("--output", default=None,
                        help="Output CSV path (default: comparison_results.csv)")
    args = parser.parse_args()

    # Load both CSVs
    p2s_data = load_keypoints_csv(args.pose2sim_csv)
    rtm_data = load_keypoints_csv(args.rtmpose3d_csv)
    print(f"Pose2Sim: {len(p2s_data)} frames")
    print(f"RTMPose3D: {len(rtm_data)} frames")

    # Find common frames
    common_frames = sorted(set(p2s_data.keys()) & set(rtm_data.keys()))
    if not common_frames:
        print("Error: no common frames between the two CSVs.")
        return
    print(f"Common frames: {len(common_frames)} — {common_frames}")

    # Build keypoint info
    keypoint_info = [(idx, KEYPOINT_MAPPING[idx]) for idx in SHARED_COCO_INDICES]

    # Compare each frame
    all_discrepancies = []  # (N_frames, 13)
    all_scales = []

    for frame in common_frames:
        # Apply axis swap to Pose2Sim camera-space data
        p2s_swapped = apply_axis_swap(p2s_data[frame])
        rtm_kpts = rtm_data[frame]

        discrepancies, scale = align_and_compare(p2s_swapped, rtm_kpts)
        all_discrepancies.append(discrepancies)
        all_scales.append(scale)

    all_discrepancies = np.array(all_discrepancies)  # (N_frames, 13)
    all_scales = np.array(all_scales)  # (N_frames,)

    # Save output CSV
    output_path = args.output or "comparison_results.csv"
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "keypoint_index", "keypoint_name",
                         "discrepancy_m", "scale_factor"])
        for i, frame in enumerate(common_frames):
            for j, (coco_idx, name) in enumerate(keypoint_info):
                writer.writerow([
                    frame, coco_idx, name,
                    f"{all_discrepancies[i, j]:.6f}",
                    f"{all_scales[i]:.6f}",
                ])

    print(f"\nSaved {len(common_frames) * len(keypoint_info)} rows to {output_path}")

    # Print summary statistics
    print(f"\nScale factor: mean={all_scales.mean():.4f}, "
          f"std={all_scales.std():.4f}, "
          f"range=[{all_scales.min():.4f}, {all_scales.max():.4f}]")

    print(f"\nPer-keypoint discrepancy (metres) across {len(common_frames)} frames:")
    print(f"{'Keypoint':<15} {'Mean':>8} {'Median':>8} {'Max':>8}")
    print("-" * 41)
    for j, (coco_idx, name) in enumerate(keypoint_info):
        col = all_discrepancies[:, j]
        print(f"{name:<15} {col.mean():>8.4f} {np.median(col):>8.4f} {col.max():>8.4f}")

    overall = all_discrepancies.mean()
    print("-" * 41)
    print(f"{'Overall':<15} {overall:>8.4f}")


if __name__ == "__main__":
    main()
