"""Compare Pose2Sim and RTMPose3D 3D keypoints using Procrustes alignment."""

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


def procrustes_align(source, target):
    """Procrustes alignment: find rotation, scale, translation mapping source -> target.

    Args:
        source: (K, 3) points to be aligned (RTMPose3D).
        target: (K, 3) reference points (Pose2Sim).

    Returns:
        aligned: (K, 3) source after alignment.
        scale: best-fit scale factor.
        rotation: (3, 3) rotation matrix.
    """
    # 1. Centre both at hip midpoint
    src_hip = (source[LHIP_POS] + source[RHIP_POS]) / 2
    tgt_hip = (target[LHIP_POS] + target[RHIP_POS]) / 2
    src_c = source - src_hip
    tgt_c = target - tgt_hip

    # 2. Cross-covariance matrix and SVD
    H = src_c.T @ tgt_c  # (3, 3)
    U, S, Vt = np.linalg.svd(H)

    # Ensure proper rotation (no reflection)
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1, 1, np.sign(d)])
    rotation = Vt.T @ sign_matrix @ U.T  # (3, 3)

    # 3. Apply rotation, then solve for scale
    src_rotated = src_c @ rotation.T
    scale = np.sum(src_rotated * tgt_c) / np.sum(src_rotated ** 2)

    # 4. Full alignment
    aligned = src_rotated * scale

    return aligned, scale, rotation


def align_and_compare(p2s_kpts, rtm_kpts):
    """Procrustes-align RTMPose3D onto Pose2Sim and compute per-joint discrepancy.

    Args:
        p2s_kpts: (13, 3) Pose2Sim camera-space keypoints.
        rtm_kpts: (13, 3) RTMPose3D keypoints.

    Returns:
        discrepancies: (13,) per-joint Euclidean distance in metres.
        scale: best-fit scale factor.
        rotation: (3, 3) best-fit rotation matrix.
        rtm_aligned: (13, 3) RTMPose3D keypoints after Procrustes alignment.
        p2s_centered: (13, 3) Pose2Sim keypoints centred at hip midpoint.
    """
    p2s_hip = (p2s_kpts[LHIP_POS] + p2s_kpts[RHIP_POS]) / 2
    p2s_centered = p2s_kpts - p2s_hip

    rtm_aligned, scale, rotation = procrustes_align(rtm_kpts, p2s_kpts)

    discrepancies = np.linalg.norm(p2s_centered - rtm_aligned, axis=1)

    return discrepancies, scale, rotation, rtm_aligned, p2s_centered


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
        discrepancies, scale, rotation, _, _ = align_and_compare(
            p2s_data[frame], rtm_data[frame])
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
