"""Visualize 3D keypoints from Pose2Sim and RTMPose3D side-by-side on a single frame."""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from compare import load_keypoints_csv, align_and_compare, SHARED_COCO_INDICES
from pose2sim_to_camera import KEYPOINT_MAPPING

# Skeleton connections (pairs of COCO indices)
SKELETON = [
    (0, 5), (0, 6),       # Nose -> shoulders
    (5, 7), (7, 9),       # Left arm
    (6, 8), (8, 10),      # Right arm
    (5, 6),               # Shoulder bridge
    (5, 11), (6, 12),     # Torso
    (11, 12),             # Hip bridge
    (11, 13), (13, 15),   # Left leg
    (12, 14), (14, 16),   # Right leg
]

# Map COCO index to position in the 13-keypoint array
COCO_TO_POS = {idx: i for i, idx in enumerate(SHARED_COCO_INDICES)}

# Colours
COLOR_P2S = "tab:blue"
COLOR_RTM = "tab:orange"


def plot_skeleton(ax, kpts, label, show_labels=True, color="tab:blue"):
    """Plot a single skeleton on a 3D axis."""
    ax.scatter(kpts[:, 0], kpts[:, 1], kpts[:, 2],
               s=30, color=color, depthshade=True, label=label)

    if show_labels:
        for i, coco_idx in enumerate(SHARED_COCO_INDICES):
            name = KEYPOINT_MAPPING[coco_idx]
            ax.text(kpts[i, 0], kpts[i, 1], kpts[i, 2], f" {name}",
                    fontsize=6, color=color)

    for a, b in SKELETON:
        if a in COCO_TO_POS and b in COCO_TO_POS:
            pa, pb = COCO_TO_POS[a], COCO_TO_POS[b]
            ax.plot([kpts[pa, 0], kpts[pb, 0]],
                    [kpts[pa, 1], kpts[pb, 1]],
                    [kpts[pa, 2], kpts[pb, 2]],
                    c=color, linewidth=1.5)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Pose2Sim vs RTMPose3D 3D skeletons"
    )
    parser.add_argument("--pose2sim-csv", required=True,
                        help="CSV from pose2sim_to_camera.py")
    parser.add_argument("--rtmpose3d-csv", required=True,
                        help="CSV from rtmpose3d_extract.py")
    parser.add_argument("--frame", type=int, default=None,
                        help="Frame number to plot (default: first common frame)")
    parser.add_argument("--no-labels", action="store_true",
                        help="Hide keypoint name labels")
    args = parser.parse_args()

    # Load both CSVs
    p2s_data = load_keypoints_csv(args.pose2sim_csv)
    rtm_data = load_keypoints_csv(args.rtmpose3d_csv)

    common_frames = sorted(set(p2s_data.keys()) & set(rtm_data.keys()))
    if not common_frames:
        print("Error: no common frames between the two CSVs.")
        return

    frame = args.frame if args.frame is not None else common_frames[0]
    if frame not in p2s_data or frame not in rtm_data:
        print(f"Error: frame {frame} not found in both CSVs. "
              f"Available: {common_frames[0]}-{common_frames[-1]}")
        return

    # Procrustes-align RTMPose3D onto Pose2Sim (both centred at hip midpoint)
    discrepancies, scale, rotation, rtm_aligned, p2s_centered = align_and_compare(
        p2s_data[frame], rtm_data[frame])

    print(f"Frame {frame}: scale={scale:.4f}, "
          f"mean discrepancy={discrepancies.mean():.4f} m")

    show_labels = not args.no_labels

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    plot_skeleton(ax, p2s_centered, "Pose2Sim", show_labels=show_labels,
                  color=COLOR_P2S)
    plot_skeleton(ax, rtm_aligned, "RTMPose3D (aligned)", show_labels=show_labels,
                  color=COLOR_RTM)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"Pose2Sim vs RTMPose3D (Procrustes) — Frame {frame}")
    ax.legend()

    # Equal aspect ratio based on combined data
    all_kpts = np.concatenate([p2s_centered, rtm_aligned], axis=0)
    mid = all_kpts.mean(axis=0)
    max_range = (all_kpts.max(axis=0) - all_kpts.min(axis=0)).max() / 2
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
