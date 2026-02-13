"""Visualize 3D keypoints from a CSV file in an interactive matplotlib plot."""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from compare import load_keypoints_csv, SHARED_COCO_INDICES
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


def plot_skeleton(ax, kpts, frame_num, show_labels=True, color="tab:blue", alpha=1.0):
    """Plot a single skeleton on a 3D axis."""
    ax.scatter(kpts[:, 0], kpts[:, 1], kpts[:, 2],
               s=30, color=color, alpha=alpha, depthshade=True)

    if show_labels:
        for i, coco_idx in enumerate(SHARED_COCO_INDICES):
            name = KEYPOINT_MAPPING[coco_idx]
            ax.text(kpts[i, 0], kpts[i, 1], kpts[i, 2], f" {name}",
                    fontsize=6, alpha=alpha)

    for a, b in SKELETON:
        if a in COCO_TO_POS and b in COCO_TO_POS:
            pa, pb = COCO_TO_POS[a], COCO_TO_POS[b]
            ax.plot([kpts[pa, 0], kpts[pb, 0]],
                    [kpts[pa, 1], kpts[pb, 1]],
                    [kpts[pa, 2], kpts[pb, 2]],
                    c=color, alpha=alpha, linewidth=1.5)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize 3D keypoints from CSV in matplotlib"
    )
    parser.add_argument("csv", help="Path to keypoints CSV file")
    parser.add_argument("--num-frames", type=int, default=None,
                        help="Number of frames to plot (default: all)")
    parser.add_argument("--no-labels", action="store_true",
                        help="Hide keypoint name labels")
    args = parser.parse_args()

    data = load_keypoints_csv(args.csv)
    frames = sorted(data.keys())
    print(f"Loaded {len(frames)} frames")

    if args.num_frames is not None and args.num_frames < len(frames):
        indices = np.linspace(0, len(frames) - 1, args.num_frames, dtype=int)
        frames = [frames[i] for i in indices]
        print(f"Plotting {len(frames)} frames: {frames}")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Use colour gradient across frames
    cmap = plt.cm.viridis
    for i, frame in enumerate(frames):
        t = i / max(len(frames) - 1, 1)
        color = cmap(t)
        alpha = 0.3 + 0.7 * t  # earlier frames more transparent
        plot_skeleton(ax, data[frame], frame,
                      show_labels=(not args.no_labels and i == len(frames) - 1),
                      color=color, alpha=alpha)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"3D Keypoints — {len(frames)} frames")

    # Equal aspect ratio
    all_kpts = np.concatenate([data[f] for f in frames], axis=0)
    mid = all_kpts.mean(axis=0)
    max_range = (all_kpts.max(axis=0) - all_kpts.min(axis=0)).max() / 2
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
