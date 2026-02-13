"""Extract 3D keypoints from video using RTMPose3D."""

import argparse
import csv
import os
import re

import cv2
import numpy as np

from pose2sim_to_camera import (
    KEYPOINT_MAPPING,
    find_trc_file,
    load_trc,
)

# Indices into RTMPose3D's 133-keypoint output that correspond to our shared set
SHARED_COCO_INDICES = sorted(KEYPOINT_MAPPING.keys())  # [0, 5, 6, 7, ..., 16]


def find_video(videos_dir, camera_name):
    """Find the video file matching a camera name.

    Extracts a camera number pattern (e.g. 'cam01') from the calibration
    camera name (e.g. 'int_cam01_img') and matches it against video filenames
    (e.g. 'pi-cam-01_20251110_120255@30p.mp4').
    """
    # Extract cam number pattern: 'int_cam01_img' -> '01', 'int_cam02_img' -> '02'
    match = re.search(r"cam(\d+)", camera_name)
    if not match:
        raise ValueError(f"Cannot extract camera number from '{camera_name}'")
    cam_num = match.group(1)  # e.g. '01'

    # Match against video files: look for 'cam-01' or 'cam01' patterns
    video_exts = (".mp4", ".avi", ".mov", ".mkv")
    candidates = []
    for f in os.listdir(videos_dir):
        if not f.lower().endswith(video_exts):
            continue
        # Match 'cam-01', 'cam01', 'cam_01', 'cam-02' etc.
        if re.search(rf"cam[_-]?0*{int(cam_num)}(?!\d)", f, re.IGNORECASE):
            candidates.append(f)

    if len(candidates) == 0:
        available = [f for f in os.listdir(videos_dir) if f.lower().endswith(video_exts)]
        raise FileNotFoundError(
            f"No video matching camera '{camera_name}' (pattern: cam{cam_num}) "
            f"in {videos_dir}. Available: {available}"
        )
    if len(candidates) > 1:
        raise ValueError(f"Multiple videos match camera '{camera_name}': {candidates}")
    return os.path.join(videos_dir, candidates[0])


def extract_frames(video_path, frame_numbers):
    """Read specific frames from a video file.

    Args:
        video_path: Path to video file.
        frame_numbers: 1D array of frame numbers to extract.

    Returns:
        List of (frame_number, image_bgr) tuples.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {total_frames} frames, {fps:.1f} fps")

    max_frame = frame_numbers.max()
    if max_frame >= total_frames:
        print(f"Warning: requested frame {max_frame} but video has {total_frames} frames")

    results = []
    for frame_num in frame_numbers:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_num))
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: could not read frame {frame_num}, skipping")
            continue
        results.append((int(frame_num), frame))

    cap.release()
    return results


def run_rtmpose3d(model, frames_data):
    """Run RTMPose3D on extracted frames and return shared keypoints.

    Args:
        model: RTMPose3DInference instance.
        frames_data: List of (frame_number, image_bgr) tuples.

    Returns:
        frame_numbers: list of frame numbers with successful detections
        keypoints: ndarray of shape (N_frames, 13, 3) — the 13 shared keypoints
        scores: ndarray of shape (N_frames, 13) — confidence scores
    """
    frame_numbers = []
    keypoints_list = []
    scores_list = []

    for frame_num, image in frames_data:
        result = model(image, single_person=True)

        if result["keypoints_3d"].size == 0:
            print(f"Warning: no person detected in frame {frame_num}, skipping")
            continue

        # Shape [1, 133, 3] -> [133, 3], then select shared indices
        kpts_3d = result["keypoints_3d"][0]  # [133, 3]
        kpts_scores = result["scores"][0]  # [133]

        shared_kpts = kpts_3d[SHARED_COCO_INDICES]  # [13, 3]
        shared_scores = kpts_scores[SHARED_COCO_INDICES]  # [13]

        frame_numbers.append(frame_num)
        keypoints_list.append(shared_kpts)
        scores_list.append(shared_scores)

        print(f"  Frame {frame_num}: detected, mean score {shared_scores.mean():.3f}")

    if not keypoints_list:
        return [], np.array([]), np.array([])

    return frame_numbers, np.stack(keypoints_list), np.stack(scores_list)


def main():
    parser = argparse.ArgumentParser(
        description="Extract 3D keypoints from video using RTMPose3D"
    )
    parser.add_argument("--dir", required=True, help="Pose2Sim result directory")
    parser.add_argument("--camera", required=True, help="Camera name (e.g. int_cam01_img)")
    parser.add_argument("--num-frames", type=int, default=None,
                        help="Sample N frames evenly spaced across the full range")
    parser.add_argument("--device", default="cuda:0", help="Device (cuda:0, cpu)")
    parser.add_argument("--output", default=None, help="Output CSV path (default: auto)")
    parser.add_argument("--extract-only", action="store_true",
                        help="Only extract and save video frames (skip RTMPose3D inference)")
    args = parser.parse_args()

    pose3d_dir = os.path.join(args.dir, "pose-3d")
    videos_dir = os.path.join(args.dir, "videos")

    # Determine frame range from TRC file
    trc_path = find_trc_file(pose3d_dir)
    print(f"TRC file: {trc_path}")
    frames, times, _ = load_trc(trc_path)
    print(f"TRC frame range: {frames[0]}-{frames[-1]} ({len(frames)} frames)")

    # Subsample frames if requested
    if args.num_frames is not None and args.num_frames < len(frames):
        indices = np.linspace(0, len(frames) - 1, args.num_frames, dtype=int)
        frames = frames[indices]
        print(f"Sampled {len(frames)} frames: {frames.tolist()}")

    # Find video for this camera
    video_path = find_video(videos_dir, args.camera)
    print(f"Video: {video_path}")

    # Extract frames from video
    frames_data = extract_frames(video_path, frames)
    print(f"Extracted {len(frames_data)} frames from video")

    # If extract-only mode, save frames as images and exit
    if args.extract_only:
        out_dir = os.path.join(args.dir, "extracted_frames")
        os.makedirs(out_dir, exist_ok=True)
        for frame_num, image in frames_data:
            out_path = os.path.join(out_dir, f"{args.camera}_frame{frame_num:06d}.jpg")
            cv2.imwrite(out_path, image)
            print(f"  Saved {out_path} ({image.shape[1]}x{image.shape[0]})")
        print(f"Saved {len(frames_data)} frames to {out_dir}")
        return

    # Initialize RTMPose3D (lazy import so --extract-only works without the package)
    from rtmpose3d import RTMPose3DInference
    print(f"Loading RTMPose3D model on {args.device}...")
    model = RTMPose3DInference(device=args.device)
    print("Model loaded.")

    # Run inference
    print("Running inference...")
    detected_frames, keypoints_3d, scores = run_rtmpose3d(model, frames_data)
    print(f"Detected person in {len(detected_frames)}/{len(frames_data)} frames")

    if len(detected_frames) == 0:
        print("No detections — nothing to save.")
        return

    # Sanity check: Y values (depth) should be positive
    y_vals = keypoints_3d[..., 1]
    print(f"Depth (Y) range: [{y_vals.min():.3f}, {y_vals.max():.3f}] m")

    # Build keypoint info list (same order as SHARED_COCO_INDICES)
    keypoint_info = [(idx, KEYPOINT_MAPPING[idx]) for idx in SHARED_COCO_INDICES]

    # Save to CSV
    output_path = args.output or os.path.join(
        pose3d_dir,
        f"rtmpose3d_{args.camera}.csv",
    )
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "keypoint_index", "keypoint_name", "X", "Y", "Z"])
        for i, frame_num in enumerate(detected_frames):
            for j, (coco_idx, name) in enumerate(keypoint_info):
                x, y, z = keypoints_3d[i, j]
                writer.writerow([frame_num, coco_idx, name, f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"])

    print(f"Saved {len(detected_frames) * len(keypoint_info)} rows to {output_path}")


if __name__ == "__main__":
    main()
