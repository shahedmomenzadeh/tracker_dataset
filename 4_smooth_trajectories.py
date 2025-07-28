import json
import numpy as np
import os
import glob
import argparse
from collections import deque
from scipy.interpolate import CubicSpline

# --- Configuration ---
DATASET_ROOT = "dataset/"

# --- Smoothing Parameters ---
# The size of the sliding window used to check for local outliers.
WINDOW_SIZE = 30
# The number of standard deviations away from the median velocity for a point to be considered an outlier.
THRESHOLD_STD_DEV = 2.0
# These classes are instruments whose trajectories will be cleaned.
INSTRUMENT_CLASSES = {
    "Cannula", "Cap-Cystotome", "Cap-Forceps", "Forceps", "IA-Handpiece",
    "Lens-Injector", "Phaco-Handpiece", "Primary-Knife", "Second-Instrument",
    "Secondary-Knife"
}

# --- Core Smoothing Function ---

def smooth_trajectory_with_spline(keypoints, num_keypoints, window_size, threshold):
    """
    Detects outliers in a trajectory based on velocity and corrects them
    using Cubic Spline interpolation.

    Args:
        keypoints (list): The flat list of keypoints for an annotation.
        num_keypoints (int): The number of keypoints per frame (e.g., 2 for center and tip).
        window_size (int): The size of the sliding window for outlier detection.
        threshold (float): The standard deviation threshold to identify an outlier.

    Returns:
        list: The new, smoothed flat list of keypoints.
    """
    # 1. Extract tip trajectory and calculate frame-to-frame velocities
    tip_track, velocities = [], [0.0]
    for i in range(0, len(keypoints), num_keypoints * 3):
        # Tip is the second keypoint, its data starts at index 3
        tip_data = keypoints[i+3 : i+6]
        tip_track.append([tip_data[0], tip_data[1]] if tip_data[2] == 2 else None)
    
    for i in range(1, len(tip_track)):
        if tip_track[i] is not None and tip_track[i-1] is not None:
            velocities.append(np.linalg.norm(np.array(tip_track[i]) - np.array(tip_track[i-1])))
        else:
            velocities.append(0.0)

    # 2. Pass 1: Detect Outliers using a sliding window on velocity
    outlier_indices = set()
    window = deque(maxlen=window_size)
    for i, velocity in enumerate(velocities):
        window.append(velocity)
        if len(window) < window_size // 2: continue

        median_vel = np.median(window)
        std_dev_vel = np.std(window)
        # Set a minimum std deviation to handle flat-line velocity sections
        if std_dev_vel < 1.0: std_dev_vel = 1.0

        if velocity > median_vel + threshold * std_dev_vel:
            outlier_indices.add(i)

    if not outlier_indices:
        return keypoints # No changes needed

    print(f"    -> Detected {len(outlier_indices)} outliers. Applying spline correction.")

    # 3. Pass 2: Correct Outliers with Cubic Spline Interpolation
    # Gather all non-outlier points to build the spline
    good_indices, good_points = [], []
    for i, point in enumerate(tip_track):
        if i not in outlier_indices and point is not None:
            good_indices.append(i)
            good_points.append(point)

    # A cubic spline needs at least 4 points for good results
    if len(good_indices) < 4:
        print(f"    -> Warning: Not enough good points ({len(good_indices)}) for a reliable spline. Outliers will be removed (set to null).")
        for i in outlier_indices:
            tip_track[i] = None
    else:
        # Create splines for x and y coordinates
        spline_x = CubicSpline(good_indices, [p[0] for p in good_points])
        spline_y = CubicSpline(good_indices, [p[1] for p in good_points])
        # Use the splines to predict new positions for the outlier frames
        for i in outlier_indices:
            tip_track[i] = [spline_x(i), spline_y(i)]
            
    # 4. Reconstruct the flat keypoints list with the corrected data
    new_keypoints = list(keypoints)
    for i, point in enumerate(tip_track):
        idx = i * num_keypoints * 3
        if point:
            new_keypoints[idx + 3:idx + 6] = [int(point[0]), int(point[1]), 2]
        else:
            # If a point was an outlier and couldn't be interpolated, mark it as not visible
            new_keypoints[idx + 3:idx + 6] = [0, 0, 0]

    return new_keypoints


def process_annotations(data, window_size, threshold):
    """
    Main processing function that iterates through annotations and applies smoothing.
    """
    category_map = {cat['id']: cat for cat in data['categories']}

    for ann in data["annotations"]:
        class_name = category_map.get(ann['category_id'], {}).get('name')
        if class_name in INSTRUMENT_CLASSES:
            print(f"  Processing trajectory for '{class_name}' (ID: {ann['id']})...")
            num_kps = len(category_map[ann['category_id']]['keypoints'])
            if num_kps < 2: continue # Ensure 'tip' keypoint exists
            
            ann['keypoints'] = smooth_trajectory_with_spline(
                ann['keypoints'], num_kps, window_size, threshold
            )
    return data

# --- Main Execution Block ---

def main():
    parser = argparse.ArgumentParser(
        description="Smooth instrument trajectories by removing outliers with a Cubic Spline.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--dataset_dir", type=str, default=DATASET_ROOT,
        help="Path to the root directory of the dataset."
    )
    parser.add_argument(
        "--video", nargs='+',
        help="Specify one or more video names (e.g., '0020' '0481') to process. If not provided, all videos are processed."
    )
    parser.add_argument(
        "--window_size", type=int, default=WINDOW_SIZE,
        help=f"Size of the sliding window for outlier detection (default: {WINDOW_SIZE})."
    )
    parser.add_argument(
        "--threshold", type=float, default=THRESHOLD_STD_DEV,
        help=f"Standard deviation threshold to identify an outlier (default: {THRESHOLD_STD_DEV})."
    )
    args = parser.parse_args()

    if not os.path.exists(args.dataset_dir):
        print(f"[Error] Input dataset directory not found at '{args.dataset_dir}'")
        return

    if args.video:
        video_names = args.video
    else:
        print("No specific video provided. Processing all videos in dataset folder...")
        video_names = [os.path.basename(d) for d in glob.glob(os.path.join(args.dataset_dir, '*')) if os.path.isdir(d)]

    if not video_names:
        print(f"No video subdirectories found in '{args.dataset_dir}'")
        return

    for video_name in video_names:
        video_folder_path = os.path.join(args.dataset_dir, video_name)
        print(f"\n--- Processing video: {video_name} ---")
        
        # Define file paths for the current workflow
        input_path = os.path.join(video_folder_path, "annotation_miss_handled.json")
        output_path = os.path.join(video_folder_path, "annotation_smooth.json")
        
        if not os.path.exists(input_path):
            print(f"  [Warning] Input file '{os.path.basename(input_path)}' not found. Skipping.")
            continue
        
        with open(input_path, 'r') as f:
            data = json.load(f)

        # Run the smoothing process
        smoothed_data = process_annotations(data, args.window_size, args.threshold)

        # Save the new annotation file
        with open(output_path, 'w') as f:
            json.dump(smoothed_data, f, indent=4)
        print(f"  Saved smoothed annotations to {output_path}")
        
        print(f"--- Finished smoothing for {video_name} ---")

if __name__ == "__main__":
    main()
