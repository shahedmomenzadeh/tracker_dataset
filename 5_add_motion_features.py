import numpy as np
import json
import os
import glob
import argparse

# --- Configuration ---
DATASET_ROOT = "dataset/"

# These classes are instruments and will have motion features calculated.
INSTRUMENT_CLASSES = {
    "Cannula", "Cap-Cystotome", "Cap-Forceps", "Forceps", "IA-Handpiece",
    "Lens-Injector", "Phaco-Handpiece", "Primary-Knife", "Second-Instrument",
    "Secondary-Knife"
}

# --- Helper Functions ---

def _calculate_kinematics(position_track):
    """
    Calculates velocity, acceleration, and jerk from a trajectory of 2D points.

    Args:
        position_track (list): A list of 2D points [x, y] or None.

    Returns:
        A tuple of three lists: (velocities, accelerations, jerks).
    """
    num_frames = len(position_track)
    velocities = [None] * num_frames
    accelerations = [None] * num_frames
    jerks = [None] * num_frames

    # Calculate Velocities (pixels/frame)
    for i in range(1, num_frames):
        p1 = position_track[i-1]
        p2 = position_track[i]
        if p1 is not None and p2 is not None:
            velocities[i] = [p2[0] - p1[0], p2[1] - p1[1]]

    # Calculate Accelerations (pixels/frame^2)
    for i in range(1, num_frames):
        v1 = velocities[i-1]
        v2 = velocities[i]
        if v1 is not None and v2 is not None:
            accelerations[i] = [v2[0] - v1[0], v2[1] - v1[1]]

    # Calculate Jerks (pixels/frame^3)
    for i in range(1, num_frames):
        a1 = accelerations[i-1]
        a2 = accelerations[i]
        if a1 is not None and a2 is not None:
            jerks[i] = [a2[0] - a1[0], a2[1] - a1[1]]
            
    return velocities, accelerations, jerks

# --- Main Processing Function ---

def process_video_annotations(data):
    """
    Adds motion features to all instrument annotations in the dataset.
    """
    category_map = {cat['id']: cat for cat in data['categories']}
    num_frames = len(data['videos'][0]['file_names'])
    
    # 1. Find the Pupil's center trajectory first. This is our reference.
    pupil_center_track = [None] * num_frames
    for ann in data['annotations']:
        if category_map.get(ann['category_id'], {}).get('name') == "Pupil":
            num_keypoints = len(category_map[ann['category_id']]['keypoints'])
            for i in range(num_frames):
                kp_base_idx = i * num_keypoints * 3
                # Center is the first keypoint
                center_data = ann['keypoints'][kp_base_idx : kp_base_idx + 3]
                if center_data[2] == 2: # If center is visible
                    pupil_center_track[i] = [center_data[0], center_data[1]]
            break # Assume only one pupil annotation

    # 2. Iterate through annotations and process instruments
    for ann in data['annotations']:
        class_name = category_map.get(ann['category_id'], {}).get('name')
        
        if class_name in INSTRUMENT_CLASSES:
            print(f"  Calculating motion features for '{class_name}' (ID: {ann['id']}).")
            num_keypoints = len(category_map[ann['category_id']]['keypoints'])
            
            # a. Extract the instrument's absolute tip trajectory
            tip_track_abs = [None] * num_frames
            for i in range(num_frames):
                kp_base_idx = i * num_keypoints * 3
                # Tip is the second keypoint
                tip_data = ann['keypoints'][kp_base_idx + 3 : kp_base_idx + 6]
                if tip_data[2] == 2: # If tip is visible
                    tip_track_abs[i] = [tip_data[0], tip_data[1]]

            # b. Calculate absolute kinematics
            vel_abs, acc_abs, jerk_abs = _calculate_kinematics(tip_track_abs)

            # c. Calculate the relative position trajectory
            tip_track_rel = [None] * num_frames
            for i in range(num_frames):
                if tip_track_abs[i] is not None and pupil_center_track[i] is not None:
                    tip_track_rel[i] = [
                        tip_track_abs[i][0] - pupil_center_track[i][0],
                        tip_track_abs[i][1] - pupil_center_track[i][1]
                    ]

            # d. Calculate relative kinematics
            vel_rel, acc_rel, jerk_rel = _calculate_kinematics(tip_track_rel)

            # e. Add the new "motion_features" object to the annotation
            ann['motion_features'] = {
                "absolute": {
                    "velocity": vel_abs,
                    "acceleration": acc_abs,
                    "jerk": jerk_abs
                },
                "relative_to_pupil": {
                    "position": tip_track_rel,
                    "velocity": vel_rel,
                    "acceleration": acc_rel,
                    "jerk": jerk_rel
                }
            }
            
    return data

# --- Main Execution Block ---

def main():
    parser = argparse.ArgumentParser(
        description="Add kinematic motion features to a surgical video dataset.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--dataset_dir", type=str, default=DATASET_ROOT,
        help="Path to the root directory of the dataset to process."
    )
    parser.add_argument(
        "--video", nargs='+',
        help="Specify one or more video names (e.g., '0020') to process. If not provided, all videos are processed."
    )
    args = parser.parse_args()

    # --- Setup ---
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

    # --- Processing Loop ---
    for video_name in video_names:
        video_folder_path = os.path.join(args.dataset_dir, video_name)
        print(f"\n--- Processing video: {video_name} ---")
        
        # Define file paths for the current workflow
        input_path = os.path.join(video_folder_path, "annotation_smooth.json")
        output_path = os.path.join(video_folder_path, "annotation_full.json")
        
        if not os.path.exists(input_path):
            print(f"  [Warning] Input file '{os.path.basename(input_path)}' not found. Skipping.")
            continue
        
        with open(input_path, 'r') as f:
            data = json.load(f)

        # Process the data to add motion features
        enriched_data = process_video_annotations(data)

        # Save the new, fully-featured annotation file
        with open(output_path, 'w') as f:
            json.dump(enriched_data, f, indent=4)
        print(f"  Saved final annotations with motion features to {output_path}")
        
        print(f"--- Finished processing {video_name} ---")

if __name__ == "__main__":
    main()
