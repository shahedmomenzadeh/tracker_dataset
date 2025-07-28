import json
import numpy as np
import os
import glob
import argparse
import matplotlib.pyplot as plt

# --- Configuration ---
DATASET_ROOT = "dataset/"
VISUALIZATION_OUTPUT_DIR = "visualizations/"

# --- Helper Functions ---

def calculate_magnitude(vectors):
    """Calculates the magnitude of a list of 2D vectors."""
    magnitudes = []
    for v in vectors:
        if v is not None and len(v) == 2:
            magnitudes.append(np.linalg.norm(v))
        else:
            magnitudes.append(np.nan) # Use NaN for missing data to create gaps in plots
    return magnitudes

def plot_kinematics(ax, data, title, color):
    """Plots a single kinematic feature (velocity, acceleration, or jerk) vs. time."""
    ax.plot(data, label=title, color=color, linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Magnitude (pixels/frame^n)")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

def plot_trajectory(ax, trajectory, title, color, alpha=1.0):
    """Plots a 2D trajectory."""
    # Filter out None values for plotting
    valid_points = np.array([p for p in trajectory if p is not None])
    if valid_points.size > 0:
        ax.plot(valid_points[:, 0], valid_points[:, 1], 'o-', label=title, color=color, markersize=2, linewidth=1, alpha=alpha)
    ax.set_title(title)
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_aspect('equal', adjustable='box')
    # Only invert Y axis for absolute trajectory to match image coordinates
    if "Absolute" in title:
        ax.invert_yaxis() 
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

# --- Main Processing Function ---

def generate_plots_for_video(data, video_name, instruments_to_plot, cleaned_data=None):
    """
    Generates and saves a grid of plots for each specified instrument.
    """
    category_map = {cat['id']: cat for cat in data['categories']}
    cleaned_category_map = {cat['id']: cat for cat in cleaned_data['categories']} if cleaned_data else None
    
    for ann in data['annotations']:
        category_id = ann['category_id']
        class_name = category_map.get(category_id, {}).get('name')
        
        if not class_name or class_name not in instruments_to_plot:
            continue
        
        if 'motion_features' not in ann:
            print(f"  -> Skipping '{class_name}': No 'motion_features' found in annotation.")
            continue
            
        print(f"  -> Generating plots for '{class_name}' (ID: {ann['id']})...")
        
        features = ann['motion_features']
        
        # Extract the smoothed absolute trajectory from the main keypoints list
        num_keypoints = len(category_map[category_id]['keypoints'])
        kp_stride = num_keypoints * 3
        absolute_trajectory = []
        for i in range(0, len(ann['keypoints']), kp_stride):
            tip_data = ann['keypoints'][i+3 : i+6]
            if tip_data[2] == 2:
                absolute_trajectory.append([tip_data[0], tip_data[1]])
            else:
                absolute_trajectory.append(None)

        # --- If comparison is enabled, find and extract the original trajectory ---
        original_absolute_trajectory = None
        if cleaned_data:
            original_ann = None
            for o_ann in cleaned_data['annotations']:
                o_class_name = cleaned_category_map.get(o_ann['category_id'], {}).get('name')
                if o_class_name == class_name:
                    original_ann = o_ann
                    break
            
            if original_ann:
                print(f"     Found corresponding 'cleaned' annotation for comparison.")
                o_num_keypoints = len(cleaned_category_map[original_ann['category_id']]['keypoints'])
                o_kp_stride = o_num_keypoints * 3
                original_absolute_trajectory = []
                for i in range(0, len(original_ann['keypoints']), o_kp_stride):
                    tip_data = original_ann['keypoints'][i+3 : i+6]
                    if tip_data[2] == 2:
                        original_absolute_trajectory.append([tip_data[0], tip_data[1]])
                    else:
                        original_absolute_trajectory.append(None)

        # --- Create a 4x2 plot grid ---
        fig, axes = plt.subplots(4, 2, figsize=(18, 24))
        fig.suptitle(f"Motion Analysis for '{class_name}'\nVideo: {video_name}", fontsize=20, y=0.96)

        # Row 1: Trajectories (Plotting both if comparison is enabled)
        abs_ax = axes[0, 0]
        plot_trajectory(abs_ax, absolute_trajectory, 'Smoothed Trajectory', 'royalblue')
        if original_absolute_trajectory:
            plot_trajectory(abs_ax, original_absolute_trajectory, 'Original (Cleaned)', 'red', alpha=0.7)
        abs_ax.set_title("Absolute Trajectory") # Reset title after potentially multiple plots
        abs_ax.legend()

        plot_trajectory(axes[0, 1], features['relative_to_pupil']['position'], 'Relative Trajectory (to Pupil)', 'seagreen')

        # Row 2: Velocity
        plot_kinematics(axes[1, 0], calculate_magnitude(features['absolute']['velocity']), 'Absolute Velocity', 'royalblue')
        plot_kinematics(axes[1, 1], calculate_magnitude(features['relative_to_pupil']['velocity']), 'Relative Velocity', 'seagreen')

        # Row 3: Acceleration
        plot_kinematics(axes[2, 0], calculate_magnitude(features['absolute']['acceleration']), 'Absolute Acceleration', 'royalblue')
        plot_kinematics(axes[2, 1], calculate_magnitude(features['relative_to_pupil']['acceleration']), 'Relative Acceleration', 'seagreen')

        # Row 4: Jerk
        plot_kinematics(axes[3, 0], calculate_magnitude(features['absolute']['jerk']), 'Absolute Jerk', 'royalblue')
        plot_kinematics(axes[3, 1], calculate_magnitude(features['relative_to_pupil']['jerk']), 'Relative Jerk', 'seagreen')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.94])
        
        output_filename = os.path.join(VISUALIZATION_OUTPUT_DIR, f"{video_name}_{class_name}_motion_analysis.png")
        plt.savefig(output_filename)
        print(f"     Saved plot to {output_filename}")
        plt.close(fig)

# --- Main Execution Block ---

def main():
    parser = argparse.ArgumentParser(
        description="Visualize motion features from an annotation_full.json file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--dataset_dir", type=str, default=DATASET_ROOT,
        help="Path to the root directory of the dataset."
    )
    parser.add_argument(
        "--video", nargs='+',
        help="Specify one or more video names (e.g., '0020'). If not provided, all videos are processed."
    )
    parser.add_argument(
        "--instruments", nargs='+',
        help="Specify instrument names to plot (e.g., 'Forceps' 'Cannula'). If not provided, all instruments are plotted."
    )
    parser.add_argument(
        "--compare_cleaned", action='store_true',
        help="If set, also plots the trajectory from 'annotation_cleaned.json' for comparison."
    )
    args = parser.parse_args()

    os.makedirs(VISUALIZATION_OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(args.dataset_dir):
        print(f"[Error] Input dataset directory not found at '{args.dataset_dir}'")
        return

    if args.video:
        video_names = args.video
    else:
        print("No specific video provided. Processing all videos...")
        video_names = [os.path.basename(d) for d in glob.glob(os.path.join(args.dataset_dir, '*')) if os.path.isdir(d)]

    if not video_names:
        print(f"No video subdirectories found in '{args.dataset_dir}'")
        return

    for video_name in video_names:
        video_folder_path = os.path.join(args.dataset_dir, video_name)
        print(f"\n--- Processing video: {video_name} ---")
        
        full_path = os.path.join(video_folder_path, "annotation_full.json")
        if not os.path.exists(full_path):
            print(f"  [Warning] Input file 'annotation_full.json' not found. Skipping.")
            continue
        
        with open(full_path, 'r') as f:
            data = json.load(f)

        cleaned_data_for_comparison = None
        if args.compare_cleaned:
            cleaned_path = os.path.join(video_folder_path, "annotation_cleaned.json")
            if os.path.exists(cleaned_path):
                print(f"  -> Loading 'annotation_cleaned.json' for comparison.")
                with open(cleaned_path, 'r') as f:
                    cleaned_data_for_comparison = json.load(f)
            else:
                print(f"  [Warning] Comparison file 'annotation_cleaned.json' not found. Comparison skipped.")

        all_instrument_names = {cat['name'] for cat in data['categories'] if cat['name'] != 'Pupil' and cat['name'] != 'Cornea'}
        if args.instruments:
            instruments_to_plot = set(args.instruments)
        else:
            instruments_to_plot = all_instrument_names

        generate_plots_for_video(data, video_name, instruments_to_plot, cleaned_data=cleaned_data_for_comparison)
        
        print(f"--- Finished plotting for {video_name} ---")

if __name__ == "__main__":
    main()
