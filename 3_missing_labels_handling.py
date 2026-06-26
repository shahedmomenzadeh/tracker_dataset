import cv2
import numpy as np
import json
import os
import glob
import math

# --------------------------------------------------------------------------
# ✏️ 1. CONFIGURATION
#    Modify the variables in this section to match your needs.
# --------------------------------------------------------------------------

# Path to the root directory of the dataset.
DATASET_ROOT = "dataset/"

# The maximum number of consecutive missing frames to interpolate.
MAX_GAP_SIZE = 10

# A list of specific video folder names to process.
# LEAVE EMPTY (e.g., []) to process ALL video folders found in DATASET_ROOT.
# Example: VIDEOS_TO_PROCESS = ["0020", "0481"]
VIDEOS_TO_PROCESS = [] #<-- CHANGE THIS

# The filename of the input annotation file (the one to read from).
INPUT_ANNOTATION_FILENAME = "annotation_cleaned.json"

# The filename for the output annotation file (the one that will be created).
OUTPUT_ANNOTATION_FILENAME = "annotation_miss_handled.json"


# --------------------------------------------------------------------------
# ⚙️ 2. CORE LOGIC
#    You don't need to change the code below this line.
# --------------------------------------------------------------------------

# --- Helper Functions ---

def ease_in_out_sine(t):
    """A non-linear easing function for smoother interpolation."""
    return -(math.cos(math.pi * t) - 1) / 2

def interpolate_bbox(bbox_start, bbox_end, t):
    """Interpolates bounding box [x, y, w, h] using an easing function."""
    t_eased = ease_in_out_sine(t)
    return [
        int(bbox_start[0] * (1 - t_eased) + bbox_end[0] * t_eased),
        int(bbox_start[1] * (1 - t_eased) + bbox_end[1] * t_eased),
        int(bbox_start[2] * (1 - t_eased) + bbox_end[2] * t_eased),
        int(bbox_start[3] * (1 - t_eased) + bbox_end[3] * t_eased),
    ]

def interpolate_keypoints(kp_start, kp_end, t):
    """Interpolates keypoints [x, y, v] using an easing function."""
    t_eased = ease_in_out_sine(t)
    # Only interpolate if both points are visible (v=2)
    if kp_start[2] == 2 and kp_end[2] == 2:
        return [
            int(kp_start[0] * (1 - t_eased) + kp_end[0] * t_eased),
            int(kp_start[1] * (1 - t_eased) + kp_end[1] * t_eased),
            2 # Mark as visible
        ]
    return [0, 0, 0] # Return non-visible if start or end is not visible

def generate_interpolated_mask(seg_start, bbox_start, bbox_interpolated):
    """
    Generates a new segmentation mask by resizing a template mask.
    """
    if not seg_start or not seg_start[0]:
        return None

    x_s, y_s, w_s, h_s = bbox_start
    if w_s <= 0 or h_s <= 0: return None

    template_mask = np.zeros((h_s, w_s), dtype=np.uint8)
    poly_start = np.array(seg_start[0], dtype=np.int32).reshape((-1, 1, 2))
    poly_start[:, :, 0] -= x_s
    poly_start[:, :, 1] -= y_s
    cv2.fillPoly(template_mask, [poly_start], 255)

    x_i, y_i, w_i, h_i = bbox_interpolated
    if w_i <= 0 or h_i <= 0: return None

    resized_mask = cv2.resize(template_mask, (w_i, h_i), interpolation=cv2.INTER_NEAREST)

    contours, _ = cv2.findContours(resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None

    main_contour = contours[0]
    main_contour[:, :, 0] += x_i
    main_contour[:, :, 1] += y_i

    return [main_contour.flatten().tolist()]


# --- Main Processing Function ---

def process_annotations(data, max_gap_size):
    """
    Finds and fills missing label gaps in the annotation data.
    """
    total_gaps_filled = 0
    for ann in data['annotations']:
        num_frames = len(ann['segmentations'])
        idx = 0
        while idx < num_frames:
            # Find the start of a potential gap
            if ann['segmentations'][idx] is None and idx > 0 and ann['segmentations'][idx-1] is not None:
                start_gap_idx = idx

                # Find the end of the gap
                end_gap_idx = -1
                for j in range(start_gap_idx, num_frames):
                    if ann['segmentations'][j] is not None:
                        end_gap_idx = j
                        break

                # If a valid gap is found within the threshold, process it
                if end_gap_idx != -1:
                    gap_size = end_gap_idx - start_gap_idx
                    if 0 < gap_size <= max_gap_size:
                        # print(f"  Found gap of size {gap_size} for annotation ID {ann['id']} from frame {start_gap_idx} to {end_gap_idx-1}. Interpolating...")
                        total_gaps_filled += gap_size

                        bbox_start = ann['bboxes'][start_gap_idx - 1]
                        bbox_end = ann['bboxes'][end_gap_idx]
                        seg_start = ann['segmentations'][start_gap_idx - 1]

                        # Find category to determine keypoint stride
                        category = next((cat for cat in data['categories'] if cat['id'] == ann['category_id']), None)
                        if not category or 'keypoints' not in category: continue

                        num_keypoints = len(category['keypoints'])
                        kp_stride = num_keypoints * 3

                        kp_list_start = ann['keypoints'][(start_gap_idx - 1) * kp_stride : start_gap_idx * kp_stride]
                        kp_list_end = ann['keypoints'][end_gap_idx * kp_stride : (end_gap_idx + 1) * kp_stride]

                        for i in range(gap_size):
                            frame_idx = start_gap_idx + i
                            t = (i + 1) / (gap_size + 1.0)

                            inter_bbox = interpolate_bbox(bbox_start, bbox_end, t)
                            ann['bboxes'][frame_idx] = inter_bbox

                            inter_seg = generate_interpolated_mask(seg_start, bbox_start, inter_bbox)
                            ann['segmentations'][frame_idx] = inter_seg

                            if inter_seg and inter_seg[0]:
                                contour = np.array(inter_seg[0]).reshape(-1, 2)
                                ann['areas'][frame_idx] = int(cv2.contourArea(contour))
                            else:
                                ann['areas'][frame_idx] = 0

                            inter_kp_list = []
                            for kp_idx in range(num_keypoints):
                                kp_start = kp_list_start[kp_idx*3 : (kp_idx+1)*3]
                                kp_end = kp_list_end[kp_idx*3 : (kp_idx+1)*3]
                                inter_kp = interpolate_keypoints(kp_start, kp_end, t)
                                inter_kp_list.extend(inter_kp)

                            start_kp_json_idx = frame_idx * kp_stride
                            ann['keypoints'][start_kp_json_idx : start_kp_json_idx + kp_stride] = inter_kp_list

                    idx = end_gap_idx
                else:
                    # No end to the gap was found, stop searching for this annotation
                    idx = num_frames
            else:
                idx += 1

    print(f"Total missing labels filled: {total_gaps_filled}")
    return data


# --------------------------------------------------------------------------
# ▶️ 3. EXECUTION
#    This block runs the script using the configuration above.
# --------------------------------------------------------------------------

def run_interpolation_script():
    if not os.path.exists(DATASET_ROOT):
        print(f"❌ [Error] Input dataset directory not found at '{DATASET_ROOT}'")
        return

    # If no specific videos are listed, find all subdirectories in the dataset folder
    if not VIDEOS_TO_PROCESS:
        print(f"No specific video provided. Processing all videos in '{DATASET_ROOT}'...")
        video_names = [os.path.basename(d) for d in glob.glob(os.path.join(DATASET_ROOT, '*')) if os.path.isdir(d)]
    else:
        video_names = VIDEOS_TO_PROCESS

    if not video_names:
        print(f"❌ No video subdirectories found in '{DATASET_ROOT}'")
        return

    print(f"Found {len(video_names)} video(s) to process: {sorted(video_names)}")

    for video_name in sorted(video_names):
        video_folder_path = os.path.join(DATASET_ROOT, video_name)
        print(f"\n--- Processing video: {video_name} ---")

        # Define file paths
        input_annotation_path = os.path.join(video_folder_path, INPUT_ANNOTATION_FILENAME)
        output_annotation_path = os.path.join(video_folder_path, OUTPUT_ANNOTATION_FILENAME)

        if not os.path.exists(input_annotation_path):
            print(f"  [Warning] Input file '{INPUT_ANNOTATION_FILENAME}' not found for {video_name}. Skipping.")
            continue

        with open(input_annotation_path, 'r') as f:
            cleaned_data = json.load(f)

        # Process the data to fill gaps
        handled_data = process_annotations(cleaned_data, MAX_GAP_SIZE)

        # Save the new annotation file
        with open(output_annotation_path, 'w') as f:
            json.dump(handled_data, f, indent=4)
        print(f"  ✅ Saved new annotations to {output_annotation_path}")

        print(f"--- Finished processing {video_name} ---")

    print("\n🎉 All done!")

