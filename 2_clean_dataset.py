import json
import os
import argparse

# --- Configuration ---
DATASET_ROOT = "dataset/"

# These classes are always preserved, regardless of the allowed instruments list.
ALWAYS_KEPT_CLASSES = {"Cornea", "Pupil"}

# --- Core Cleaning Function ---

def clean_annotations(data, allowed_instruments):
    """
    Filters the annotations list in the dataset based on a set of allowed classes.

    Args:
        data (dict): The loaded JSON data from an annotation file.
        allowed_instruments (set): A set of instrument class names that are permitted.

    Returns:
        tuple: A tuple containing the modified data (dict) and the number of removed annotations.
    """
    # Combine user-specified instruments with the classes that are always kept.
    final_allowed_classes = set(allowed_instruments).union(ALWAYS_KEPT_CLASSES)
    
    # Create a mapping from category ID to category name for easy lookup.
    category_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}
    
    original_annotation_count = len(data['annotations'])
    cleaned_annotations = []
    
    # Iterate through each annotation instance in the file.
    for ann in data['annotations']:
        category_id = ann['category_id']
        class_name = category_id_to_name.get(category_id)
        
        # If the class name is in our final allowed list, keep the annotation.
        if class_name in final_allowed_classes:
            cleaned_annotations.append(ann)
            
    # Replace the old annotations list with the new, filtered one.
    data['annotations'] = cleaned_annotations
    removed_count = original_annotation_count - len(cleaned_annotations)
    
    return data, removed_count

# --- Main Execution Block ---

def main():
    """
    Parses arguments and runs the dataset cleaning process.
    Saves the output as a new 'annotation_cleaned.json' file.
    """
    parser = argparse.ArgumentParser(
        description="Clean a video dataset by removing annotations for non-allowed instruments.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--dataset_dir", type=str, default=DATASET_ROOT,
        help="Path to the root directory of the dataset to be cleaned (default: 'dataset/')."
    )
    parser.add_argument(
        "--videos", nargs='+', required=True,
        help="A space-separated list of video names (folder names) to process (e.g., 0020 0481)."
    )
    parser.add_argument(
        "--allowed_instruments", nargs='+', required=True,
        help="A space-separated list of instrument class names that are allowed in the specified videos."
    )
    args = parser.parse_args()

    # --- Setup ---
    if not os.path.exists(args.dataset_dir):
        print(f"[Error] Dataset directory not found at '{args.dataset_dir}'")
        return

    allowed_instruments_set = set(args.allowed_instruments)
    print(f"Cleaning specified videos to only contain these instruments: {sorted(list(allowed_instruments_set))}")
    print(f"(Note: '{', '.join(ALWAYS_KEPT_CLASSES)}' will always be kept)\n")

    # --- Processing Loop ---
    for video_name in args.videos:
        video_folder_path = os.path.join(args.dataset_dir, video_name)
        print(f"--- Processing video: {video_name} ---")

        if not os.path.isdir(video_folder_path):
            print(f"  [Warning] Video folder not found: {video_folder_path}. Skipping.")
            continue

        # 1. Define file paths
        original_annotation_path = os.path.join(video_folder_path, "annotation.json")
        cleaned_annotation_path = os.path.join(video_folder_path, "annotation_cleaned.json")

        if not os.path.exists(original_annotation_path):
            print(f"  [Warning] 'annotation.json' not found for {video_name}. Skipping.")
            continue

        # 2. Load the original annotation data
        with open(original_annotation_path, 'r') as f:
            data_to_clean = json.load(f)

        # 3. Clean the annotation data
        cleaned_data, removed_count = clean_annotations(data_to_clean, allowed_instruments_set)
        
        if removed_count > 0:
            print(f"  Removed {removed_count} annotations for non-allowed instruments.")
        else:
            print("  No non-allowed instruments found. Annotation file is already clean.")

        # 4. Save the new, cleaned data to annotation_cleaned.json
        with open(cleaned_annotation_path, 'w') as f:
            json.dump(cleaned_data, f, indent=4)
        print(f"  Saved cleaned annotations to '{cleaned_annotation_path}'")
        
        print(f"--- Finished cleaning {video_name} ---")

if __name__ == "__main__":
    main()
