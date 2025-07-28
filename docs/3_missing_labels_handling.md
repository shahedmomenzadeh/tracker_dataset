# 3_missing_labels_handling.py Documentation

This document provides details about the `3_missing_labels_handling.py` script.

## What does this code do?

This script is designed to address the issue of missing labels in the annotation data. In object tracking, it's common for a detector to fail to identify an object in a few frames, leading to gaps in the trajectory data. This script fills in these missing annotations using an interpolation technique, ensuring that each tracked object has a continuous presence across the frames it appears in.

## What is the input of this code?

The script takes the cleaned JSON annotation files from the `dataset/` directory as input. These files contain object trajectories that may have missing frames.

## What is the output of this code?

The script creates a new JSON file named `annotation_miss_handled.json` in the video's subdirectory. It does not modify the `annotation_cleaned.json` file.

## What are the configurations?

This script does not have any user-configurable parameters at the top of the file. The logic for interpolation is self-contained.

## The algorithm that is implemented in the code

1.  **Argument Parsing**: The script takes video names as command-line arguments.
2.  **File Discovery**: It looks for `annotation_cleaned.json` in each specified video's directory.
3.  **Data Loading and Grouping**: For each JSON file:
    -   The annotation data is loaded.
    -   The annotations are grouped by object `id`. This creates a separate trajectory for each tracked object.
4.  **Trajectory Processing**: For each object's trajectory:
    -   The frames are sorted chronologically.
    -   The script identifies gaps in the frame sequence. A gap is a sequence of frame numbers where the object was not detected.
5.  **Linear Interpolation**: For each identified gap:
    -   The script uses the bounding box (`bbox`) of the object from the frame just *before* the gap and the frame just *after* the gap.
    -   It performs linear interpolation on the coordinates of the bounding box (`x1`, `y1`, `x2`, `y2`) for each missing frame within the gap.
    -   A new annotation entry is created for each missing frame with the interpolated bounding box. The `class` and `id` are carried over from the known frames.
6.  **Updating Annotations**: The newly created interpolated annotations are added to the main `annotations` list in the JSON data structure.
7.  **Saving the File**: The updated JSON data, now with filled gaps, is saved to a new file, `annotation_miss_handled.json`.
8.  **Reporting**: The script prints the number of interpolated annotations added for each file.

## How is the structure of the dataset after running each code?

A new file, `annotation_miss_handled.json`, is created. Its structure is identical to the input file, but the `annotations` list will now contain more entries, as the gaps in object trajectories have been filled with interpolated data.

Example of an `annotations` list *before* running the script:
```json
"annotations": [
  {"frame": 10, "id": 5, "class": "Cannula", "bbox": [100, 100, 120, 120]},
  // Missing annotation for frame 11 for id 5
  {"frame": 12, "id": 5, "class": "Cannula", "bbox": [104, 104, 124, 124]}
]
```

Example of the `annotations` list *after* running the script:
```json
"annotations": [
  {"frame": 10, "id": 5, "class": "Cannula", "bbox": [100, 100, 120, 120]},
  {"frame": 11, "id": 5, "class": "Cannula", "bbox": [102, 102, 122, 122]}, // Interpolated
  {"frame": 12, "id": 5, "class": "Cannula", "bbox": [104, 104, 124, 124]}
]
```
