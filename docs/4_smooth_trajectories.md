# 4_smooth_trajectories.py Documentation

This document provides details about the `4_smooth_trajectories.py` script.

## What does this code do?

This script is designed to smooth the trajectories of tracked objects in the dataset. The raw trajectories, even after filling missing labels, can be noisy or contain outlier points due to detection inaccuracies. This script applies a smoothing algorithm to the bounding box coordinates of instrument trajectories to make them more fluid and realistic.

## What is the input of this code?

The script takes the JSON annotation files from the `dataset/` directory as input. It is expected that these files have already been processed by the previous scripts (`1_dataset_generator.py`, `2_clean_dataset.py`, `3_missing_labels_handling.py`).

## What is the output of this code?

The script creates a new JSON file named `annotation_smooth.json` in the video's subdirectory.

## What are the configurations?

The script's behavior can be customized through the following configuration variables found at the beginning of the file:

-   `DATASET_ROOT`: The root directory where the dataset's JSON files are stored. Default is `"dataset/"`.
-   `WINDOW_SIZE`: The size of the sliding window used for outlier detection and smoothing. Default is `30`.
-   `THRESHOLD_STD_DEV`: The number of standard deviations from the median velocity that defines an outlier. Default is `2.0`.
-   `INSTRUMENT_CLASSES`: A set of class names that are considered instruments and whose trajectories should be smoothed.

## The algorithm that is implemented in the code

1.  **Argument Parsing**: The script can take a specific video name as a command-line argument.
2.  **File Discovery**: It looks for `annotation_miss_handled.json` in the video's directory.
3.  **Data Loading and Grouping**: The annotations are grouped by object `id` to form individual trajectories.
4.  **Instrument Filtering**: The script only processes trajectories for objects whose class is in the `INSTRUMENT_CLASSES` set.
5.  **Trajectory Smoothing**: The `smooth_trajectory_with_spline` function is applied to the trajectory of each instrument. This function performs the following steps:
    -   **Keypoint Extraction**: It extracts the center point of the bounding box for each frame in the trajectory.
    -   **Outlier Detection**: It uses a sliding window approach. For each point in the trajectory, it looks at the points within the `WINDOW_SIZE`. It calculates the velocity between consecutive points. If a point's velocity is a significant outlier (greater than `THRESHOLD_STD_DEV` standard deviations from the median velocity in the window), it is flagged.
    -   **Outlier Removal**: The flagged outlier points are removed from the trajectory.
    -   **Cubic Spline Interpolation**: A cubic spline is fitted to the remaining, non-outlier keypoints. This mathematical function creates a smooth curve that passes through the keypoints.
    -   **New Coordinate Generation**: The spline function is then used to generate a new, smoothed set of coordinates for *all* original frames in the trajectory (including where outliers were removed).
6.  **Annotation Update**: The `bbox` values in the original JSON data structure are updated with the new coordinates derived from the smoothed trajectory. The size of the bounding box is kept the same as the original; only its position is updated.
7.  **Saving the File**: The modified JSON data is saved to a new file, `annotation_smooth.json`.
8.  **Reporting**: The script prints which file is being processed and reports on the smoothing of each instrument's trajectory.

## How is the structure of the dataset after running each code?

A new file, `annotation_smooth.json`, is created. The structure of the JSON file remains the same, but the values of the `bbox` coordinates within the `annotations` list will now represent the smoothed path of the instrument.

The dataset consists of 12 classes:
- **Instruments**: "Cannula", "Cap-Cystotome", "Cap-Forceps", "Forceps", "IA-Handpiece", "Lens-Injector", "Phaco-Handpiece", "Primary-Knife", "Second-Instrument", "Secondary-Knife"
- **Tissues**: "Cornea", "Pupil"
