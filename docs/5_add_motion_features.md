# 5_add_motion_features.py Documentation

This document provides details about the `5_add_motion_features.py` script.

## What does this code do?

This script enriches the dataset by calculating and adding motion features for the tracked instruments. Specifically, it computes the velocity, acceleration, and jerk for each instrument's trajectory. These features can be very valuable for more advanced analysis of surgical activities, such as skill assessment or event detection.

## What is the input of this code?

The script takes the smoothed JSON annotation files from the `dataset/` directory as input. It is expected that these files have been processed by all the preceding scripts in the pipeline.

## What is the output of this code?

The script creates a new JSON file named `annotation_full.json` in the video's subdirectory. This is the final, fully-enriched annotation file.

## What are the configurations?

The script's behavior can be customized through the following configuration variables found at the beginning of the file:

-   `DATASET_ROOT`: The root directory where the dataset's JSON files are stored. Default is `"dataset/"`.
-   `INSTRUMENT_CLASSES`: A set of class names that are considered instruments and for which motion features will be calculated.

## The algorithm that is implemented in the code

1.  **Argument Parsing**: The script can take a specific video name as a command-line argument.
2.  **File Discovery**: It looks for `annotation_smooth.json` in the video's directory.
3.  **Data Loading and Grouping**: The annotations are grouped by object `id` to reconstruct the trajectory of each tracked object.
4.  **Instrument Filtering**: The script identifies which trajectories belong to instruments based on the `INSTRUMENT_CLASSES` set.
5.  **Kinematic Feature Calculation**: For each instrument trajectory, the `_calculate_kinematics` function is called.
    -   It takes the list of center points of the bounding boxes for the trajectory.
    -   **Velocity**: It calculates the velocity at each frame by finding the displacement vector between the current frame's position and the previous frame's position. The result is in pixels/frame.
    -   **Acceleration**: It calculates acceleration by finding the difference in velocity vectors between the current and previous frames. The result is in pixels/frame².
    -   **Jerk**: It calculates jerk by finding the difference in acceleration vectors between the current and previous frames. The result is in pixels/frame³.
6.  **Annotation Update**: The script iterates through the main `annotations` list in the JSON data. For each annotation corresponding to an instrument, it adds the calculated `velocity`, `acceleration`, and `jerk` for that specific frame.
7.  **Saving the File**: The modified JSON data, now including the motion features, is saved to a new file, `annotation_full.json`.
8.  **Reporting**: The script prints messages indicating which file is being processed and when the motion features have been successfully added.

## How is the structure of the dataset after running each code?

A new file, `annotation_full.json`, is created. The structure of the JSON file is modified. For each object in the `annotations` list that is an instrument, a new `motion_features` key is added, containing the calculated kinematic data.

The dataset consists of 12 classes:
- **Instruments**: "Cannula", "Cap-Cystotome", "Cap-Forceps", "Forceps", "IA-Handpiece", "Lens-Injector", "Phaco-Handpiece", "Primary-Knife", "Second-Instrument", "Secondary-Knife"
- **Tissues**: "Cornea", "Pupil"

Example of an instrument annotation in `annotation_full.json`:
```json
{
  "frame": 10,
  "id": 5,
  "class": "Cannula",
  "bbox": [101.5, 101.5, 121.5, 121.5],
  "motion_features": {
      "absolute": {
          "velocity": [1.5, 1.5],
          "acceleration": [0.1, 0.1],
          "jerk": [0.0, -0.05]
      },
      "relative_to_pupil": {
          "position": [50.0, 50.0],
          "velocity": [1.0, 1.0],
          "acceleration": [0.0, 0.0],
          "jerk": [-0.1, -0.05]
      }
  }
}
```
