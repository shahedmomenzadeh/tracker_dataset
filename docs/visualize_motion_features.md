# visualize_motion_features.py Documentation

This document provides details about the `visualize_motion_features.py` script.

## What does this code do?

This script is a visualization tool used to plot the kinematic features (velocity, acceleration, and jerk) and the 2D trajectory of surgical instruments. It reads the final, enriched JSON annotation file for a video and generates a set of plots for specified instruments. This helps in analyzing the motion patterns of the instruments during a surgical procedure.

## What is the input of this code?

The script takes a single, fully processed JSON annotation file from the `dataset/` directory as input. This JSON file must contain the motion features (velocity, acceleration, jerk) that were added by the `5_add_motion_features.py` script.

## What is the output of this code?

The script generates and saves a series of plots as image files (e.g., PNG). The plots are saved in the `visualizations/` directory. For each specified instrument, it can generate:
1.  A plot of velocity magnitude vs. frame number.
2.  A plot of acceleration magnitude vs. frame number.
3.  A plot of jerk magnitude vs. frame number.
4.  A plot of the 2D trajectory (path of the instrument in the video frame).

## What are the configurations?

The script's behavior can be customized through the following configuration variables found at the beginning of the file:

-   `DATASET_ROOT`: The root directory where the dataset's JSON files are stored. Default is `"dataset/"`.
-   `VISUALIZATION_OUTPUT_DIR`: The directory where the generated plot images will be saved. Default is `"visualizations/"`.

## The algorithm that is implemented in the code

1.  **Initialization**: The `main` function defines which video's data to process and which instruments to generate plots for.
2.  **Data Loading**: The script loads the specified JSON annotation file.
3.  **Data Grouping and Filtering**:
    -   It groups all annotations by object `id`.
    -   It filters these groups to only include the trajectories of the instruments specified in the `instruments_to_plot` list.
4.  **Plot Generation**: For each selected instrument trajectory:
    -   It creates a figure with multiple subplots using `matplotlib`.
    -   **Kinematic Data Extraction**: It extracts the velocity, acceleration, and jerk data for the instrument's trajectory. The `calculate_magnitude` helper function is used to convert the 2D vectors of these features into scalar magnitudes for plotting.
    -   **Plotting**: The `plot_kinematics` helper function is called for each feature (velocity, acceleration, jerk) to create a time-series plot on a separate subplot.
    -   **Trajectory Plotting**: The `plot_trajectory` helper function is called to plot the 2D path of the instrument on another subplot.
5.  **Saving the Plots**: The final figure containing all the plots for an instrument is saved as an image file in the `VISUALIZATION_OUTPUT_DIR`. The filename indicates the video and the instrument.
6.  **Reporting**: The script prints messages indicating which plots are being generated.

## How to use the code

1.  Ensure that you have run the entire data processing pipeline (`1_...` through `5_...`) for at least one video.
2.  In the `main` function of `visualize_motion_features.py`, set the `video_name` variable to the name of the video you want to analyze.
3.  In the `instruments_to_plot` list, specify the names of the instruments for which you want to generate plots.
4.  Run the script. The plot images will be saved in the `visualizations/` directory.
