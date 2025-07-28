# create_annotated_video.py Documentation

This document provides details about the `create_annotated_video.py` script.

## What does this code do?

This script creates a new video file with annotations drawn directly onto the frames. It reads an original video and its corresponding JSON annotation file, then overlays bounding boxes and class labels for detected objects on each frame. This is a visualization tool that helps in verifying the accuracy of the object detection and tracking process.

## What is the input of this code?

1.  **Original Video**: The source video file that was used to generate the annotations. The script needs this to get the original frames.
2.  **JSON Annotation File**: The corresponding JSON file from the `dataset/` directory that contains the bounding box and class information for each frame.

## What is the output of this code?

The script produces a new video file (in `.avi` format) that is a copy of the original video but with the annotations visually rendered on it. The output videos are saved in the `visualized_videos/` directory.

## What are the configurations?

The script's behavior can be customized through the following configuration variables found at the beginning of the file:

-   `DATASET_ROOT`: The root directory where the dataset's JSON files are stored. Default is `"dataset/"`.
-   `VIDEO_OUTPUT_DIR`: The directory where the output annotated videos will be saved. Default is `"visualized_videos/"`.
-   `COLOR_DICT`: A dictionary that maps class names to specific colors (in BGR format). This ensures that each object class is consistently represented by the same color in the output video.

## The algorithm that is implemented in the code

1.  **Initialization**: The `main` function specifies the video folder and the corresponding JSON file to be processed.
2.  **Video and Data Loading**:
    -   The script loads the specified video file using OpenCV.
    -   It loads the annotation data from the specified JSON file.
3.  **Video Writer Setup**: An OpenCV `VideoWriter` object is created. This object is configured with the properties of the original video (frame width, height, and FPS) and is set to save the output video in the `VIDEO_OUTPUT_DIR`.
4.  **Frame-by-Frame Annotation**:
    -   The script iterates through each frame of the input video.
    -   For each frame, it finds the corresponding annotation data from the loaded JSON structure based on the frame number.
    -   The `draw_annotations_on_frame` function is called. This function iterates through all the objects detected in that frame and draws a colored bounding box and a text label for each one.
5.  **Writing to Output Video**: The modified frame (with annotations) is written to the output video file using the `VideoWriter`.
6.  **Cleanup**: After processing all frames, the script releases the video capture and writer objects.
7.  **Reporting**: The script prints a message indicating that the annotated video has been created successfully.

## How to use the code

To use this script, you need to:
1.  Make sure you have a video file and its corresponding JSON annotation file.
2.  In the `main` function of the script, set the `video_folder_path` variable to the path of your video and the `json_filename` to the name of your annotation file.
3.  Run the script. The output video will be generated in the `visualized_videos/` directory.
