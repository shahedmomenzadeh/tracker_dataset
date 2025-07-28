# 1_dataset_generator.py Documentation

This document provides details about the `1_dataset_generator.py` script.

## What does this code do?

This script processes video files to detect and track surgical instruments using a pre-trained YOLO (You Only Look Once) model. For each video, it generates a corresponding JSON file that contains detailed annotation data for every frame. This data includes the frame number, object ID, class label, and bounding box coordinates for each detected instrument.

## What is the input of this code?

The script takes video files as input. These video files should be placed in the `videos/` directory within the project structure.

## What is the output of this code?

The primary output is a set of JSON files, one for each input video, saved in the `dataset/` directory. Each JSON file contains the structured annotation data. Additionally, the script can be configured to save individual video frames with the detected bounding boxes and trajectories drawn on them.

## What are the configurations?

The script's behavior can be customized through the following configuration variables found at the beginning of the file:

-   `VIDEO_DIR`: The directory where the input video files are located. Default is `"videos/"`.
-   `DATASET_DIR`: The directory where the output JSON annotation files will be saved. Default is `"dataset/"`.
-   `MASKED_DATA_DIR`: The directory for storing masked data, if applicable. Default is `"masked_data/"`.
-   `MODEL_PATH`: The file path to the pre-trained YOLO model weights. Default is `'best.pt'`.
-   `CONF`: The confidence threshold for an object detection to be considered valid. Default is `0.5`.
-   `IOU_THRESHOLD`: The Intersection over Union (IoU) threshold used for tracking objects across frames. Default is `0.5`.
-   `CHANGE_THRESHOLD`: A threshold related to tracking logic. Default is `15`.
-   `EXCLUDED_CLASSES`: A list of class names that should be ignored from tracking. Default is `["Cornea", "Pupil"]` since these 2 classes are tissue classes.
-   `CLASS_MAPPING`: A dictionary to merge different class labels into a single category. This is a powerful feature for handling cases where different instruments are used for similar purposes or when the model confuses two similar-looking instruments.
    -   **Example for Phaco Videos**: The default mapping `{"Cannula": "Second-Instrument", "Cap-Cystotome": "Second-Instrument"}` is set up for phacoemulsification videos, where both the cannula and cap-cystotome can be treated as a "second instrument".
    -   **Example for Capsulorhexis Videos**: For a capsulorhexis video, you might want to map `Cannula` and `Second-Instrument` to `Cap-Cystotome`. You can easily change the mapping to: `{"Cannula": "Cap-Cystotome", "Second-Instrument": "Cap-Cystotome"}`.
    -   You can customize this mapping to fit the specific needs of your video analysis.
-   `SAVE_ANNOTATED_FRAMES`: A boolean flag that, when set to `True`, will save each frame as an image with annotations drawn on it. Default is `False`.
-   `DRAW_TRAJECTORY`: A boolean flag that, when set to `True`, will draw the movement trajectory of detected objects on the frames. Default is `True`.
-   `FINAL_CLASSES`: A list of all possible class labels that are expected in the final dataset.

## The algorithm that is implemented in the code

1.  **Initialization**: The script starts by loading the YOLO model from the specified `MODEL_PATH`.
2.  **Video Iteration**: It scans the `VIDEO_DIR` and processes each video file found.
3.  **Frame-by-Frame Processing**: For each video, the script reads it frame by frame.
4.  **Object Detection and Tracking**: On each frame, the YOLO model is used to detect objects. The `model.track()` function is used, which not only detects objects but also assigns a unique ID to each object to track it across consecutive frames.
5.  **Filtering and Mapping**: The detected objects are filtered based on the `CONF` threshold and the `EXCLUDED_CLASSES` list. The class labels are then re-mapped according to the `CLASS_MAPPING` dictionary.
6.  **Data Structuring**: The information for each valid detection (frame number, object ID, class name, and bounding box coordinates) is collected.
7.  **JSON Output**: After processing all frames of a video, the collected annotation data is saved as a JSON file in the `DATASET_DIR`. The filename of the JSON corresponds to the name of the video file.
8.  **Optional Frame Saving**: If `SAVE_ANNOTATED_FRAMES` is `True`, each processed frame with annotations is saved as an image file.

## How is the structure of the dataset after running each code?

After running `1_dataset_generator.py`, the `dataset/` directory will be populated with JSON files. Each JSON file has a structure similar to the following, representing the annotations for a single video:

```json
{
  "video_name": "example_video.mp4",
  "annotations": [
    {
      "frame": 0,
      "id": 1,
      "class": "Phaco-Handpiece",
      "bbox": [x1, y1, x2, y2]
    },
    {
      "frame": 1,
      "id": 1,
      "class": "Phaco-Handpiece",
      "bbox": [x1, y1, x2, y2]
    }
  ],
  "categories": [
    {"id": 0, "name": "Cannula"},
    {"id": 1, "name": "Cap-Cystotome"},
    ...
  ]
}
```
