# Surgical Instrument Tracker and Motion Analysis

This project provides a pipeline of Python scripts to process surgical videos, track instruments, and analyze their motion. The process starts from raw video files and produces a structured dataset with detailed annotations, including kinematic features like velocity, acceleration, and jerk.

## Getting Started

### Prerequisites

Before you begin, ensure you have Python installed on your system. This project is developed with Python 3.8+.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/shahedmomenzadeh/tracker_dataset
    cd tracker_dataset
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    A `requirements.txt` file is provided to install all necessary dependencies.
    ```bash
    pip install -r requirements.txt
    ```

## The Pipeline

The codebase is designed to be run sequentially. Each script takes the output of the previous one, processes it, and saves a new version of the annotation file.

### **1. Dataset Generation (`1_dataset_generator.py`)**

This is the first step. It takes raw videos and uses a YOLO object detection and segmentation model to identify and track surgical instruments in each frame.

*   **How to run:**
    1.  Place your surgical video files (e.g., in `.mp4` format) into the `videos/` directory.
    2.  Run the script from your terminal: `python 1_dataset_generator.py`
*   **Output:** Creates a folder for each video in `dataset/` containing `annotation.json`.

### **2. Cleaning the Dataset (`2_clean_dataset.py`)**

This script filters the generated annotations, keeping only the specified instruments.

*   **How to run:**
    1.  Run the script with arguments specifying the videos and allowed instruments. For example:
        `python 2_clean_dataset.py --videos 0020 0481 --allowed_instruments Phaco-Handpiece Primary-Knife`
*   **Output:** Creates `annotation_cleaned.json` in each processed video's folder.

### **3. Handling Missing Labels (`3_missing_labels_handling.py`)**

This script fills in gaps in the trajectories caused by detection failures.

*   **How to run:**
    1.  Run the script with the video names as arguments: `python 3_missing_labels_handling.py --video 0020 0481`
*   **Output:** Creates `annotation_miss_handled.json`.

### **4. Smoothing Trajectories (`4_smooth_trajectories.py`)**

This script smooths the raw trajectories to make them more realistic.

*   **How to run:**
    1.  Run the script, optionally specifying a video: `python 4_smooth_trajectories.py --video 0020`
*   **Output:** Creates `annotation_smooth.json`.

### **5. Adding Motion Features (`5_add_motion_features.py`)**

This final step calculates kinematic features (velocity, acceleration, jerk) for each instrument.

*   **How to run:**
    1.  Run the script: `python 5_add_motion_features.py --video 0020`
*   **Output:** Creates the final `annotation_full.json`.

## Dataset Information

### Classes

The dataset identifies 12 distinct classes, categorized as instruments and tissues:
-   **Instruments (10):** `Cannula`, `Cap-Cystotome`, `Cap-Forceps`, `Forceps`, `IA-Handpiece`, `Lens-Injector`, `Phaco-Handpiece`, `Primary-Knife`, `Second-Instrument`, `Secondary-Knife`
-   **Tissues (2):** `Cornea`, `Pupil`

### Class Mapping

An important feature of this pipeline is the ability to map different instrument classes to a single, unified class. This is handled in `1_dataset_generator.py` through the `CLASS_MAPPING` dictionary. This is useful for:
-   **Standardizing Instruments**: Treating different but functionally similar instruments as the same class.
-   **Correcting Model Confusion**: If the YOLO model frequently confuses two instruments, you can map them to the same class to improve consistency.

**Example Mappings:**
-   **For Phaco Videos**: You might map `Cannula` and `Cap-Cystotome` to `Second-Instrument`:
    ```python
    CLASS_MAPPING = {
        "Cannula": "Second-Instrument",
        "Cap-Cystotome": "Second-Instrument",
    }
    ```
-   **For Capsulorhexis Videos**: You might map `Cannula` and `Second-Instrument` to `Cap-Cystotome`:
    ```python
    CLASS_MAPPING = {
        "Cannula": "Cap-Cystotome",
        "Second-Instrument": "Cap-Cystotome",
    }
    ```
You can customize this dictionary in `1_dataset_generator.py` to suit the needs of your specific surgical videos.

### Final JSON Structure (`annotation_full.json`)

The `annotation_full.json` file is the final and most comprehensive output of the data processing pipeline. It describes all the annotations for a **single video**, containing the standard visual data (masks, boxes, keypoints) and the advanced, calculated motion features for surgical instruments.

#### Top-Level Structure

The file has five main keys at its root:

-   `"info"`: An object containing general metadata about the annotation file, such as a description, version, and year.  
    **Format**: `{"description": "...", "version": "...", ...}`

-   `"licenses"`: A list containing license information for the dataset.  
    **Format**: `[{"id": 1, "name": "ARAS", "url": ""}]`

-   `"categories"`: A list defining all possible object classes in the dataset. This list is crucial for mapping class IDs to names and is identical across all annotation files.  
    **Format**: `[{"id": 1, "name": "Cannula", ...}, ...]`

-   `"videos"`: A list containing a single object with metadata for the video this file describes.  
    **Format**: `[{"id": 1, "width": 1920, "height": 1080, "file_names": ["000001.jpg", ...]}]`

-   `"annotations"`: A list of objects, where each object represents a single, unique instance tracked through the entire video. This is the core data section.  
    **Format**: `[{...}, {...}, ...]`

#### The `categories` Section

This section defines the 12 classes used in the project, which are grouped into two types.

##### Anatomical Tissues (2 classes)

- `Cornea`
- `Pupil`

##### Surgical Instruments (10 classes)

- `Cannula`
- `Cap-Cystotome`
- `Cap-Forceps`
- `Forceps`
- `IA-Handpiece`
- `Lens-Injector`
- `Phaco-Handpiece`
- `Primary-Knife`
- `Second-Instrument`
- `Secondary-Knife`

Each category object has the following structure:

- `"id"`: A unique integer for the class (e.g., `11` for Pupil).
- `"name"`: The string name of the class (e.g., `"Pupil"`).
- `"keypoints"`: An array defining the names of the keypoints. For this project, it is always `["center", "tip"]`.
- `"skeleton"`: An empty list, as we are not defining connections between keypoints.

#### The `annotations` Section

This is a list where each object represents a tracked instance.

##### Standard Annotation Fields (for all classes)

- `"id"`: A unique integer ID for this specific object instance.
- `"video_id"`: Always `1`, linking it to the video entry.
- `"category_id"`: The integer ID linking to the object's class in the `"categories"` list.
- `"segmentations"`: A list of segmentation masks, one for each frame. The value is a polygon `[[x1, y1, x2, y2, ...]]` if the object is present, or `null` if it's not.
- `"bboxes"`: A list of bounding boxes, one for each frame. The value is `[x, y, width, height]` or `null`.
- `"areas"`: A list of the object's pixel area, one for each frame. The value is a float or `null`.
- `"keypoints"`: A flat list containing keypoint data for every frame. For each frame, it stores 6 values: `[center_x, center_y, center_v, tip_x, tip_y, tip_v]`.
  - `v` is a visibility flag: `2` means the keypoint is visible and labeled, `0` means it is not applicable or not visible.
  - For **Tissues**, only `center` is used: `[cx, cy, 2, 0, 0, 0]`
  - For **Instruments**, only `tip` is used: `[0, 0, 0, tx, ty, 2]`

##### `"motion_features"` Field (Instruments Only)

This key exists **only** within annotations for the 10 surgical instrument classes.

- `"absolute"`: An object containing kinematic data calculated from the tip's raw pixel coordinates.
  - `"velocity"`: List of `[vx, vy]` vectors (pixels/frame) or `null`.
  - `"acceleration"`: List of `[ax, ay]` vectors (pixels/frame²) or `null`.
  - `"jerk"`: List of `[jx, jy]` vectors (pixels/frame³) or `null`.

- `"relative_to_pupil"`: An object containing kinematics calculated from the tip's position relative to the pupil's center.
  - `"position"`: List of `[rx, ry]` vectors representing `tip_position - pupil_center`.
  - `"velocity"`: List of relative velocity vectors or `null`.
  - `"acceleration"`: List of relative acceleration vectors or `null`.
  - `"jerk"`: List of relative jerk vectors or `null`.

#### Example Snippet

This example shows the structure for a "Forceps" annotation (with motion features) and a "Pupil" annotation.

```json
"annotations": [
    {
        "id": 101,
        "video_id": 1,
        "category_id": 12,
        "segmentations": [ [[...]], null, ... ],
        "bboxes": [ [x,y,w,h], null, ... ],
        "areas": [ 1234.5, null, ... ],
        "keypoints": [ 500, 510, 2, 0, 0, 0,  null, ... ]
    },
    {
        "id": 102,
        "video_id": 1,
        "category_id": 4,
        "segmentations": [ [[...]], null, ... ],
        "bboxes": [ [x,y,w,h], null, ... ],
        "areas": [ 987.0, null, ... ],
        "keypoints": [ 0, 0, 0, 600, 450, 2,  null, ... ],
        "motion_features": {
            "absolute": {
                "velocity": [ null, [5, -2], ... ],
                "acceleration": [ null, null, [1, 0], ... ],
                "jerk": [ null, null, null, [0, 1], ... ]
            },
            "relative_to_pupil": {
                "position": [ [100, -60], [105, -62], ... ],
                "velocity": [ null, [5, -2], ... ],
                "acceleration": [ null, null, [1, 0], ... ],
                "jerk": [ null, null, null, [0, 1], ... ]
            }
        }
    }
]
```

## Visualization Tools

There are two scripts provided to help you visualize the results.

### **Creating an Annotated Video (`create_annotated_video.py`)**

This tool generates a video file with bounding boxes and class labels drawn on each frame.

*   **How to use:**
    1.  Open the `create_annotated_video.py` script.
    2.  In the `main()` function, set the `video_folder_path` to the path of the original video and `json_filename` to the path of the annotation file you want to visualize (e.g., `dataset/0020/annotation_full.json`).
    3.  Run the script: `python create_annotated_video.py`
    4.  The output video will be saved in the `visualized_videos/` directory.

### **Visualizing Motion Features (`visualize_motion_features.py`)**

This tool generates plots for the kinematic features and the 2D trajectory of the instruments.

*   **How to use:**
    1.  Open the `visualize_motion_features.py` script.
    2.  In the `main()` function, set the `video_name` variable and the `instruments_to_plot` list.
    3.  Run the script: `python visualize_motion_features.py`
    4.  The output plots will be saved in the `visualizations/` directory.
