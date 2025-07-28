import cv2
import numpy as np
import json
import os
import time
import glob
import shutil
import collections
from ultralytics import YOLO

# --- Custom JSON Encoder for NumPy types ---
class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# --- Configuration ---

# 1. Directory and Path Configuration
VIDEO_DIR = "videos/"
DATASET_DIR = "dataset/"
MASKED_DATA_DIR = "masked_data/"

# 2. Class and Model Configuration
MODEL_PATH = 'best.pt'
CONF = 0.5
IOU_THRESHOLD = 0.5
CHANGE_THRESHOLD = 15
EXCLUDED_CLASSES = ["Cornea", "Pupil"] 

# 3. Class Mapping for Similar Instruments
# For processing Phaco Videos -> maps Cannula and Cap-Cystotome to Second-Instrument
# or for Capsulrhexis Videos -> you can modify this mapping as needed, to map Cannula and Second-Instrument to Cap-Cystotome.
CLASS_MAPPING = {
    "Cannula": "Second-Instrument",
    "Cap-Cystotome": "Second-Instrument",
}

# 4. Visualization and Saving Options
SAVE_ANNOTATED_FRAMES = False
DRAW_TRAJECTORY = True

# Define the full list of final classes for the dataset.
FINAL_CLASSES = [
    "Cannula", "Cap-Cystotome", "Cap-Forceps", "Cornea", "Forceps",
    "IA-Handpiece", "Lens-Injector", "Phaco-Handpiece", "Primary-Knife",
    "Pupil", "Second-Instrument", "Secondary-Knife"
]

# Colors are in BGR format
COLOR_DICT = {
    "Cannula": (255, 0, 0), "Cap-Cystotome": (0, 255, 0), "Cap-Forceps": (0, 0, 255),
    "Cornea": (255, 255, 0), "Forceps": (255, 0, 255), "IA-Handpiece": (0, 255, 255),
    "Lens-Injector": (125, 125, 0), "Phaco-Handpiece": (0, 125, 125), "Primary-Knife": (125, 0, 125),
    "Pupil": (50, 200, 200), "Second-Instrument": (200, 200, 50), "Secondary-Knife": (200, 50, 200),
    "Default": (0, 0, 0)
}

# --- Initialization ---
for folder in [DATASET_DIR, MASKED_DATA_DIR]:
    os.makedirs(folder, exist_ok=True)
model = YOLO(MODEL_PATH)

# --- Utility and Core Functions ---

def remap_class(predicted_class):
    return CLASS_MAPPING.get(predicted_class, predicted_class)

def apply_mask(image, mask, color, alpha=0.5):
    colored_mask = np.zeros_like(image, dtype=np.uint8)
    colored_mask[mask] = color
    overlay = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
    image[mask] = overlay[mask]
    return image

def compute_tool_tip(mask, pupil_center):
    if pupil_center is None or not np.any(mask): return None
    points = np.argwhere(mask); points_xy = points[:, [1, 0]].astype(np.float32)
    if len(points_xy) < 2: return None
    vx, vy, x0, y0 = cv2.fitLine(points_xy, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    dir_vec, p0 = np.array([vx, vy]), np.array([x0, y0])
    if vx * vy <= 0:
        point1, point2 = [np.min(points_xy[:, 0]), np.max(points_xy[:, 1])], [np.max(points_xy[:, 0]), np.min(points_xy[:, 1])]
    else:
        point1, point2 = [np.min(points_xy[:, 0]), np.min(points_xy[:, 1])], [np.max(points_xy[:, 0]), np.max(points_xy[:, 1])]
    for point in [point1, point2]:
        t = np.dot(np.array(point) - p0, dir_vec) / np.dot(dir_vec, dir_vec)
        point[:] = p0 + t * dir_vec
    dist1 = np.linalg.norm(np.array(point1) - np.array(pupil_center))
    dist2 = np.linalg.norm(np.array(point2) - np.array(pupil_center))
    tip_projected = point1 if dist1 < dist2 else point2
    closest_idx = np.argmin(np.linalg.norm(points_xy - tip_projected, axis=1))
    return tuple(points_xy[closest_idx].astype(int))

# --- Tracking Classes ---

class TrackedObject:
    def __init__(self, obj_id, bbox, mask, p_class, change_threshold=CHANGE_THRESHOLD):
        self.id, self.bbox, self.mask = obj_id, np.array(bbox, dtype=np.float32), mask
        self.stable_class, self.pending_class, self.pending_count = p_class, None, 0
        self.change_threshold, self.tip_history = change_threshold, []

    def update(self, bbox, mask, p_class, tip=None):
        self.bbox, self.mask = np.array(bbox, dtype=np.float32), mask
        if p_class == self.stable_class: self.pending_class, self.pending_count = None, 0
        else:
            if self.pending_class == p_class: self.pending_count += 1
            else: self.pending_class, self.pending_count = p_class, 1
            if self.pending_count >= self.change_threshold:
                self.stable_class, self.pending_class, self.pending_count = p_class, None, 0
        if tip:
            self.tip_history.append(tip)
            if len(self.tip_history) > 50: self.tip_history.pop(0)

    def get_bbox(self): return tuple(self.bbox.astype(int))

class ObjectTracker:
    """
    A robust object tracker designed for the constraint that only one instance
    of any given class can exist at one time.
    """
    def __init__(self, change_threshold=CHANGE_THRESHOLD):
        self.tracked_objects = {}  # Use a dictionary with class_name as the key
        self.next_id = 0
        self.change_threshold = change_threshold

    def update(self, detections):
        """Updates tracked objects based on the new 'one-instance-per-class' logic."""
        
        # 1. Group all new detections by their class name
        detections_by_class = collections.defaultdict(list)
        for det in detections:
            # det = (bbox, mask, p_class, conf, tip)
            p_class = det[2]
            detections_by_class[p_class].append(det)

        updated_classes = set()

        # 2. For each class, find the best detection and update/create a tracker
        for p_class, class_detections in detections_by_class.items():
            # 2a. If multiple detections for a class, pick the one with the highest confidence
            best_detection = max(class_detections, key=lambda d: d[3])
            bbox, mask, _, _, tip = best_detection
            updated_classes.add(p_class)

            # 2b. Check if an object of this class is already being tracked
            if p_class in self.tracked_objects:
                # If yes, update it
                self.tracked_objects[p_class].update(bbox, mask, p_class, tip)
            else:
                # If no, create a new tracked object
                new_obj = TrackedObject(self.next_id, bbox, mask, p_class, self.change_threshold)
                self.tracked_objects[p_class] = new_obj
                self.next_id += 1

        # 3. Remove any tracked objects for classes that are no longer detected
        disappeared_classes = set(self.tracked_objects.keys()) - updated_classes
        for p_class in disappeared_classes:
            del self.tracked_objects[p_class]
            
        # Return the list of current tracked object values
        return list(self.tracked_objects.values())

# --- Dataset Generation Class ---

class DatasetGenerator:
    def __init__(self, video_name, frame_size, all_classes):
        self.video_name, (self.width, self.height) = video_name, frame_size
        self.categories = [{"id": i + 1, "name": name, "keypoints": ["center", "tip"], "skeleton": []}
                           for i, name in enumerate(all_classes)]
        self.class_to_id = {name: i + 1 for i, name in enumerate(all_classes)}
        self.dataset = {
            "info": {"description": f"Annotations for {video_name}", "version": "1.0", "year": time.strftime("%Y")},
            "licenses": [{"id": 1, "name": "ARAS", "url": ""}],
            "categories": self.categories, "videos": [], "annotations": []
        }

    def add_video_entry(self, frame_files):
        self.dataset["videos"].append({
            "id": 1, "width": self.width, "height": self.height, "file_names": frame_files})

    def add_annotation(self, instance_id, category_name, segmentations, bboxes, areas, keypoints):
        if category_name not in self.class_to_id: return
        self.dataset["annotations"].append({
            "id": instance_id, "video_id": 1, "category_id": self.class_to_id[category_name],
            "iscrowd": 0, "segmentations": segmentations, "bboxes": bboxes, "areas": areas,
            "keypoints": keypoints})

    def save_json(self, output_path):
        with open(output_path, 'w') as f:
            json.dump(self.dataset, f, indent=4, cls=NumpyEncoder)

# --- Main Processing Functions ---

def detect_yolo(frame):
    results = model(frame, conf=CONF, verbose=False)
    detections = []
    if not results or results[0].masks is None: return detections
    for r in results:
        for box, mask, cls, conf_val in zip(r.boxes.xyxy, r.masks.data, r.boxes.cls, r.boxes.conf):
            # Important: Remapping happens here
            predicted_class = remap_class(model.names[int(cls)])
            x1, y1, x2, y2 = box.cpu().numpy().astype(int)
            resized_mask = cv2.resize(mask.cpu().numpy(), (frame.shape[1], frame.shape[0]),
                                      interpolation=cv2.INTER_NEAREST).astype(bool)
            detections.append([(x1, y1, x2, y2), resized_mask, predicted_class, float(conf_val)])
    return detections

def process_video(video_path):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    video_dataset_dir = os.path.join(DATASET_DIR, video_name)
    masked_video_path = os.path.join(MASKED_DATA_DIR, f"{video_name}.mp4")
    annotated_frames_dir = os.path.join(MASKED_DATA_DIR, video_name)
    os.makedirs(video_dataset_dir, exist_ok=True)
    if SAVE_ANNOTATED_FRAMES: os.makedirs(annotated_frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    out = cv2.VideoWriter(masked_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    
    tracker = ObjectTracker()
    dataset_generator = DatasetGenerator(video_name, (frame_width, frame_height), FINAL_CLASSES)

    instance_tracks, static_tracks = {}, {}
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        detections = detect_yolo(frame)
        
        pupil_detection = max([d for d in detections if d[2] == 'Pupil'], key=lambda x: x[3], default=None)
        cornea_detection = max([d for d in detections if d[2] == 'Cornea'], key=lambda x: x[3], default=None)
        pupil_center = np.mean(np.argwhere(pupil_detection[1]), axis=0)[[1, 0]] if pupil_detection and np.any(pupil_detection[1]) else None

        trackable_detections = [det + [compute_tool_tip(det[1], pupil_center)]
                                for det in detections if det[2] not in EXCLUDED_CLASSES]
        
        tracked_objects = tracker.update(trackable_detections)

        active_classes = {obj.stable_class for obj in tracked_objects}
        for obj in tracked_objects:
            if obj.stable_class not in instance_tracks:
                instance_tracks[obj.stable_class] = {'id': obj.id, 'data': [None] * frame_count}
            instance_tracks[obj.stable_class]['data'].append({
                'seg': obj.mask, 'bbox': obj.get_bbox(), 'area': np.sum(obj.mask),
                'tip': obj.tip_history[-1] if obj.tip_history else None})
        
        for p_class, track in instance_tracks.items():
            if p_class not in active_classes:
                track['data'].append(None)

        for static_obj_name, det in [('Pupil', pupil_detection), ('Cornea', cornea_detection)]:
            if static_obj_name not in static_tracks:
                static_tracks[static_obj_name] = {'id': abs(hash(static_obj_name)) % (10**8), 'data': [None] * frame_count}
            if det:
                center = np.mean(np.argwhere(det[1]), axis=0)[[1, 0]] if np.any(det[1]) else None
                static_tracks[static_obj_name]['data'].append({
                    'seg': det[1], 'bbox': det[0], 'area': np.sum(det[1]), 'center': center})
            else:
                static_tracks[static_obj_name]['data'].append(None)

        overlay_frame = frame.copy()
        # Draw static objects
        for det in (pupil_detection, cornea_detection):
            if det:
                bbox, mask, p_class, _ = det
                color = COLOR_DICT.get(p_class, COLOR_DICT["Default"])
                x1, y1, x2, y2 = bbox
                cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(overlay_frame, p_class, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                if mask is not None: overlay_frame = apply_mask(overlay_frame, mask, color, alpha=0.4)
        # Draw tracked instruments
        for obj in tracked_objects:
            color = COLOR_DICT.get(obj.stable_class, COLOR_DICT["Default"])
            x1, y1, x2, y2 = obj.get_bbox()
            cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(overlay_frame, f"{obj.stable_class}:{obj.id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            if obj.mask is not None: overlay_frame = apply_mask(overlay_frame, obj.mask, color, alpha=0.4)
            if DRAW_TRAJECTORY and len(obj.tip_history) > 1:
                for i in range(1, len(obj.tip_history)):
                    if obj.tip_history[i-1] and obj.tip_history[i]:
                        cv2.line(overlay_frame, obj.tip_history[i-1], obj.tip_history[i], (0, 255, 0), 2)
        
        out.write(overlay_frame)
        if SAVE_ANNOTATED_FRAMES: cv2.imwrite(os.path.join(annotated_frames_dir, f"{frame_count:06d}.jpg"), overlay_frame)
        frame_count += 1
        print(f"Processing {video_name}: Frame {frame_count}/{total_frames}", end='\r')

    # --- Final JSON Generation ---
    frame_files = [f"{i+1:06d}.jpg" for i in range(frame_count)]
    dataset_generator.add_video_entry(frame_files)
    
    all_tracks = {**instance_tracks, **static_tracks}

    for track_key, track in all_tracks.items():
        is_instrument = track_key not in EXCLUDED_CLASSES
        segmentations, bboxes, areas, keypoints_flat = [], [], [], []
        
        for frame_data in track['data']:
            if frame_data:
                contours, _ = cv2.findContours(frame_data['seg'].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                seg_poly = [p for c in contours for p in c.flatten()] if contours else []
                x, y, w, h = cv2.boundingRect(contours[0]) if contours else (0,0,0,0)
                
                segmentations.append([seg_poly] if seg_poly else None)
                bboxes.append([x, y, w, h])
                areas.append(frame_data['area'])
                
                if is_instrument:
                    tip = frame_data.get('tip')
                    keypoints_flat.extend([0, 0, 0, tip[0], tip[1], 2] if tip else [0, 0, 0, 0, 0, 0])
                else: # Pupil or Cornea
                    center = frame_data.get('center')
                    keypoints_flat.extend([center[0], center[1], 2, 0, 0, 0] if center is not None else [0, 0, 0, 0, 0, 0])
            else:
                segmentations.append(None); bboxes.append(None); areas.append(None)
                keypoints_flat.extend([0, 0, 0, 0, 0, 0])
        
        instance_id = track['id']
        class_name = track_key if isinstance(track_key, str) else instance_tracks[track_key]['class_name']
        dataset_generator.add_annotation(instance_id, class_name, segmentations, bboxes, areas, keypoints_flat)
    
    json_output_path = os.path.join(video_dataset_dir, "annotation.json")
    dataset_generator.save_json(json_output_path)
    
    print(f"\nSaving raw frames for {video_name}...")
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret: break
        cv2.imwrite(os.path.join(video_dataset_dir, f"{i+1:06d}.jpg"), frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\nSuccessfully processed {video_name}. Dataset saved to {video_dataset_dir}")

def main():
    video_files = glob.glob(os.path.join(VIDEO_DIR, "*.*"))
    if not video_files: print(f"No videos found in '{VIDEO_DIR}' folder.")
    for video_path in video_files:
        print(f"--- Processing video: {video_path} ---")
        start_time = time.time()
        process_video(video_path)
        end_time = time.time()
        print(f"--- Finished {os.path.basename(video_path)} in {end_time - start_time:.2f} seconds ---\n")

if __name__ == "__main__":
    main()
