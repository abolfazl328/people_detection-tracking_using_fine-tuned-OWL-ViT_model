import cv2
import os
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from filterpy.kalman import KalmanFilter
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from PIL import Image



TEXT_PROMPTS = ["person", "a person walking"] 
VLM_CONF_THRESHOLD = 0.1 # Confidence threshold for VLM detections

IOU_THRESHOLD = 0.5    # More aggressive IOU for greedy matching
MAX_AGE = 10          # The maximum number of frames a track can be "lost"
MIN_HITS = 15           # The minimum number of hits to confirm a track

INPUT_VIDEO_PATH = './input.mp4'
OUTPUT_VIDEO_PATH = f'./vlm_outputs/vlm_output_{VLM_CONF_THRESHOLD}_{IOU_THRESHOLD}_{MAX_AGE}_{MIN_HITS}.mp4'

ZONE = np.array([[100, 500], [800, 500], [450, 200]], np.int32)

class KalmanTracker:
    def __init__(self):
        self.next_track_id = 0
        self.tracks = {}

    def update(self, boxes):
        # Predict next state for all existing tracks
        for track_id, track in self.tracks.items():
            track['kf'].predict()
            track['age'] += 1

        # Match detections to existing tracks using IOU
        matched_indices = []
        if len(boxes) > 0 and len(self.tracks) > 0:
            track_ids = list(self.tracks.keys())
            track_boxes = np.array([t['kf'].x[:4, 0] for t in self.tracks.values()])
            
            # Simple greedy matching for demonstration
            # In a real scenario, you'd use Hungarian algorithm
            for i, box in enumerate(boxes):
                max_iou = 0
                best_match = -1
                for j, track_box in enumerate(track_boxes):
                    iou = self.iou(box, [track_box[0], track_box[1], track_box[0]+track_box[2], track_box[1]+track_box[3]])
                    if iou > max_iou:
                        max_iou = iou
                        best_match = j
                
                if max_iou > IOU_THRESHOLD: # IOU threshold for matching
                    track_id = track_ids[best_match]
                    self.tracks[track_id]['kf'].update(np.array([box[0], box[1], box[2]-box[0], box[3]-box[1]]).reshape(4,1))
                    self.tracks[track_id]['age'] = 0
                    self.tracks[track_id]['hits'] += 1
                    matched_indices.append(i)

        # Create new tracks for unmatched detections
        for i, box in enumerate(boxes):
            if i not in matched_indices:
                kf = self.create_kalman_filter()
                kf.x[:4] = np.array([box[0], box[1], box[2]-box[0], box[3]-box[1]]).reshape(4,1)
                self.tracks[self.next_track_id] = {'kf': kf, 'age': 0, 'hits': 1}
                self.next_track_id += 1

        # Remove old tracks that have not been seen for a while
        dead_tracks = [track_id for track_id, track in self.tracks.items() if track['age'] > MAX_AGE]
        for track_id in dead_tracks:
            del self.tracks[track_id]

        # Return active tracks
        active_tracks = {}
        for track_id, track in self.tracks.items():
            if track['hits'] > MIN_HITS: # Require a few hits to be considered a stable track
                pos = track['kf'].x
                active_tracks[track_id] = [pos[0,0], pos[1,0], pos[0,0]+pos[2,0], pos[1,0]+pos[3,0]]
        
        return active_tracks

    def create_kalman_filter(self):
        kf = KalmanFilter(dim_x=8, dim_z=4)
        kf.F = np.array([[1,0,0,0,1,0,0,0], [0,1,0,0,0,1,0,0], [0,0,1,0,0,0,1,0], [0,0,0,1,0,0,0,1],
                         [0,0,0,0,1,0,0,0], [0,0,0,0,0,1,0,0], [0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,1]])
        kf.H = np.array([[1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0], [0,0,1,0,0,0,0,0], [0,0,0,1,0,0,0,0]])
        kf.R *= 10.
        kf.P *= 1000.
        kf.Q *= 0.01
        return kf

    def iou(self, bbox1, bbox2):
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area != 0 else 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading OWL-ViT model...")
processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
print("Model loaded.")

model.to(device)
print("Model loaded and moved to GPU.")

tracker = KalmanTracker() # Using your preferred tracker class
cap = cv2.VideoCapture(INPUT_VIDEO_PATH)

width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); fps=int(cap.get(cv2.CAP_PROP_FPS))
fourcc=cv2.VideoWriter_fourcc(*'mp4v'); out=cv2.VideoWriter(OUTPUT_VIDEO_PATH,fourcc,fps,(width,height))
person_counter=0; frames_inside_zone=defaultdict(int)

with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Processing video with VLM") as pbar:
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = processor(text=TEXT_PROMPTS, images=pil_image, return_tensors="pt").to(device)
        with torch.no_grad(): outputs = model(**inputs)
        target_sizes = torch.Tensor([pil_image.size[::-1]])
        vlm_results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=VLM_CONF_THRESHOLD)
        boxes = vlm_results[0]["boxes"].cpu().numpy()
        tracked_objects = tracker.update(boxes)
        # (Drawing and counting logic is the same)
        for object_id, box in tracked_objects.items():
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {object_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            footpoint = (int((x1 + x2) / 2), y2)
            is_inside = cv2.pointPolygonTest(ZONE, footpoint, False) >= 0
            if is_inside:
                frames_inside_zone[object_id] += 1
                if frames_inside_zone[object_id] == 10: person_counter += 1
            else:
                frames_inside_zone[object_id] = 0
        cv2.polylines(frame, [ZONE], isClosed=True, color=(255, 0, 0), thickness=2)
        cv2.putText(frame, f'Count: {person_counter}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        out.write(frame)
        pbar.update(1)

cap.release(); out.release()
print(f"Processing complete. Final video saved to: {OUTPUT_VIDEO_PATH}")

