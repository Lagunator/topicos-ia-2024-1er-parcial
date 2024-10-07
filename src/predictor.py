from ultralytics import YOLO
import numpy as np
import cv2
from shapely.geometry import Polygon, Point
from src.models import Detection, PredictionType, Segmentation
from src.config import get_settings

SETTINGS = get_settings()

def annotate_detection(image_array: np.ndarray, detection: Detection) -> np.ndarray:
    ann_color = (255, 0, 0)
    annotated_img = image_array.copy()
    for label, conf, box in zip(detection.labels, detection.confidences, detection.boxes):
        x1, y1, x2, y2 = box
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), ann_color, 3)
        cv2.putText(
            annotated_img,
            f"{label}: {conf:.1f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            ann_color,
            2,
        )
    return annotated_img

def annotate_segmentation(image_array: np.ndarray, segmentation: Segmentation, draw_boxes: bool = True) -> np.ndarray:
    annotated_img = image_array.copy()

    for polygon, box, label in zip(segmentation.polygons, segmentation.boxes, segmentation.labels):
        color = (0, 255, 0) if label == "safe" else (255, 0, 0)  
        
        pts = np.array(polygon, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(annotated_img, [pts], isClosed=True, color=color, thickness=3)

       
        if draw_boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated_img,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2,
            )

    return annotated_img

class GunDetector:
    def __init__(self) -> None:
        print(f"loading od model: {SETTINGS.od_model_path}")
        self.od_model = YOLO(SETTINGS.od_model_path)
        print(f"loading seg model: {SETTINGS.seg_model_path}")
        self.seg_model = YOLO(SETTINGS.seg_model_path)

    def detect_guns(self, image_array: np.ndarray, threshold: float = 0.5):
        results = self.od_model(image_array, conf=threshold)[0]
        labels = results.boxes.cls.tolist()
        indexes = [
            i for i in range(len(labels)) if labels[i] in [3, 4]
        ]  
        boxes = [
            [int(v) for v in box]
            for i, box in enumerate(results.boxes.xyxy.tolist())
            if i in indexes
        ]
        confidences = [
            c for i, c in enumerate(results.boxes.conf.tolist()) if i in indexes
        ]
        labels_txt = [
            results.names[labels[i]] for i in indexes
        ]
        return Detection(
            pred_type=PredictionType.object_detection,
            n_detections=len(boxes),
            boxes=boxes,
            labels=labels_txt,
            confidences=confidences,
        )
    
    def segment_people(self, image_array: np.ndarray, threshold: float = 0.5, max_distance: int = 100):
        results = self.seg_model(image_array, conf=threshold)[0]
        labels = results.boxes.cls.tolist()
        people_indexes = [
            i for i in range(len(labels)) if labels[i] == 0
        ]  

        segments = [
            [list(map(int, pt)) for pt in polygon]
            for i, polygon in enumerate(results.masks.xy)
            if i in people_indexes
        ]
        boxes = [
            [int(v) for v in box]
            for i, box in enumerate(results.boxes.xyxy.tolist())
            if i in people_indexes
        ]

        detection = self.detect_guns(image_array, threshold)
        gun_bboxes = detection.boxes
        labels_txt = []

        for segment, box in zip(segments, boxes):
           
            person_polygon = Polygon(segment)
            person_center = person_polygon.centroid
            person_point = Point(person_center.x, person_center.y)

            
            is_danger = False
            for gun_bbox in gun_bboxes:
                x1, y1, x2, y2 = gun_bbox
                gun_box_center = Point((x1 + x2) / 2, (y1 + y2) / 2)
                distance = person_point.distance(gun_box_center)
                if distance <= max_distance:
                    is_danger = True
                    break
                person_box = Polygon([(box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3])])
                gun_box = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

                if person_box.intersects(gun_box):
                    is_danger = True
                    break

            if is_danger:
                labels_txt.append("danger")
            else:
                labels_txt.append("safe")

        return Segmentation(
            pred_type=PredictionType.segmentation,
            n_detections=len(segments),
            polygons=segments,
            boxes=boxes,
            labels=labels_txt,
        )
