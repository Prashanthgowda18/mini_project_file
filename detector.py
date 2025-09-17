# detector.py
from ultralytics import YOLO
import cv2
import numpy as np

class Detector:
    def __init__(self, model_path="yolov8n.pt", conf=0.4, device='cpu'):
        self.model = YOLO(model_path)
        self.conf = conf
        # optionally set model.fuse() or .to(device)
    
    def detect(self, frame):
        """
        Returns list of detections as: [ (x1,y1,x2,y2,conf, cls) , ... ]
        Only returns detections for class 'person' (class id 0)
        """
        # ultralytics YOLO predict expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model.predict(rgb, imgsz=640, conf=self.conf, verbose=False)
        dets = []
        # results is list; take first
        r = results[0]
        boxes = r.boxes
        if boxes is None:
            return dets
        for box in boxes:
            cls = int(box.cls.cpu().numpy()[0])
            conf = float(box.conf.cpu().numpy()[0])
            # filter to person (coco class 0)
            if cls != 0:
                continue
            xyxy = box.xyxy.cpu().numpy()[0]  # x1,y1,x2,y2
            x1, y1, x2, y2 = [int(x) for x in xyxy]
            dets.append((x1, y1, x2, y2, conf, cls))
        return dets
