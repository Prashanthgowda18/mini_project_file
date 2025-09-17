# main.py
import cv2
import time
import numpy as np
from detector import Detector
from tracker import CentroidTracker
import paho.mqtt.client as mqtt
from config import *
import math

# Safety (simulate)
def door_alert():
    print("[SAFETY] Door alert sound triggered!")

def engine_stop():
    print("[SAFETY] Engine stop triggered!")

def main():
    # Initialize detector and tracker
    det = Detector(MODEL_PATH, conf=CONFIDENCE)
    tracker = CentroidTracker(maxDisappeared=40, maxDistance=60)

    cap = cv2.VideoCapture(VIDEO_SOURCE )
    if not cap.isOpened():
        print("Error opening video source:", VIDEO_SOURCE)
        return

    # MQTT client
    client = mqtt.Client()
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_start()
    except Exception as e:
        print("MQTT connect failed:", e)
        client = None

    last_publish = time.time()
    count_inside = 0
    counted_ids = set()

    # compute line
    ret, frame = cap.read()
    if not ret:
        print("Cannot read from source")
        return
    H, W = frame.shape[:2]
    line_y = int(H * LINE_POSITION)

    print("Starting loop. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect people
        dets = det.detect(frame)
        rects = [(x1,y1,x2,y2) for (x1,y1,x2,y2,conf,cls) in dets]

        # Update tracker
        objects = tracker.update(rects)

        # Draw counting line
        cv2.line(frame, (0, line_y), (W, line_y), (0,255,255), 2)

        # For each tracked object, check history and crossing
        for objectID, tobj in list(objects.items()):
            cX, cY = tobj.centroid
            x1,y1,x2,y2 = tobj.bbox
            # draw bbox & id
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"ID {objectID}", (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            # compute if crossed line recently
            if len(tobj.history) >= 2:
                prev_y = tobj.history[-2][1]
                curr_y = tobj.history[-1][1]
                # crossing downward (entering if line_y is downward)
                if prev_y < line_y and curr_y >= line_y:
                    # avoid double counting same id
                    if objectID not in counted_ids:
                        count_inside += 1
                        counted_ids.add(objectID)
                        print(f"ID {objectID} ENTERED. Count: {count_inside}")
                # crossing upward (exiting)
                elif prev_y > line_y and curr_y <= line_y:
                    if objectID not in counted_ids:
                        count_inside = max(0, count_inside - 1)
                        counted_ids.add(objectID)
                        print(f"ID {objectID} EXITED. Count: {count_inside}")

            # optional: reset counted_ids for objects that disappeared long back
            # cleanup counted_ids when object deregistered
        # cleanup: remove counted_ids for objects not present
        present_ids = set(objects.keys())
        counted_ids = {i for i in counted_ids if i in present_ids}

        # display count
        cv2.putText(frame, f"Inside: {count_inside}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

        # safety check
        if count_inside >= CAPACITY_THRESHOLD:
            door_alert()
            engine_stop()

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        # publish periodically
        if client and (time.time() - last_publish) > PUBLISH_INTERVAL_SEC:
            payload = {"count": count_inside, "timestamp": int(time.time())}
            try:
                import json
                client.publish(MQTT_TOPIC, json.dumps(payload))
            except Exception as e:
                print("MQTT publish failed:", e)
            last_publish = time.time()

    cap.release()
    cv2.destroyAllWindows()
    if client:
        client.loop_stop()
        client.disconnect()

if _name_ == "_main_":
    main()