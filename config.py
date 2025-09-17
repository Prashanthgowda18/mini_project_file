# config.py
VIDEO_SOURCE = 0  # 0 for webcam, or "video.mp4"
MODEL_PATH = "yolov8n.pt"  # small YOLOv8; download model if needed
CONFIDENCE = 0.4

# Counting line: y coordinate (in pixels). We'll set relative if prefer.
LINE_POSITION = 0.5  # fraction of frame height where counting line is (0..1)
DIRECTION_TOLERANCE = 5  # pixels to consider direction movement

# Threshold for overcrowding (example)
CAPACITY_THRESHOLD = 30

# MQTT
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "transport/bus1/passenger_count"

# Safety actions (GPIO pins if real hardware)
USE_GPIO = False
DOOR_ALERT_PIN = 17
ENGINE_STOP_PIN = 27

# Misc
PUBLISH_INTERVAL_SEC = 5  # publish counts at this interval
