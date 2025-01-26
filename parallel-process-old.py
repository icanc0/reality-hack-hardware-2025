import cv2
import numpy as np
import json
import redis
import time

def connect_redis(host='192.168.86.53', port=6379, db=0):
    return redis.Redis(host=host, port=port, db=db)

def read_depth_map(image_path):
    depth_map = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if depth_map is None:
        raise ValueError(f"Could not read image from {image_path}")
    gray = cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY)
    depth_map = 255 - gray
    return depth_map

def get_detection_from_redis(redis_client):
    pubsub = redis_client.pubsub()
    pubsub.subscribe('yolo_detections')
    
    # Wait for message
    message = pubsub.get_message(timeout=1.0)
    while message is None or message['type'] == 'subscribe':
        message = pubsub.get_message(timeout=1.0)
        time.sleep(0.1)
    
    pubsub.unsubscribe()
    
    if message and message['data']:
        print(f"Updated detection data: {message['data']}")
        return json.loads(message['data'])
    return None

def calculate_relative_depth(depth_map, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    roi = depth_map[y1:y2, x1:x2]
    
    if roi.size == 0:
        return None
    
    avg_depth = float(np.mean(roi))
    return avg_depth / 255.0

def process_detections(depth_map, detection_data):
    if not detection_data or 'object_detection' not in detection_data:
        return None
        
    detections = detection_data['object_detection']
    
    for detection in detections:
        if "bbox" not in detection:
            continue
        bbox = detection["bbox"]
        relative_depth = calculate_relative_depth(depth_map, bbox)
        if relative_depth is not None:
            detection["relative_depth"] = relative_depth
    
    return detection_data

def main():
    redis_client = connect_redis()
    depth_map = read_depth_map("/home/root/sus.png")
    
    while True:
        try:
            detection_data = get_detection_from_redis(redis_client)
            if detection_data:
                updated_data = process_detections(depth_map, detection_data)
                if updated_data:
                    print(f"Updated detection data: {updated_data}")
            time.sleep(0.1)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

if __name__ == "__main__":
    main()
