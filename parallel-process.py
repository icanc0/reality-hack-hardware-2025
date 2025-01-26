import cv2
import numpy as np
import json
import redis
import http.server
import socketserver
from threading import Thread

def connect_redis():
    return redis.Redis(host='192.168.86.53', port=6379, db=0)

def read_depth_map(image_path):
    depth_map = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if depth_map is None:
        raise ValueError(f"Could not read image from {image_path}")
    gray = cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY)
    return 255 - gray

def convert_normalized_to_pixel_coords(rect, img_height, img_width):
    x = int(rect['x'] * img_width)
    y = int(rect['y'] * img_height)
    w = int(rect['width'] * img_width)
    h = int(rect['height'] * img_height)
    return [x, y, x + w, y + h]

def get_latest_detection(redis_client, depth_map):
    pubsub = redis_client.pubsub()
    pubsub.subscribe('yolo_detections')
    height, width = depth_map.shape
    
    message = pubsub.get_message(timeout=1.0)
    while message is None or message['type'] != 'message':
        message = pubsub.get_message(timeout=1.0)
    
    pubsub.unsubscribe()
    
    try:
        data = json.loads(message['data'])
        timestamp = data['parameters']['timestamp']
        
        processed_detections = []
        for detection in data['object_detection']:
            if 'rectangle' in detection:
                bbox = convert_normalized_to_pixel_coords(detection['rectangle'], height, width)
                roi = depth_map[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                relative_depth = float(np.mean(roi)) / 255.0 if roi.size > 0 else None
                
                processed_detections.append({
                    "label": detection['label'],
                    "confidence": detection['confidence'],
                    "bbox": bbox,
                    "relative_depth": relative_depth
                })
        
        return {
            "timestamp": timestamp,
            "detections": processed_detections
        }
    except Exception as e:
        print(f"Error processing message: {e}")
        return {"timestamp": None, "detections": []}

class DetectionHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.redis_client = connect_redis()
        self.depth_map = read_depth_map("/home/root/sus.png")
        super().__init__(*args, **kwargs)

    def do_GET(self):
        if self.path == '/detections':
            latest_data = get_latest_detection(self.redis_client, self.depth_map)
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(latest_data).encode())
        else:
            self.send_response(404)
            self.end_headers()

def main():
    PORT = 5003
    with socketserver.TCPServer(("", PORT), DetectionHandler) as httpd:
        print(f"Serving at port {PORT}")
        httpd.serve_forever()

if __name__ == "__main__":
    main()