import cv2
import numpy as np
import os
import threading
import torch
import socket
import logging
import time
import psutil
import uuid
import subprocess
from flask import Flask, render_template, Response, jsonify, request, session
from collections import deque
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Configuration
class Config:
    CAMERA_INDEX = 0
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 30
    RTSP_ENABLED = True
    RTSP_PORT = 8554
    RTSP_MOUNT = "/stream"
    RTSP_BITRATE = 2000000
    MODEL_PATH = 'yolov5s.pt'
    MODEL_CONF = 0.45
    MODEL_IOU = 0.45
    MAX_CLIENTS = 100
    FRAME_BUFFER_SIZE = 30
    LOG_LEVEL = 'INFO'

    @classmethod
    def get_rtsp_url(cls):
        return f"rtsp://localhost:{cls.RTSP_PORT}{cls.RTSP_MOUNT}"

# Initialize Flask
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Logging
logging.basicConfig(level=Config.LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('webcam_app')

# Global Variables
camera = None
rtsp_process = None
frame_buffer = deque(maxlen=Config.FRAME_BUFFER_SIZE)
client_settings = {}
main_detection_enabled = True
detection_model = None

# Prometheus Metrics
REQUEST_COUNT = Counter("request_count", "Total number of requests received")
CAMERA_FPS = Gauge("camera_fps", "Frames per second of the camera stream")
CAMERA_UPTIME = Gauge("camera_uptime", "Total uptime of the camera in seconds")
DETECTION_ENABLED = Gauge("detection_enabled", "1 if global detection is enabled, 0 if disabled")
OBJECTS_DETECTED = Counter("objects_detected", "Total number of objects detected")
FRAME_PROCESS_TIME = Histogram("frame_process_time", "Time taken to process each frame")
CPU_USAGE = Gauge("cpu_usage", "Current CPU usage percentage")
MEMORY_USAGE = Gauge("memory_usage", "Current memory usage percentage")
CONTAINER_HEALTH = Gauge("container_health", "1 if container is healthy, 0 if not")
TOTAL_DETECTIONS = Counter("total_detections", "Total objects detected since startup")
ACTIVE_CLIENTS = Gauge("active_clients", "Number of clients currently viewing the stream")

# Camera Capture
class CameraCapture:
    def __init__(self):
        self.camera = None
        self.last_frame = None
        self.last_update = 0
        self.running = False
        
    def initialize(self):
        global camera
        try:
            self.camera = cv2.VideoCapture(Config.CAMERA_INDEX)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
            self.camera.set(cv2.CAP_PROP_FPS, Config.CAMERA_FPS)
            
            if not self.camera.isOpened():
                raise RuntimeError("Could not open camera")
                
            camera = self.camera
            logger.info("Camera initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Camera initialization failed: {str(e)}")
            return False
            
    def read_frame(self):
        if not self.camera or not self.camera.isOpened():
            if not self.initialize():
                return None
                
        ret, frame = self.camera.read()
        if ret:
            self.last_frame = frame
            self.last_update = time.time()
            return frame
        return None
        
    def get_last_frame(self):
        return self.last_frame
        
    def release(self):
        if self.camera and self.camera.isOpened():
            self.camera.release()
        self.camera = None

# RTSP Server
def start_rtsp_server():
    global rtsp_process
    
    if not Config.RTSP_ENABLED:
        logger.info("RTSP streaming disabled in config")
        return False
        
    if not CameraCapture().initialize():
        return False
        
    command = [
        'gst-launch-1.0',
        'appsrc', 'name=source', 'is-live=true',
        '!', 'videoconvert',
        '!', 'x264enc', 'tune=zerolatency', f'bitrate={Config.RTSP_BITRATE//1000}',
        '!', 'rtph264pay', 'config-interval=1',
        '!', 'tcpserversink', f'port={Config.RTSP_PORT}'
    ]
    
    try:
        rtsp_process = subprocess.Popen(command)
        logger.info(f"RTSP server started at {Config.get_rtsp_url()}")
        return True
    except Exception as e:
        logger.error(f"Failed to start RTSP server: {str(e)}")
        return False

# Object Detection
def load_detection_model():
    global detection_model
    try:
        logger.info("Loading YOLOv5 model...")
        detection_model = torch.hub.load('ultralytics/yolov5', 'custom', path=Config.MODEL_PATH)
        detection_model.conf = Config.MODEL_CONF
        detection_model.iou = Config.MODEL_IOU
        detection_model.classes = None
        detection_model.max_det = 50
        logger.info("YOLOv5 model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

def detect_objects(frame):
    if detection_model is None:
        if not load_detection_model():
            return frame, []
            
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detection_model(rgb_frame)
        detections = results.pandas().xyxy[0]
        
        # Draw detections
        for _, detection in detections.iterrows():
            x1, y1 = int(detection['xmin']), int(detection['ymin'])
            x2, y2 = int(detection['xmax']), int(detection['ymax'])
            label = detection['name']
            confidence = detection['confidence']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        OBJECTS_DETECTED.inc(len(detections))
        TOTAL_DETECTIONS.inc(len(detections))
        return frame, detections
    except Exception as e:
        logger.error(f"Detection error: {str(e)}")
        return frame, []

# Frame Processing
def capture_and_process_frames():
    camera = CameraCapture()
    fps_counter = 0
    fps_timer = time.time()
    start_time = time.time()
    
    while True:
        frame = camera.read_frame()
        if frame is None:
            time.sleep(0.1)
            continue
            
        # Update FPS counter
        fps_counter += 1
        if time.time() - fps_timer >= 1.0:
            CAMERA_FPS.set(fps_counter)
            fps_counter = 0
            fps_timer = time.time()
            
        # Process frame
        if main_detection_enabled:
            processed_frame, _ = detect_objects(frame.copy())
        else:
            processed_frame = frame.copy()
            
        # Add to buffer
        frame_dict = {
            'raw': cv2.imencode('.jpg', frame)[1].tobytes(),
            'processed': cv2.imencode('.jpg', processed_frame)[1].tobytes()
        }
        
        frame_buffer.append(frame_dict)
        CAMERA_UPTIME.set(time.time() - start_time)
        ACTIVE_CLIENTS.set(len(client_settings))

# Web Routes
def get_client_id():
    if 'client_id' not in session:
        session['client_id'] = str(uuid.uuid4())
    return session['client_id']

def get_ip_address():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

@app.route('/')
def index():
    REQUEST_COUNT.inc()
    client_id = get_client_id()
    if client_id not in client_settings:
        client_settings[client_id] = {
            'show_camera': True,
            'detection_enabled': True
        }
    return render_template('index.html', ip_address=get_ip_address(), client_id=client_id)

@app.route('/video_feed')
def video_feed():
    client_id = get_client_id()
    
    if client_id not in client_settings:
        client_settings[client_id] = {
            'show_camera': True,
            'detection_enabled': True
        }
    
    def generate():
        while True:
            if not client_settings[client_id]['show_camera']:
                blank_frame = np.zeros((480, 640, 3), np.uint8)
                blank_frame = cv2.putText(blank_frame, "Camera Disabled", (200, 240), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                ret, buffer = cv2.imencode('.jpg', blank_frame)
                if ret:
                    yield (b'--frame\r\n'
                          b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.5)
                continue
                
            if frame_buffer:
                frame_data = frame_buffer[-1]
                if client_settings[client_id]['detection_enabled']:
                    yield (b'--frame\r\n'
                          b'Content-Type: image/jpeg\r\n\r\n' + frame_data['processed'] + b'\r\n')
                else:
                    yield (b'--frame\r\n'
                          b'Content-Type: image/jpeg\r\n\r\n' + frame_data['raw'] + b'\r\n')
            else:
                time.sleep(0.1)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Other routes (/metrics, /status, /start_camera, /stop_camera, etc.) would follow here...

# Initialization
def initialize():
    # Start RTSP server if enabled
    if Config.RTSP_ENABLED:
        start_rtsp_server()
    
    # Start frame capture thread
    threading.Thread(target=capture_and_process_frames, daemon=True).start()
    
    # Load detection model
    load_detection_model()

# Cleanup
def cleanup():
    global camera, rtsp_process
    if camera:
        camera.release()
    if rtsp_process:
        rtsp_process.terminate()
    logger.info("Application resources released")

if __name__ == '__main__':
    import atexit
    atexit.register(cleanup)
    
    initialize()
    
    # Create basic template if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    if not os.path.exists('templates/index.html'):
        with open('templates/index.html', 'w') as f:
            f.write('''<!DOCTYPE html>
<html>
<head>
    <title>Webcam Stream</title>
    <style>
        body { font-family: Arial; text-align: center; }
        #video-container { margin: 20px auto; }
        .controls { margin: 15px 0; }
        button { padding: 8px 15px; margin: 5px; }
    </style>
</head>
<body>
    <h1>Webcam Streaming</h1>
    <div id="video-container">
        <img src="/video_feed" width="640" height="480">
    </div>
    <div class="controls">
        <button id="start-btn">Start Camera</button>
        <button id="stop-btn">Stop Camera</button>
        <button id="toggle-btn">Toggle Detection</button>
    </div>
    <script>
        // Add your JavaScript controls here
    </script>
</body>
</html>''')
    
    logger.info(f"Starting server at http://{get_ip_address()}:5000")
    app.run(host='0.0.0.0', port=5000, threaded=True)