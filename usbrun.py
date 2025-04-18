from flask import Flask, render_template, Response, jsonify, request, session
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
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
from queue import Queue

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Required for session management

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global Variables
camera = None
output_frame = None
lock = threading.Lock()
frame_queue = Queue(maxsize=10)  # Buffer for latest frames

# Client-specific settings
client_settings = {}  # Dictionary to store settings for each client
main_detection_enabled = True  # Global setting for main processing

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

# Load YOLOv5 Model
model = None
def load_model():
    global model
    try:
        if model is None:
            logger.info("Loading YOLOv5 model...")
            model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt')
            model.conf = 0.45  # Confidence threshold
            model.iou = 0.45   # IoU threshold
            model.classes = None
            model.max_det = 50
            logger.info("YOLOv5 model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")

# Object Detection
def detect_objects(frame):
    global model
    if model is None:
        load_model()
        if model is None:
            return frame
    try:
        start = time.time()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(rgb_frame)
        detections = results.pandas().xyxy[0]
        detected_count = 0
        for _, detection in detections.iterrows():
            x1, y1 = int(detection['xmin']), int(detection['ymin'])
            x2, y2 = int(detection['xmax']), int(detection['ymax'])
            label = detection['name']
            confidence = detection['confidence']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            detected_count += 1
        OBJECTS_DETECTED.inc(detected_count)
        TOTAL_DETECTIONS.inc(detected_count)
        FRAME_PROCESS_TIME.observe(time.time() - start)
    except Exception as e:
        logger.error(f"Detection error: {str(e)}")
    return frame

# Frame Producer - Process frames once and store in queue
def capture_frames():
    global camera, frame_queue, lock, main_detection_enabled
    load_model()
    fps_counter = 0
    fps_timer = time.time()
    start_time = time.time()
    
    while True:
        if camera is None or not camera.isOpened():
            logger.warning("Camera unavailable, waiting...")
            time.sleep(1)
            continue
            
        success, frame = camera.read()
        if not success:
            logger.warning("Failed to capture frame, retrying...")
            time.sleep(0.1)
            continue
            
        # Update FPS counter
        fps_counter += 1
        if time.time() - fps_timer >= 1.0:
            CAMERA_FPS.set(fps_counter)
            fps_counter = 0
            fps_timer = time.time()
            
        # Store both raw and processed frames
        raw_frame = frame.copy()
        
        if main_detection_enabled:
            processed_frame = detect_objects(frame)
        else:
            processed_frame = frame
            
        # Encode frames
        (flag1, encoded_raw) = cv2.imencode(".jpg", raw_frame)
        (flag2, encoded_processed) = cv2.imencode(".jpg", processed_frame)
        
        if not flag1 or not flag2:
            continue
            
        # Add frames to queue
        frames_dict = {
            'raw': bytearray(encoded_raw),
            'processed': bytearray(encoded_processed)
        }
        
        if frame_queue.full():
            try:
                frame_queue.get_nowait()  # Remove oldest frame
            except:
                pass
        try:
            frame_queue.put_nowait(frames_dict)
        except:
            pass
            
        CAMERA_UPTIME.set(time.time() - start_time)
        ACTIVE_CLIENTS.set(len(client_settings))

# Generate personalized video stream for each client
def generate_frames(client_id):
    global frame_queue, client_settings
    
    # Default settings for a new client
    if client_id not in client_settings:
        client_settings[client_id] = {
            'show_camera': True,
            'detection_enabled': True
        }
    
    logger.info(f"Starting stream for client: {client_id}, Total clients: {len(client_settings)}")
    
    try:
        while True:
            if not client_settings.get(client_id, {}).get('show_camera', True):
                # Send a blank frame if client has turned off camera
                blank_frame = np.zeros((480, 640, 3), np.uint8)
                blank_frame = cv2.putText(blank_frame, "Camera Disabled", (200, 240), 
                                         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                (flag, encoded_blank) = cv2.imencode(".jpg", blank_frame)
                if flag:
                    yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                           bytearray(encoded_blank) + b'\r\n')
                time.sleep(0.5)  # Send updates less frequently when camera is off
                continue
                
            # Get the latest frame from the queue
            try:
                frames_dict = frame_queue.get(timeout=1.0)
                
                # Choose frame based on client preferences
                if client_settings[client_id].get('detection_enabled', True):
                    frame_data = frames_dict['processed']
                else:
                    frame_data = frames_dict['raw']
                    
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
            except:
                # If no frame is available, send a heartbeat
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n\r\n')
    finally:
        # Remove client when connection ends
        if client_id in client_settings:
            del client_settings[client_id]
            logger.info(f"Client disconnected: {client_id}, Remaining clients: {len(client_settings)}")
        ACTIVE_CLIENTS.set(len(client_settings))

# Get IP Address
def get_ip_address():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

# Ensure client_id is available
def get_client_id():
    if 'client_id' not in session:
        session['client_id'] = str(uuid.uuid4())
    return session['client_id']

# Routes
@app.route('/')
def index():
    REQUEST_COUNT.inc()
    client_id = get_client_id()
    # Ensure this client has settings
    if client_id not in client_settings:
        client_settings[client_id] = {
            'show_camera': True,
            'detection_enabled': True
        }
    return render_template('index.html', ip_address=get_ip_address(), client_id=client_id)

@app.route('/metrics')
def metrics():
    CPU_USAGE.set(psutil.cpu_percent())
    MEMORY_USAGE.set(psutil.virtual_memory().percent)
    CONTAINER_HEALTH.set(1 if psutil.virtual_memory().available > 100000000 else 0)
    ACTIVE_CLIENTS.set(len(client_settings))
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

@app.route('/video_feed')
def video_feed():
    client_id = get_client_id()
    return Response(generate_frames(client_id), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    client_id = get_client_id()
    client_status = client_settings.get(client_id, {'show_camera': False, 'detection_enabled': False})
    
    return jsonify({
        "camera_running": camera is not None and camera.isOpened(),
        "global_detection_enabled": main_detection_enabled,
        "client_camera_on": client_status.get('show_camera', False),
        "client_detection_enabled": client_status.get('detection_enabled', False),
        "active_clients": len(client_settings),
        "uptime": time.time() - start_time if 'start_time' in globals() else 0,
        "system_cpu": psutil.cpu_percent(),
        "system_memory": psutil.virtual_memory().percent
    })

@app.route('/start_camera', methods=['POST'])
def start_camera():
    global camera
    client_id = get_client_id()

    # Update client settings
    if client_id not in client_settings:
        client_settings[client_id] = {}
    client_settings[client_id]['show_camera'] = True
    
    # Only start actual camera if not already running
    if camera is None:
        try:
            camera = cv2.VideoCapture(0)
            # Set camera parameters for better performance
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_FPS, 30)
            
            if not camera.isOpened():
                camera = None  # Reset if failed to open
                logger.error("Failed to open camera")
                return jsonify({"status": "Failed to open camera", "success": False})
            
            # Start the producer thread if not already running
            if not any(t.name == "frame_producer" for t in threading.enumerate()):
                producer_thread = threading.Thread(target=capture_frames, daemon=True)
                producer_thread.name = "frame_producer"
                producer_thread.start()
                logger.info("Frame producer thread started")
                global start_time
                start_time = time.time()
        except Exception as e:
            logger.error(f"Error starting camera: {str(e)}")
            return jsonify({"status": f"Error: {str(e)}", "success": False})
    
    # Update detection preference if specified
    if request.is_json and 'detection' in request.json:
        client_settings[client_id]['detection_enabled'] = request.json.get('detection')
    
    return jsonify({"status": "Camera started for your view", "success": True})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    client_id = get_client_id()
    
    # Update client settings to not show camera
    if client_id not in client_settings:
        client_settings[client_id] = {}
    client_settings[client_id]['show_camera'] = False
    
    return jsonify({"status": "Camera stopped for your view", "success": True})

@app.route('/toggle_detection', methods=['POST'])
def toggle_detection():
    client_id = get_client_id()
    
    # Initialize client settings if needed
    if client_id not in client_settings:
        client_settings[client_id] = {'show_camera': True, 'detection_enabled': True}
    
    # Toggle detection for this client
    current = client_settings[client_id].get('detection_enabled', True)
    client_settings[client_id]['detection_enabled'] = not current
    
    return jsonify({
        "status": "Detection toggled for your view", 
        "detection_enabled": client_settings[client_id]['detection_enabled'], 
        "success": True
    })

# Toggle global detection processing
@app.route('/toggle_global_detection', methods=['POST'])
def toggle_global_detection():
    global main_detection_enabled
    main_detection_enabled = not main_detection_enabled
    DETECTION_ENABLED.set(1 if main_detection_enabled else 0)
    return jsonify({
        "status": "Global detection processing toggled", 
        "detection_enabled": main_detection_enabled, 
        "success": True
    })

# Shutdown handler
def cleanup():
    global camera
    if camera is not None:
        camera.release()
    logger.info("Application shutting down, resources released")

# Register cleanup handler
import atexit
atexit.register(cleanup)

if __name__ == '__main__':
    # Create a basic HTML template if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    if not os.path.exists('templates/index.html'):
        with open('templates/index.html', 'w') as f:
            f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Personal Camera Stream</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; text-align: center; }
        #video-container { margin: 20px auto; max-width: 100%; }
        #camera-feed { max-width: 100%; border: 1px solid #ddd; }
        .controls { margin: 15px 0; }
        button { padding: 8px 15px; margin: 5px; cursor: pointer; }
        .status { margin: 15px 0; }
        .personal-note { background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin: 10px 0; }
        @media (min-width: 768px) {
            #video-container { max-width: 640px; }
        }
    </style>
</head>
<body>
    <h1>Multi-Client Camera Stream</h1>
    <div class="personal-note">
        <h3>Your Personal View</h3>
        <p>This is your personal stream view. Controls only affect what you see.</p>
        <p>Client ID: <span id="client-id"></span></p>
    </div>
    <div class="status">
        <p>Server IP: <span id="server-ip"></span></p>
        <p>Your View Status: <span id="status-text">Initializing...</span></p>
        <p>Total connected viewers: <span id="client-count">-</span></p>
    </div>
    <div class="controls">
        <button id="start-btn">Start My Camera View</button>
        <button id="stop-btn">Stop My Camera View</button>
        <button id="toggle-btn">Toggle My Detection</button>
    </div>
    <div id="video-container">
        <img id="camera-feed" src="/video_feed" alt="Camera Feed">
    </div>

    <script>
        // Set client ID
        document.getElementById('client-id').textContent = window.location.pathname === '/' ? 'default' : window.location.pathname.substring(1);
        document.getElementById('server-ip').textContent = location.hostname;
        
        // Control buttons
        document.getElementById('start-btn').addEventListener('click', async () => {
            const response = await fetch('/start_camera', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ detection: true })
            });
            updateStatus();
            // Refresh the feed
            refreshFeed();
        });

        document.getElementById('stop-btn').addEventListener('click', async () => {
            const response = await fetch('/stop_camera', { method: 'POST' });
            updateStatus();
        });

        document.getElementById('toggle-btn').addEventListener('click', async () => {
            const response = await fetch('/toggle_detection', { method: 'POST' });
            updateStatus();
        });

        // Refresh the feed to ensure changes take effect
        function refreshFeed() {
            const videoImg = document.getElementById('camera-feed');
            videoImg.src = '/video_feed?t=' + new Date().getTime();
        }

        // Update status regularly
        async function updateStatus() {
            try {
                const response = await fetch('/status');
                const data = await response.json();
                
                let statusText = 'Camera ';
                statusText += data.client_camera_on ? 'enabled' : 'disabled';
                statusText += ' in your view, detection ';
                statusText += data.client_detection_enabled ? 'ON' : 'OFF';
                
                document.getElementById('status-text').textContent = statusText;
                document.getElementById('client-count').textContent = data.active_clients;
            } catch (error) {
                document.getElementById('status-text').textContent = 'Error connecting to server';
            }
        }

        // Check status every 3 seconds
        updateStatus();
        setInterval(updateStatus, 3000);

        // Handle stream errors
        const videoImg = document.getElementById('camera-feed');
        videoImg.onerror = function() {
            // If the stream fails, try to reconnect after a delay
            setTimeout(() => {
                videoImg.src = '/video_feed?t=' + new Date().getTime();
            }, 5000);
        };
    </script>
</body>
</html>
            ''')
    
    logger.info(f"Server starting at http://{get_ip_address()}:5000")
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)