#!/bin/bash

echo "Deploying Flask Camera Application with Camera Server architecture..."

# Create directory if it doesn't exist
mkdir -p kubernetes-files
cd kubernetes-files

# Create the YAML files
cat > camera-server-deployment.yaml << 'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: camera-server
  labels:
    app: camera-server
spec:
  replicas: 1
  selector:
    matchLabels:
      app: camera-server
  template:
    metadata:
      labels:
        app: camera-server
    spec:
      containers:
      - name: camera-server
        image: amardocker608/flask-camera-app:latest
        ports:
        - containerPort: 5000
        env:
        - name: RUN_AS_SERVER
          value: "true"
        volumeMounts:
        - name: video-device
          mountPath: /dev/video0
        resources:
          requests:
            memory: "500Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        securityContext:
          privileged: true
      volumes:
      - name: video-device
        hostPath:
          path: /dev/video0
EOF

cat > camera-server-service.yaml << 'EOF'
apiVersion: v1
kind: Service
metadata:
  name: camera-server
spec:
  selector:
    app: camera-server
  ports:
  - port: 8080
    targetPort: 5000
  type: ClusterIP
EOF

cat > flask-app-deployment.yaml << 'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: flask-camera-app
  labels:
    app: flask-camera-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: flask-camera-app
  template:
    metadata:
      labels:
        app: flask-camera-app
    spec:
      containers:
      - name: flask-camera-app
        image: amardocker608/flask-camera-app:latest
        ports:
        - containerPort: 5000
        env:
        - name: RUN_AS_SERVER
          value: "false"
        - name: USE_CENTRAL_SERVER
          value: "true"
        - name: CAMERA_SERVER_URL
          value: "http://camera-server:8080/video_feed"
        resources:
          requests:
            memory: "500Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
EOF

cat > flask-app-service.yaml << 'EOF'
apiVersion: v1
kind: Service
metadata:
  name: flask-camera-app-service
spec:
  selector:
    app: flask-camera-app
  ports:
  - port: 80
    targetPort: 5000
  type: LoadBalancer
EOF

cat > flask-app-hpa.yaml << 'EOF'
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: flask-camera-app-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: flask-camera-app
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 30
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
    scaleDown:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 30
EOF

# Create load testing script
cat > load-test.sh << 'EOF'
#!/bin/bash

# Simple load test script
SERVICE_IP="$(minikube service flask-camera-app-service --url | head -n 1)"
if [ -z "$SERVICE_IP" ]; then
    echo "Error: Couldn't get service URL. Make sure the service is running."
    exit 1
fi

NUM_CONCURRENT_USERS=30
DURATION_SECONDS=300

echo "Starting load test with $NUM_CONCURRENT_USERS concurrent users for $DURATION_SECONDS seconds"
echo "Press Ctrl+C to stop the test"

# Function to simulate a user session
simulate_user() {
  while true; do
    curl -s "$SERVICE_IP/"
    curl -s -X POST "$SERVICE_IP/start_camera"
    curl -s "$SERVICE_IP/video_feed" -o /dev/null &
    CURL_PID=$!
    sleep 3
    kill $CURL_PID 2>/dev/null
    sleep 2
  done
}

# Start the user simulations in the background
for ((i=1; i<=$NUM_CONCURRENT_USERS; i++)); do
  simulate_user &
  echo "Started user $i"
  sleep 0.2
done

# Let the test run for the specified duration
sleep $DURATION_SECONDS

# Clean up
echo "Test completed. Cleaning up..."
pkill -P $$
EOF

chmod +x load-test.sh

# Apply Kubernetes manifests
echo "Applying camera server deployment and service..."
kubectl apply -f camera-server-deployment.yaml
kubectl apply -f camera-server-service.yaml

echo "Waiting for camera server to be ready..."
kubectl rollout status deployment/camera-server

echo "Applying Flask application deployment, service, and HPA..."
kubectl apply -f flask-app-deployment.yaml
kubectl apply -f flask-app-service.yaml
kubectl apply -f flask-app-hpa.yaml

echo "Waiting for Flask application deployment to be ready..."
kubectl rollout status deployment/flask-camera-app

# Get the service URL
echo "Getting service URLs..."
echo "Camera Server (internal): http://camera-server:8080"
echo "Flask App (external):"
kubectl get service flask-camera-app-service

# Instructions for accessing the app
cat << EOF

Your Flask Camera application is now deployed on Kubernetes with auto-scaling!

To access the application:
  - If using Minikube: minikube service flask-camera-app-service
  - If using a cloud provider: Use the EXTERNAL-IP shown above

Monitor your deployment:
  - Watch pods: kubectl get pods -w
  - Check HPA: kubectl get hpa flask-camera-app-hpa -w
  - View logs:
    * Camera server: kubectl logs -f -l app=camera-server
    * Flask app: kubectl logs -f -l app=flask-camera-app

Run the load test to trigger auto-scaling:
  ./load-test.sh

EOF