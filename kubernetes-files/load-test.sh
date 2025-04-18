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
