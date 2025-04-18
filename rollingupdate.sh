#!/bin/bash

# Script to perform rolling updates of the camera app
# Usage: ./update-deployment.sh [new-image-tag]

NEW_TAG=${1:-"latest"}
DEPLOYMENT_NAME="flask-camera-app"

echo "Updating $DEPLOYMENT_NAME to version: $NEW_TAG"

# Update the deployment with the new image tag
kubectl set image deployment/$DEPLOYMENT_NAME $DEPLOYMENT_NAME=amardocker608/flask-camera-app:$NEW_TAG

# Check the rollout status
kubectl rollout status deployment/$DEPLOYMENT_NAME

if [ $? -eq 0 ]; then
  echo "Deployment updated successfully to version: $NEW_TAG"
else
  echo "Deployment update failed!"
  echo "Rolling back..."
  kubectl rollout undo deployment/$DEPLOYMENT_NAME
fi