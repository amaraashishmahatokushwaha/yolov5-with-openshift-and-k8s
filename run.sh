#!/bin/bash

# Apply the Kubernetes resources in the correct order
kubectl apply -f kubernetes/configmap.yml
kubectl apply -f kubernetes/deployment.yml
kubectl apply -f kubernetes/service.yml
kubectl apply -f kubernetes/hpa-custom.yml

# Wait for deployment to be ready
#kubectl rollout status deployment/flask-camera-app

# Get the service URL (if using LoadBalancer)
echo "Getting service URL..."
kubectl get service flask-camera-app-service

# Instructions for accessing the app
cat << EOF

Your Flask Camera application is now deployed on Kubernetes!

If you're using Minikube, run:
  minikube service flask-camera-app-service

If you're using a cloud provider with LoadBalancer support, access:
  http://EXTERNAL_IP (shown above)

To monitor your HPA:
  kubectl get hpa flask-camera-app-hpa --watch

To view logs:
  kubectl logs -f -l app=flask-camera-app

To run the load test (adjust the SERVICE_IP first):
  ./load-test-script.sh
EOF