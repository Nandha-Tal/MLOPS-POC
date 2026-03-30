#!/bin/bash
# Deploy the K8s Monitoring stack to a Kubernetes cluster
set -e

IMAGE="${IMAGE:-mlops-monitoring:latest}"
NAMESPACE="${NAMESPACE:-mlops-monitoring}"

echo "=============================="
echo " K8s Monitoring Deployment"
echo "=============================="
echo "  Image     : $IMAGE"
echo "  Namespace : $NAMESPACE"
echo ""

# Build and push image
echo "[1/4] Building Docker image..."
docker build -t "$IMAGE" .
docker push "$IMAGE" 2>/dev/null || echo "  (skipping push — configure registry if needed)"

# Apply manifests in order
echo "[2/4] Applying K8s manifests..."
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/rbac.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml

# Wait for rollout
echo "[3/4] Waiting for deployment rollout..."
kubectl rollout status deployment/mlops-monitoring-api -n "$NAMESPACE" --timeout=120s

echo "[4/4] Deployment complete!"
echo ""
echo "Access services:"
echo "  kubectl port-forward svc/mlops-monitoring-api 8080:80 -n $NAMESPACE"
echo "  kubectl port-forward svc/mlflow-server 5000:5000 -n $NAMESPACE"
