kind create cluster

kubectl get service
kubectl get deployment
kubectl get pod

kind load docker-image zoomcamp-model:v001
kubectl apply -f hw_deployment.yaml

kubectl get deployment
kubectl get pod

kubectl port-forward credit-card-7f8c9dd644-j7n7l 9696:9696

kubectl exec -it credit-card-7f8c9dd644-j7n7l -- bash

kubectl apply -f hw_service.yaml
kubectl get service

kubectl port-forward service/credit-card 8080:80

kubectl delete -f hw_deployment.yaml
