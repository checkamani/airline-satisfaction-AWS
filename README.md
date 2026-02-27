# Airline Passenger Satisfaction - ML Project

## Problem
Predict whether a passenger is satisfied based on flight experience features.

## Dataset
Source: KaggleHub  
`teejmahal20/airline-passenger-satisfaction`

## Project Structure
- `app/` Flask web app (UI + prediction endpoint)
- `model/` saved trained model (`.pkl`)
- `src/` training / helper scripts
- `requirements.txt` Python dependencies

## How to Run Locally
```bash
pip install -r requirements.txt
python app/app.py

## Run with Kubernetes
Make sure Minikube is running:
```bash
minikube start
kubectl apply -f k8s/
kubectl get pods
minikube service airline-satisfaction-svc
