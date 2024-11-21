#!/bin/bash

gcloud run deploy redis \
  --image gperdrizet/agatha:redis \
  --service-account ask-agatha-service@ask-agatha.iam.gserviceaccount.com \
  --update-secrets=REDIS_PASSWORD=redis_password:latest \
  --update-env-vars REDIS_IP=$REDIS_IP \
  --update-env-vars REDIS_PORT=$REDIS_PORT \
  --port $REDIS_PORT \
  --no-cpu-throttling \
  --network=default \
  --subnet=default \
  --ingress=all \
  --vpc-egress=private-ranges-only  \
  --region=us-central1

gcloud beta run deploy agatha \
  --service-account=ask-agatha-service@ask-agatha.iam.gserviceaccount.com \
  --concurrency=2 \
  --max-instances=1 \
  --no-cpu-throttling \
  --gpu-type="nvidia-l4" \
  --network=default \
  --subnet=default \
  --ingress=all \
  --vpc-egress=private-ranges-only  \
  --region=us-central1 \
  --container api \
  --image=gperdrizet/agatha:api \
  --update-secrets=REDIS_PASSWORD=redis_password:latest \
  --update-secrets=HF_TOKEN=hf_token:latest \
  --set-env-vars=FLASK_PORT=$FLASK_PORT \
  --set-env-vars=HOST_IP=$HOST_IP \
  --set-env-vars=REDIS_IP=https://redis-224558092745.us-central1.run.app \
  --set-env-vars=REDIS_PORT=$REDIS_PORT \
  --port=$FLASK_PORT \
  --cpu=4 \
  --memory=16Gi \
  --gpu=1 \
  --container=bot \
  --image=gperdrizet/agatha:bot \
  --update-secrets=TELEGRAM_TOKEN=telegram_token:latest \
  --set-env-vars=HOST_IP=https://agatha-224558092745.us-central1.run.app \
  --set-env-vars=FLASK_PORT=$FLASK_PORT