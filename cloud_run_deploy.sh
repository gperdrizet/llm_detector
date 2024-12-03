#!/bin/bash

# gcloud run deploy redis \
#   --image gperdrizet/agatha:redis \
#   --service-account ask-agatha-service@ask-agatha.iam.gserviceaccount.com \
#   --update-secrets=REDIS_PASSWORD=redis_password:latest \
#   --update-env-vars REDIS_IP=$REDIS_IP \
#   --update-env-vars REDIS_PORT=$REDIS_PORT \
#   --port $REDIS_PORT \
#   --no-cpu-throttling \
#   --network=default \
#   --subnet=default \
#   --ingress=all \
#   --vpc-egress=private-ranges-only  \
#   --region=us-central1 \
#   --no-allow-unauthenticated

gcloud beta run deploy agatha \
  --execution-environment gen2 \
  --service-account=ask-agatha-service@ask-agatha.iam.gserviceaccount.com \
  --concurrency=1 \
  --min-instances=1 \
  --max-instances=1 \
  --gpu-type="nvidia-l4" \
  --no-cpu-throttling \
  --network=default \
  --subnet=default \
  --ingress=all \
  --vpc-egress=private-ranges-only  \
  --region=us-central1 \
  --no-allow-unauthenticated \
  --container api \
  --image=gperdrizet/agatha:api \
  --update-secrets=REDIS_PASSWORD=redis_password:latest \
  --update-secrets=HF_TOKEN=hf_token:latest \
  --set-env-vars=FLASK_PORT=$FLASK_PORT \
  --set-env-vars=HOST_IP=$HOST_IP \
  --set-env-vars=REDIS_IP=https://redis-224558092745.us-central1.run.app \
  --set-env-vars=REDIS_PORT=$REDIS_PORT \
  --port=$FLASK_PORT \
  --cpu=6 \
  --memory=24Gi \
  --gpu=1 \
  --container=bot \
  --image=gperdrizet/agatha:bot \
  --update-secrets=TELEGRAM_TOKEN=telegram_token:latest \
  --set-env-vars=HOST_IP=https://agatha-224558092745.us-central1.run.app \
  --set-env-vars=FLASK_PORT=$FLASK_PORT \
  --cpu=2