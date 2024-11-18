#!/bin/bash

# Redis container
gcloud run deploy redis \
  --image gperdrizet/agatha:redis \
  --service-account ask-agatha-service@ask-agatha.iam.gserviceaccount.com \
  --update-secrets=REDIS_PASSWORD=redis_password:latest \
  --update-env-vars REDIS_PORT=$REDIS_PORT \
  --port $REDIS_PORT \
  --command "./start_server.sh" \
  --no-cpu-throttling