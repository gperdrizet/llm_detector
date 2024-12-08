#!/bin/bash

# Helper script to build and push local deployment images

# Build the images
cd ./redis
./build_redis_image.sh

cd ../api
./build_api_image.sh

cd ../telegram_bot
./build_bot_image.sh

# Push the images
docker push gperdrizet/agatha:redis-local
docker push gperdrizet/agatha:api-local
docker push gperdrizet/agatha:bot-local