#!/bin/bash

# Helper script to build all project Docker images

cd ./redis
./build_redis_image.sh

cd ../api
./build_api_image.sh

cd ../telegram_bot
./build_bot_image.sh