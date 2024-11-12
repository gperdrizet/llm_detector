#!/bin/bash

# Start the API docker container
docker run -p 5000:5000 -e HF_TOKEN=$HF_TOKEN --gpus all --name agatha_api -d gperdrizet/agatha:api