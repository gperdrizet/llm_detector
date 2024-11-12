#!/bin/bash

docker build \
--build-arg host_ip=$HOST_IP \
--build-arg flask_port=$FLASK_PORT \
--build-arg redis_ip=$REDIS_IP \
--build-arg redis_port=$REDIS_PORT \
--secret redis_password=$REDIS_PASSWORD \
--secret hf_token=$HF_token \
-t gperdrizet/agatha:api .