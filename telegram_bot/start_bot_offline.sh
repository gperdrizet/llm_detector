#!/bin/bash

docker run --detach --restart=always \
--env HOST_IP=$HOST_IP \
--env FLASK_PORT=$FLASK_PORT \
--env TELEGRAM_TOKEN=$TELEGRAM_TOKEN \
gperdrizet/agatha:bot-offline