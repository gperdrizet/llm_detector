#!/bin/sh

# Set memory overcommit
sysctl vm.overcommit_memory=1

# Start redis server
redis-server /usr/local/etc/redis/redis.conf \
--loglevel warning \
--bind $REDIS_IP \
--requirepass $REDIS_PASSWORD