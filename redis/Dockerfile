FROM redis:7.2-alpine

# Move our redis.conf in
COPY ./redis.conf /usr/local/etc/redis/redis.conf

# Move the server start helper scrip in
WORKDIR /redis
COPY ./start_server.sh .