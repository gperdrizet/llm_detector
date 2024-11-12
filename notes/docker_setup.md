# Containerization

1. Docker setup
2. Nvidia container toolkit
3. Container builds

## 1. Docker setup

Start with a fresh, up-to-date docker install if needed. From the [Ubuntu install instructions](https://docs.docker.com/engine/install/ubuntu/).

Remove old install:

```text
for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do sudo apt-get remove $pkg; done
sudo apt-get purge docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin docker-ce-rootless-extras
sudo rm -rf /var/lib/docker
sudo rm -rf /var/lib/containerd
```

Then install Docker from the Docker apt repository:

```text
sudo apt update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

Verify the install with:

```text
$ sudo docker run hello-world

Unable to find image 'hello-world:latest' locally
latest: Pulling from library/hello-world
c1ec31eb5944: Pull complete 
Digest: sha256:d000bc569937abbe195e20322a0bde6b2922d805332fd6d8a68b19f524b7d21d
Status: Downloaded newer image for hello-world:latest

Hello from Docker!
This message shows that your installation appears to be working correctly.
```

OK! Looks good. Now, make a docker user group and add yourself.

```text
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
```

Set Docker to run on boot:

```text
sudo systemctl enable docker.service
sudo systemctl enable containerd.service
```

Finally, set up log rotation and set the data location. Add the following to */etc/docker/daemon.json*, creating the file if it doesn't exist. The data-root parameter is not necessary, omit it to accept the system default or set is to an alternative location. We are using a 'fast_scratch' NVMe SSD for our images and containers.

```json
{
    "data-root": "/mnt/fast_scratch/docker",
    "log-driver": "json-file",
    "log-opts": {
        "max-file": "3",
        "max-size": "10m"
    }
}
```

Then restart the daemon:

```text
sudo systemctl restart docker
```

## 2. Nvidia container toolkit

To run Nvidia/CUDA containers we need the Nvidia container toolkit. First add the repo:

```text
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

Then update and install the toolkit:

```text
sudo apt update
sudo apt install nvidia-container-toolkit
```

Now, configure docker and restart the daemon:

```text
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

This will make changes to */etc/docker/daemon.json*, adding the *runtimes* stanza:

```json
{
    "data-root": "/mnt/fast_scratch/docker",
    "log-driver": "json-file",
    "log-opts": {
        "max-file": "3",
        "max-size": "10m"
    },
    "runtimes": {
        "nvidia": {
            "args": [],
            "path": "nvidia-container-runtime"
        }
    }
}
```

Test with:

```text
$ docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi

Unable to find image 'ubuntu:latest' locally
latest: Pulling from library/ubuntu
bccd10f490ab: Pull complete 
Digest: sha256:77906da86b60585ce12215807090eb327e7386c8fafb5402369e421f44eff17e
Status: Downloaded newer image for ubuntu:latest
Tue Mar 12 21:50:44 2024       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.42.01    Driver Version: 470.42.01    CUDA Version: 11.4     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla K80           On   | 00000000:05:00.0 Off |                    0 |
| N/A   28C    P8    26W / 149W |      0MiB / 11441MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  Tesla K80           On   | 00000000:06:00.0 Off |                    0 |
| N/A   33C    P8    29W / 149W |      0MiB / 11441MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

OK, looks good.

## 3. Container builds

Next, build the three containers we need for this project

1. Redis
2. The API
3. The Telegram bot

The containers will make use of some secrets set in the host OS via environment variables. They are set via the Python's virtualenv *.venv/bin/activate*:

```bash
export REDIS_PASSWORD="<password>"
export HF_TOKEN="<hf_token>"
```

### 3. Redis image build

Going to build a custom image based on the official *redis:alpine-7.1* image. Doing so will give us more configuration control over the server and also. We can also keep our own tailored copy of the image in the project DockerHub repo. To start, we will be using a few non-default settings:

1. vm.overcommit_memory=1
2. Password set via compose.yaml by passing in a host environment variable.
3. Redis IP set to 0.0.0.0, port to 6379 inside the container, also via compose.yaml.

Redis will be launched inside the container via a runner script accepting a custom configuration file:

```bash
#!/bin/sh

# Set memory overcommit
sysctl vm.overcommit_memory=1

# Start redis server
redis-server /usr/local/etc/redis/redis.conf \
--loglevel warning \
--bind $REDIS_IP \
--requirepass $REDIS_PASSWORD
```

The image is pretty basic, just includes the runner script and *redis.conf*. Here is the Dockerfile:

```bash
FROM redis:7.2-alpine

# Move our redis.conf in
COPY ./redis.conf /usr/local/etc/redis/redis.conf

# Move the server start helper scrip in
WORKDIR /redis
COPY ./start_server.sh .
```

## 4. API image build

The second container will be for the API. It needs to launch the Flask-Celery app via Guincorn and communicate with the Redis server container. Internal ports and IP addresses are specified via compose.yaml. Secrets are set via compose.yaml by passing in host environment variables.

Here is the Dockerfile to build the API container image:

```bash
FROM nvidia/cuda:11.4.3-runtime-ubuntu20.04

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Set the working directory and move the source code in
WORKDIR /agatha_api
COPY . /agatha_api

# Install python 3.8 & pip
RUN apt-get update
RUN apt-get install -y python3 python3-pip
RUN python3 -m pip install --upgrade pip

# Install dependencies
RUN pip install -r requirements.txt

# Install bitsandbytes
WORKDIR /agatha_api/bitsandbytes-0.42.0
RUN python3 setup.py install

# Set the working directory back
WORKDIR /agatha_api
```

The API is started via a helper script inside of the container:

```bash
#!/bin/bash

# Start the API with Gunicorn
gunicorn -w 1 --bind $HOST_IP:$FLASK_PORT 'api:flask_app'
```

## 5. Telegram bot image build

### TODO

## 6. Image uploads

Once we are sure the images are working. Clean up the system and do a final fresh build of all of the images

```bash
docker rmi gperdrizet/agatha:redis
docker rmi gperdrizet/agatha:api
docker system prune
```

### Redis

```bash
cd redis
./build_redis_image.sh
```

### Agatha API

```bash
cd api
./build_agatha_api_image.sh
```

Then push the images to DockerHub:

```bash
docker push gperdrizet/agatha:redis
docker push gperdrizet/agatha:api
```

Then remove the local copies of the images so we can test everything out:

```bash
docker rmi gperdrizet/agatha:redis
docker rmi gperdrizet/agatha:api
docker system prune
```

## 7. Docker compose

Finally, put it all together and run the project via docker compose

```text
name: agatha

services:

  redis:
    container_name: redis
    image: gperdrizet/agatha:redis
    restart: unless-stopped
    environment:
      REDIS_IP: '0.0.0.0'
      REDIS_PORT: '6379'
      REDIS_PASSWORD: $REDIS_PASSWORD
    ports:
      - '6379:6379'
    command: ./start_server.sh
    privileged: true

  agatha_api:
    container_name: agatha_api
    image: gperdrizet/agatha:api
    restart: unless-stopped
    environment:
      HOST_IP: '0.0.0.0'
      FLASK_PORT: '5000'
      REDIS_IP: redis
      REDIS_PORT: '6379'
      HF_TOKEN: $HF_TOKEN
      REDIS_PASSWORD: $REDIS_PASSWORD
    ports:
      - 5000:5000
    deploy:
      resources:
        reservations:
          devices:
          - driver: nvidia
            device_ids: ['0', '1', '2']
            capabilities: [gpu]
    command: ./start_api.sh
```

Run it:

```bash
docker compose up -d
```
