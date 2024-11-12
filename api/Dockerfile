FROM nvidia/cuda:11.4.3-runtime-ubuntu20.04

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Take command line build time arguments for host ip
# and flask port. These are used later when starting
# the API
ARG host_ip
ARG flask_port
ARG redis_ip
ARG redis_port
ARG redis_password
# ARG hf_token

# Then sent them as environment variables inside the
# container
ENV HOST_IP=$host_ip
ENV FLASK_PORT=$flask_port
ENV REDIS_IP=$redis_ip
ENV REDIS_PORT=$redis_port
ENV REDIS_PASSWORD=$redis_password
# ENV HF_TOKEN=$hf_token

# Set working directory
WORKDIR /

# Install python 3.8 & pip
RUN apt-get update
RUN apt-get install -y python3 python3-pip
RUN python3 -m pip install --upgrade pip

# Move the source code in
WORKDIR /agatha_api
COPY . /agatha_api

# Install dependencies
RUN pip install -r requirements.txt

# Install bitsandbytes
WORKDIR /agatha_api/bitsandbytes-0.42.0
RUN python3 setup.py install

# Start the API
WORKDIR /agatha_api
CMD ["./start_api.sh"]