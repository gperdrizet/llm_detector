FROM nvidia/cuda:11.4.3-runtime-ubuntu20.04

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Update & install python & pip
RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -y python3.8 python3-pip
RUN python3.8 -m pip install --upgrade pip

# Make mount point for Google Cloud storage bucket for HuggingFace model cache
RUN mkdir /mnt/hf_cache

# Set the working directory
WORKDIR /agatha_api

# Copy requirements in and install
COPY ./requirements.txt /agatha_api
RUN pip install -r requirements.txt

# Copy the rest of the source code in
COPY . /agatha_api

CMD ["./start_api.sh"]