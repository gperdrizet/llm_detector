FROM python:3.8-slim-bookworm

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Set the woring directory and move the source code in
WORKDIR /agatha_bot
COPY . /agatha_bot

# Install python 3.8 & pip
RUN apt-get update
RUN apt-get install -y python3 python3-pip

# Install dependencies
RUN pip install -r requirements.txt