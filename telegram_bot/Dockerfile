FROM python:3.10-slim-bookworm

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Update & install python & pip
RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -y python3 python3-pip
RUN python3 -m pip install --upgrade pip

# Set the woring directory and move the source code in
WORKDIR /agatha_bot
COPY . /agatha_bot

# Install dependencies
RUN pip install -r requirements.txt

CMD ["python3", "./bot.py"]