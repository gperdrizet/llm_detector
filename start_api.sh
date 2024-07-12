#!/bin/bash
gunicorn -w 1 --bind $HOST_IP:$FLASK_PORT 'api.__main__:flask_app'