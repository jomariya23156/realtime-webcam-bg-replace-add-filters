#! /bin/bash

docker run --name webcam-app -e SERVICE_PORT=8000 -p 8000:8000 --rm dl-webcam