#! /bin/bash

docker build -t dl-webcam:latest --build-arg SERVICE_PORT=8000 .