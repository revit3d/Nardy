#!/bin/bash

docker build -t triomino-tiles-classification-app:latest .

xhost +local:

docker run --rm -it -e DISPLAY=unix$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix triomino-tiles-classification-app:latest
