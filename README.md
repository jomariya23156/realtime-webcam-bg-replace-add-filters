# Real-time Webcam Background Replacement Web Application

Zoom-like feature: Real-time webcam background replacement with a web UI + Cartoonification + Image filters implemented with FastAPI using WebSocket. (Also, utilizes JavaScript for frontend functionalities)

## Demo:
YouTube video: <link> here

## Features:
- Replace the webcam background with a selected preloaded image or one uploaded by the user.
- Two available models for background segmentation: Mediapipe (default) and Hugging Face (cannot be selected from the UI, but from code).
- Cartoonify webcam stream with two options: OpenCV (Sequence of image processings) and CartoonGAN (Deep learning model).
- Apply filters to the webcam stream. Available filters include Grayscale, Saturation, Brightness, Contrast.
- Supports concurrent connections.

## How to use:
1. Clone this repository.
2. Build the Docker image using `build.sh`.
3. Run the Docker container using `run.sh`.
4. Access port `localhost:8000` on a web browser.
5. That's it.

## Development Environments:
- Mac M1
- Docker version 24.0.6, build ed223bc
- Docker Compose version v2.21.0-desktop.1