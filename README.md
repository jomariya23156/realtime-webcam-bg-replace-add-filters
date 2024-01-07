# Real-time Webcam Background Replacement Web Application

Zoom-like feature: Real-time webcam background replacement with a Web UI + Cartoonification + Image filters implemented with FastAPI using WebSocket. (Also, utilizes JavaScript for frontend functionalities)

## Demo:
YouTube video: https://youtu.be/00FC_3qZmZc  
*I'm running on Mac M1 (no gpu+recording). So, it looks quite delayed especially when applying the CartoonGAN (Deep Learning model). But in general, the model/method is fast and efficient enough to run in realtime. You can try it yourself :)  
<img src="./assets/preview.gif">

## Features:
- Replace the webcam background with a selected prepopulated image or one uploaded by the user.
- Two available models for background segmentation: **Mediapipe** (default) and **Hugging Face** (cannot be selected from the UI, but from code).
- Cartoonify webcam stream with two options: **OpenCV** (Sequence of image processings) and **CartoonGAN** (Deep learning model).
- Three available versions of CartoonGAN: **int8**, **dr**, and **fp16** (again, cannot be selected from the UI, but from code).
- Apply filters to the webcam stream. Available filters include Grayscale, Saturation, Brightness, Contrast.
- Supports concurrent connections.

## How to use:
1. Clone this repository. (and make sure you have Docker installed)
2. Build the Docker image using `build.sh`.
3. Run the Docker container using `run.sh`.
4. Access port `localhost:8000` on a web browser.
5. That's it.

## References:
- CartoonGAN: [link](https://www.kaggle.com/models/spsayakpaul/cartoongan/)
- OpenCV cartoonification: [link](https://www.analyticsvidhya.com/blog/2022/06/cartoonify-image-using-opencv-and-python/) 

## Development Environments:
- Mac M1
- Docker version 24.0.6, build ed223bc
- Docker Compose version v2.21.0-desktop.1