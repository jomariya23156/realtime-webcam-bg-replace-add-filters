import io
import asyncio
import contextlib
from pathlib import Path

import cv2
import base64
import torch
import logging
import threading
import numpy as np
from io import BytesIO
from PIL import Image
from fastapi import (FastAPI, Request, WebSocket, WebSocketDisconnect, 
                     UploadFile, File)
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from pred_models import ObjectSegmentation, MPSelfieSegmentation, CartoonGAN
from pydantic_models import Message
from typing import List
from utils import (setup_logger, bin_mask_from_cls_idx, array_to_encoded_str,
                   cartoonify_img)

model_source = 'mediapipe' # 'mediapipe' or 'hugging_face'
if model_source == 'mediapipe':
    pred_model = MPSelfieSegmentation()
elif model_source == 'hugging_face':
    # pred_model = ObjectDetection()
    pred_model = ObjectSegmentation()

cartoonifier = CartoonGAN()
cartoonifier_type = 'fp16'
cartoonify = None
cartoonify_options = ['tflite', 'opencv', 'disable']

### Hardcoded for now, later this will receive from UI ###
keep_obj_idxs = [8, 15]

logger = setup_logger()
lock = threading.Lock()
bg_img = None

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    pred_model.load_model()
    cartoonifier.load_model(model_type=cartoonifier_type)
    yield

app = FastAPI(lifespan=lifespan)

origins = [
    "http://localhost",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# setup static and template
app.mount('/static', StaticFiles(directory='static'), name='static')
templates = Jinja2Templates(directory='templates')

@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})

@app.post('/change_bg', response_model=Message, responses={404: {"model": Message}})
async def change_bg(request: Request, file: UploadFile = File(...)):
    global bg_img
    logger.info('CHANGE BG REQUEST')
    try:
        image_bytes = await file.read()
        image_array = np.frombuffer(image_bytes, np.uint8)
        new_bg = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if new_bg is None:
            raise ValueError("Reading input image return None")
        with lock:
            bg_img = new_bg
        resp_message = "Successfully updated the background image"
        return {"message": resp_message}
    except Exception as e:
        logger.exception(f'Reading image file failed with exception:\n {e}')
        resp_code = 404
        resp_message = "Reading input image file failed. Incorrect or unsupported image types."
        return JSONResponse(status_code=resp_code, content={"message": resp_message})
    
@app.put('/change_cartoonify/{option}', response_model=Message, responses={404: {"model": Message}})
async def change_cartoonify(option: str):
    global cartoonify
    logger.info('CHANGE CARTOONIFY OPTION REQUEST')
    if option not in cartoonify_options:
        resp_code = 404
        resp_message = f'Specified option is invalid. Available options: {cartoonify_options}'
        return JSONResponse(status_code=resp_code, content={"message": resp_message})
    with lock:
        cartoonify = option
    resp_message = "Successfully updated the cartoonify option"
    return {"message": resp_message}

async def receive(websocket: WebSocket, queue: asyncio.Queue):
    while True:
        bytes = await websocket.receive_bytes()
        try:
            queue.put_nowait(bytes)
            # logger.info('Added to queue')
        except asyncio.QueueFull:
            pass

async def detect(websocket: WebSocket, queue: asyncio.Queue):
    global bg_img
    while True:
        bytes = await queue.get()
        # image = Image.open(io.BytesIO(bytes))
        image_array = np.frombuffer(bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if bg_img is not None:
            # run a prediction
            if model_source == 'mediapipe':
                selected_mask = pred_model.predict(image)
            elif model_source == 'hugging_face':
                all_mask = pred_model.predict(image)
                selected_mask = bin_mask_from_cls_idx(all_mask, keep_obj_idxs)
            # replace background
            if bg_img.shape[:2] != image.shape[:2]:
                with lock:
                    bg_img = cv2.resize(bg_img, (image.shape[1], image.shape[0]))
            # bg2replace = np.random.randint(0, 255, size=(image.shape[0], image.shape[1], 1), dtype=np.uint8)
            final_img = np.where(np.expand_dims(selected_mask, 2), image, bg_img)
        else:
            final_img = image.copy()
        if cartoonify in ['opencv', 'tflite']:
            final_img = cartoonify_img(final_img, option=cartoonify, cartoonifier=cartoonifier)
        # encode image to base64
        final_img_str = array_to_encoded_str(final_img)
        b64_src = "data:image/jpg;base64,"
        processed_img_data = b64_src + final_img_str
        await websocket.send_text(processed_img_data)

@app.websocket('/object-detection')
async def ws_object_detection(websocket: WebSocket):
    await websocket.accept()
    queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    receive_task = asyncio.create_task(receive(websocket, queue))
    detect_task = asyncio.create_task(detect(websocket, queue))
    try:
        done, pending = await asyncio.wait(
            {receive_task, detect_task},
            return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        for task in done:
            task.result()
    except WebSocketDisconnect:
        pass