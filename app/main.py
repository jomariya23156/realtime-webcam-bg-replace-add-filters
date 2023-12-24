import io
import asyncio
import contextlib
from pathlib import Path

import cv2
import base64
import torch
import numpy as np
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pred_models import ObjectDetection, ObjectSegmentation, MPSelfieSegmentation
from pydantic_models import Object, Objects
from typing import List

import mediapipe as mp

def bin_mask_from_cls_idx(full_mask: torch.Tensor, cls_idx_list: List[int]) -> torch.Tensor:
    mask = full_mask.clone()
    for i in cls_idx_list:
        mask[mask==i] = 255
    mask[mask!=255] = 0
    return mask

def array_to_encoded_str(image: np.ndarray):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    result, byte_data = cv2.imencode(".jpg", image, encode_param)
    img_str = base64.encodebytes(byte_data).decode('utf-8')
    return img_str

model_source = 'mediapipe' # 'mediapipe' or 'hugging_face'
if model_source == 'mediapipe':
    pred_model = MPSelfieSegmentation()
elif model_source == 'hugging_face':
    # pred_model = ObjectDetection()
    pred_model = ObjectSegmentation()

### Hardcoded for now, later this will receive from UI ###
keep_obj_idxs = [8, 15]

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    pred_model.load_model()
    yield

app = FastAPI(lifespan=lifespan)

# setup static and template
app.mount('/static', StaticFiles(directory='static'), name='static')
templates = Jinja2Templates(directory='templates')

@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})

async def receive(websocket: WebSocket, queue: asyncio.Queue):
    while True:
        bytes = await websocket.receive_bytes()
        try:
            queue.put_nowait(bytes)
            print('Added to queue')
        except asyncio.QueueFull:
            pass

async def detect(websocket: WebSocket, queue: asyncio.Queue):
    while True:
        bytes = await queue.get()
        # image = Image.open(io.BytesIO(bytes))
        image_array = np.frombuffer(bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        # run a prediction
        if model_source == 'mediapipe':
            selected_mask = pred_model.predict(image)
        elif model_source == 'hugging_face':
            all_mask = pred_model.predict(image)
            selected_mask = bin_mask_from_cls_idx(all_mask, keep_obj_idxs)
        # replace background
        bg_image = np.random.randint(0, 255, size=(image.shape[0], image.shape[1], 1), dtype=np.uint8)
        final_img = np.where(np.expand_dims(selected_mask, 2), image, bg_image)
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