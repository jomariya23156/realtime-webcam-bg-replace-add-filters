import io
import asyncio
import contextlib
from pathlib import Path
# for some reasons, sklearn needed to be imported before other packages
from sklearn.cluster import MiniBatchKMeans

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

from pred_models import (ObjectDetection, ObjectSegmentation, 
                         MPSelfieSegmentation, CartoonGAN)
from pydantic_models import Object, Objects, Message
from typing import List

import mediapipe as mp


def setup_logger():
    FORMAT = logging.Formatter(
    '%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)-3d | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    print(f'Created logger with name {__name__}')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setFormatter(FORMAT)
    logger.addHandler(ch)
    return logger

def bin_mask_from_cls_idx(full_mask: torch.Tensor, cls_idx_list: List[int]) -> torch.Tensor:
    mask = full_mask.clone()
    for i in cls_idx_list:
        mask[mask==i] = 255
    mask[mask!=255] = 0
    return mask

def array_to_encoded_str(image: np.ndarray) -> str:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    result, byte_data = cv2.imencode(".jpg", image, encode_param)
    img_str = base64.encodebytes(byte_data).decode('utf-8')
    return img_str

def quantize_img_color(img: np.ndarray, n_clusters: int=20) -> np.ndarray:
    image = img.copy()
    (h, w) = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # reshape the image into a feature vector so that k-means
    # can be applied
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    # apply k-means using the specified number of clusters and
    # then create the quantized image based on the predictions
    clt = MiniBatchKMeans(n_clusters = n_clusters)
    labels = clt.fit_predict(image)
    quant = clt.cluster_centers_.astype("uint8")[labels]
    # reshape the feature vectors to images
    quant = quant.reshape((h, w, 3))
    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
    return quant

def cartoonify_img(image: np.ndarray, option: str='disable') -> np.ndarray:
    if option == 'opencv':
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # applying median blur to smoothen an image
        smooth_gray = cv2.medianBlur(gray, 5)
        # retrieving the edges for cartoon effect
        edge_img = cv2.adaptiveThreshold(smooth_gray, 255, 
                                        cv2.ADAPTIVE_THRESH_MEAN_C, 
                                        cv2.THRESH_BINARY, 15, 9)
        # applying bilateral filter to remove noise
        color_img = cv2.bilateralFilter(image, 9, 50, 50)
        # color_img = quantize_img_color(color_img)
        # masking edged image with our "BEAUTIFY" image
        cartoonified = cv2.bitwise_and(color_img, color_img, mask=edge_img)
        return cartoonified
    elif option == 'tflite':
        ori_size = image.shape[:2]
        cartoonified = cartoonifier.predict(image)
        cartoonified = cv2.resize(cartoonified, (ori_size[1], ori_size[0]))
        return cartoonified
    else:
        return image

model_source = 'mediapipe' # 'mediapipe' or 'hugging_face'
if model_source == 'mediapipe':
    pred_model = MPSelfieSegmentation()
elif model_source == 'hugging_face':
    # pred_model = ObjectDetection()
    pred_model = ObjectSegmentation()

cartoonify = 'tflite' # options: opencv, tflite
cartoonifier = CartoonGAN()
cartoonifier_type = 'fp16'

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
            final_img = cartoonify_img(final_img, option=cartoonify)
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