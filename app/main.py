import asyncio
import contextlib

import os
import cv2
import threading
import numpy as np
from websockets.exceptions import ConnectionClosed
from fastapi import (FastAPI, Request, WebSocket, WebSocketDisconnect, 
                     UploadFile, File, Depends, Header)
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from pred_models import ObjectSegmentation, MPSelfieSegmentation, CartoonGAN
from pydantic_models import Message
from utils import (setup_logger, bin_mask_from_cls_idx, array_to_encoded_str,
                   cartoonify_img)

logger = setup_logger()
lock = threading.Lock()

# session variables / states
# consider using in-memory database like Redis or memcached 
# instead of this variable in the production environment
session_data = {}

# dependencies injection
def get_session_token(x_session_token: str = Header(...)):
    return x_session_token

def get_current_session_vars(token: str = Depends(get_session_token)):
    # Check if session exists, otherwise create a new one
    if token not in session_data:
        session_data[token] = {"bg_img": None, "cartoonify": None}
    return session_data[token]

model_source = 'mediapipe' # 'mediapipe' or 'hugging_face'
# keep_obj_idxs -> only for 'hugging_face', 
# use to set class idices to include in mask segmentation
keep_obj_idxs = None 

# bg segmentation model
if model_source == 'mediapipe':
    pred_model = MPSelfieSegmentation()
elif model_source == 'hugging_face':
    pred_model = ObjectSegmentation()
    keep_obj_idxs = [8, 15]

cartoonifier = CartoonGAN()
cartoonifier_type = 'dr'
cartoonify_options = ['cartoongan', 'opencv', 'disable']

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

# icon source: https://icons8.com/icon/aMYEhwmQ2nm5/video-camera
@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    file_name = "favicon.ico"
    file_path = os.path.join(app.root_path, "static", "assets", file_name)
    return FileResponse(file_path)

@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})

@app.post('/reset_bg', response_model=Message, responses={404: {"model": Message}})
async def change_bg(session_vars: str = Depends(get_current_session_vars)):
    logger.info('RESET BG REQUEST')
    with lock:
        session_vars['bg_img'] = None
    resp_message = "Successfully reseted the background image"
    return {"message": resp_message}

@app.post('/change_bg', response_model=Message, responses={404: {"model": Message}})
async def change_bg(request: Request, file: UploadFile = File(...), 
                    session_vars: str = Depends(get_current_session_vars)):
    logger.info('CHANGE BG REQUEST')
    try:
        image_bytes = await file.read()
        image_array = np.frombuffer(image_bytes, np.uint8)
        new_bg = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if new_bg is None:
            raise ValueError("Reading input image return None")
        with lock:
            session_vars['bg_img'] = new_bg
        resp_message = "Successfully updated the background image"
        return {"message": resp_message}
    except Exception as e:
        logger.exception(f'Reading image file failed with exception:\n {e}')
        resp_code = 404
        resp_message = "Reading input image file failed. Incorrect or unsupported image types."
        return JSONResponse(status_code=resp_code, content={"message": resp_message})
    
@app.put('/change_cartoonify/{option}', response_model=Message, responses={404: {"model": Message}})
async def change_cartoonify(option: str, session_vars: str = Depends(get_current_session_vars)):
    logger.info('CHANGE CARTOONIFY OPTION REQUEST')
    if option not in cartoonify_options:
        resp_code = 404
        resp_message = f'Specified option is invalid. Available options: {cartoonify_options}'
        return JSONResponse(status_code=resp_code, content={"message": resp_message})
    with lock:
        session_vars['cartoonify'] = option
    resp_message = "Successfully updated the cartoonify option"
    return {"message": resp_message}

async def receive(websocket: WebSocket, queue: asyncio.Queue):
    while True:
        bytes = await websocket.receive_bytes() 
        try:
            queue.put_nowait(bytes)
        except asyncio.QueueFull:
            pass

async def process(websocket: WebSocket, queue: asyncio.Queue, session_vars: dict):
    while True:
        bytes = await queue.get()
        image_array = np.frombuffer(bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if session_vars['bg_img'] is not None:
            # run a prediction
            if model_source == 'mediapipe':
                selected_mask = pred_model.predict(image)
            elif model_source == 'hugging_face':
                all_mask = pred_model.predict(image)
                selected_mask = bin_mask_from_cls_idx(all_mask, keep_obj_idxs)
            # replace background
            if session_vars['bg_img'].shape[:2] != image.shape[:2]:
                with lock:
                    session_vars['bg_img'] = cv2.resize(session_vars['bg_img'], (image.shape[1], image.shape[0]))
            final_img = np.where(np.expand_dims(selected_mask, 2), image, session_vars['bg_img'])
        else:
            final_img = image.copy()
        if session_vars['cartoonify'] in ['opencv', 'cartoongan']:
            final_img = cartoonify_img(final_img, option=session_vars['cartoonify'], cartoonifier=cartoonifier)
        # encode image to base64
        final_img_str = array_to_encoded_str(final_img)
        b64_src = "data:image/jpg;base64,"
        processed_img_data = b64_src + final_img_str
        await websocket.send_text(processed_img_data)

@app.websocket('/image_processing/{session_token}')
async def ws_image_processing(websocket: WebSocket, session_token: str):
    await websocket.accept()
    logger.info(f'session token: {session_token}')
    session_vars = get_current_session_vars(session_token)
    queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    receive_task = asyncio.create_task(receive(websocket, queue))
    process_task = asyncio.create_task(process(websocket, queue, session_vars))
    try:
        done, pending = await asyncio.wait(
            {receive_task, process_task},
            return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        for task in done:
            task.result()
    except (WebSocketDisconnect, ConnectionClosed):
        logger.info('User disconnected')
        # Remove session data when user disconnects
        del session_data[session_token]
        logger.info(f'Removed session data of session token: {session_token}')