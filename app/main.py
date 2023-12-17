import io
import asyncio
import contextlib
from pathlib import Path

import torch
from PIL import Image
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic_models import Object, Objects
from transformers import YolosForObjectDetection, YolosImageProcessor

class ObjectDetection:
    image_processor: YolosImageProcessor | None = None
    model: YolosForObjectDetection | None = None

    def load_model(self) -> None:
        self.image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")
        self.model = YolosForObjectDetection.from_pretrained("hustvl/yolos-tiny")

    def predict(self, image:Image.Image) -> Objects:
        if not self.image_processor or not self.model:
            raise RuntimeError("Model is not loaded")
        inputs = self.image_processor(images=image, return_tensors='pt')
        outputs = self.model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results = self.image_processor.post_process_object_detection(
            outputs, target_sizes=target_sizes
        )[0]

        objects: list[Object] = []
        for score, label, box in zip(results['scores'], results['labels'], results['boxes']):
            if score > 0.6:
                box_values = box.tolist()
                label = self.model.config.id2label[label.item()]
                print('Detected:', label)
                objects.append(Object(box=box_values, label=label))
        return Objects(objects=objects)
    
object_detection = ObjectDetection()

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    object_detection.load_model()
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
        except asyncio.QueueFull:
            pass

async def detect(websocket: WebSocket, queue: asyncio.Queue):
    while True:
        bytes = await queue.get()
        image = Image.open(io.BytesIO(bytes))
        objects = object_detection.predict(image)
        await websocket.send_json(objects.model_dump())

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