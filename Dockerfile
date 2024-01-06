FROM python:3.10.13-slim

ARG SERVICE_PORT=$SERVICE_PORT

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        # for psutil (required by hugging face libs)
        gcc \
        python3-dev \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /service

COPY requirements.txt .

RUN pip install -r requirements.txt && \
    # some libs in requirements.txt require opencv-python, some require opencv-contrib-python
    # this results in inconsistent and multiple versions of opencv installed
    # so 2 lines here are for cleaning up and install only 1 opencv
    pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless && \
    pip install opencv-python-headless==4.8.0.74

COPY app/ app/

EXPOSE $SERVICE_PORT

WORKDIR /service/app

CMD gunicorn main:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:${SERVICE_PORT}