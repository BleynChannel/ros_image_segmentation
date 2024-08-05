FROM bleyn/paddle-ros:2.6.1-gpu-cuda12.0-cudnn8.9-trt8.6-noetic

COPY ./app /app
WORKDIR /app
RUN pip install -r /app/requirements.txt