from utils.data_model import *
from utils import s3

from fastapi import FastAPI
# from fastapi import Request
import uvicorn

import os
from transformers import pipeline
import torch

from PIL import Image
import requests
import socket

import time

app = FastAPI()


#### Download models from s3 to local ####

force_download = True
bucket_name = "mlops-yang-l-20250206"

model_name_dict = {
    "disaster": "tinybert-disaster-tweet-20250207", 
    "sentiment": "tinybert-sentiment-analysis-20250206", 
    "pose": "vit-human-pose-classification-20250208-v2"
}

for model_name in model_name_dict.values():
    local_path = f'models-download/{model_name}'
    s3_prefix = f'ml-models/{model_name}'
    if not os.path.isdir(local_path) or force_download:
        s3.download_dir(bucket_name, local_path, s3_prefix)
        print(f"Download {model_name} finished. ")
    print(f"{model_name} exist in local. ")

#### Download ends here ####


#### Load models ####

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
sentiment_model = pipeline("text-classification", model=f'ml-models-download/{model_name_dict["sentiment"]}', device=device)
disaster_model = pipeline("text-classification", model=f'ml-models-download/{model_name_dict["disaster"]}', device=device)
pose_model = pipeline("image-classification", model=f'ml-models-download/{model_name_dict["pose"]}', device=device)

#### Model loading ends here ####


@app.get("/")
def root():
    return f"Hello! xD Welcome to ML model production api. I am {socket.gethostname()}. "


@app.post("/api/v1/sentiment_analysis")
def sentiment_analysis(data: NLPDataInput):
    start = time.time()
    output = sentiment_model(data.text)
    end = time.time()
    prediction_time = end - start

    labels = [x["label"] for x in output]
    scores = [x["score"] for x in output]

    output_final = NLPDataOutput(
        model_name = model_name_dict["sentiment"], 
        text = data.text, 
        labels = labels, 
        scores = scores, 
        prediction_time = prediction_time
    )

    return output_final

@app.post("/api/v1/disaster_classifier")
def disaster_classifier(data: NLPDataInput):
    start = time.time()
    output = disaster_model(data.text)
    end = time.time()
    prediction_time = end - start

    labels = [x["label"] for x in output]
    scores = [x["score"] for x in output]

    output_final = NLPDataOutput(
        model_name = model_name_dict["disaster"], 
        text = data.text, 
        labels = labels, 
        scores = scores, 
        prediction_time = prediction_time
    )

    return output_final

@app.post("/api/v1/pose_classifier")
def pose_classifier(data: ImageDataInput):
    """
    Sample image url: [
        "https://img.freepik.com/free-photo/full-shot-ballerina-dancing_23-2149269656.jpg", 
        "https://mir-s3-cdn-cf.behance.net/project_modules/fs/2228d596410707.5eadbe1428060.jpg", 
        "https://www.emmasedition.com/wp-content/uploads/2019/12/How-to-Pose-Sitting-Down-2.jpg"
    ], 
    """
    start = time.time()
    image = [Image.open(requests.get(url, stream=True).raw) for url in data.url]
    output = pose_model(image)
    end = time.time()
    prediction_time = end - start

    labels = [x[0]["label"] for x in output]
    scores = [x[0]["score"] for x in output]

    output_final = ImageDataOutput(
        model_name = model_name_dict["pose"], 
        url = data.url, 
        labels = labels, 
        scores = scores, 
        prediction_time = prediction_time
    )

    return output_final


if __name__ == "__main__":
    uvicorn.run(app="app:app", port=8000, reload=False)