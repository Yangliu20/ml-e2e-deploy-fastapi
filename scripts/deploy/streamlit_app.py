# python -m streamlit run streamlit_app.py --server.enableXsrfProtection false --server.port 8053

import streamlit as st
import requests
from scripts import s3
import json
import datetime

bucket_name = "mlops-yang-l-20250206"

# Define API endpoint
API_URL = "http://127.0.0.1:8000/api/v1"
headers = {
    "Content-Type": "application/json"
}

st.title("ML Model Serving Over REST API")

model = st.selectbox("Select Model", [
    "Sentiment Classifier", 
    "Disaster Classifier", 
    "Pose Classifier"
])

if model ==  "Sentiment Classifier":
    text = st.text_area("Enter your movie review")
    user_id = st.text_input("Enter your email address")
    data = {"text": [text], "user_id": user_id}
    model_api = API_URL+"/sentiment_analysis"


if model ==  "Disaster Classifier":
    text = st.text_area("Enter your twitter")
    user_id = st.text_input("Enter your email address")
    data = {"text": [text], "user_id": user_id}
    model_api = API_URL+"/disaster_classifier"


if model ==  "Pose Classifier":
    select_file_type = st.radio("Select the image source", ["Local", "URL"])
    if select_file_type == "URL": 
        url = st.text_input("Enter your image URL")
    else:
        image = st.file_uploader("Upload the image", type=["jpg", "jpeg", "png"])
        # save in a temp local path
        file_name = f"temp_images/image_{str(datetime.datetime.utcnow())}.jpg"
        # print(file_name)
        if image is not None:
            with open(file_name, "wb") as f:
                f.write(image.read())
            url = s3.upload_image_to_s3(bucket_name, file_name)
            # print(url)

    user_id = st.text_input("Enter your email address")
    data = {"url": [url], "user_id": user_id}
    model_api = API_URL+"/pose_classifier"


if st.button("Predict"):
    with st.spinner("Predicting..."):
        response = requests.request("POST", model_api, headers=headers, data=json.dumps(data))
        output = response.json()

    st.write(output)
