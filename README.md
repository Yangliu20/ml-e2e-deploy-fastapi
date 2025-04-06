# Deployment of BERT and ViT models for real-time prediction as REST API

This project builds an end-to-end ML pipeline to serve 3 **finetuned models** (BERT and ViT) as a REST API on **AWS EC2** using **Python** and **FastAPI**, achieving sentiment analysis, disaster tweet classification, and human pose classification. 

To achieve this, the pretrained BERT and ViT models are finetuned with use-specific datasets and uploade to AWS S3 bucket. The model is served as a REST API using FastAPI, and an interactive **Streamlit** web application is also developed to serve the model by API requests through a user-friendly interface. The application is containerized with **Docker** and leverages **Nginx** for load balancing, thus achieving real-time prediction. 

## Technical implementation

- PyTorch, HuggingFace
- AWS (s3, EC2)
- FastAPI, Streamlit
- Git, Docker, Nginx

## High-level diagram of the pipeline design 

<img src="https://github.com/user-attachments/assets/8842f732-c12c-4ce7-9e2f-5109e44a3007" alt="pipeline design" width=400/>

## Demo of Streamlit interface

<img src="https://github.com/user-attachments/assets/de4baeaa-0025-4e26-b519-ca15e5620b4a" alt="streamlit demo" width=400/>
