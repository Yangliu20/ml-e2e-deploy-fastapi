FROM python:3.9-slim

WORKDIR /app_on_container

COPY ../../scripts/deploy/requirements.txt /app_on_container/

RUN pip install -r requirements.txt 

RUN apt-get update && apt-get install -y curl

COPY ../../scripts/deploy /app_on_container

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]