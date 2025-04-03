import pandas as pd
from datasets import Dataset
from datasets import load_dataset
from transformers import AutoImageProcessor
import torch
import evaluate
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer
import numpy as np
from utils.s3 import *


def TrainVitPoseClassification(
    run_id: str
    , num_train_epochs: int
    , learning_rate: float
    , batch_size: int
    , s3_result_uri: str
    , tmp_path: str = "./"
    , model_name: str = "vit-human-pose-classification"
):

    # ----------------------------
    # Data loading
    # ----------------------------

    dataset = load_dataset("Bingsu/Human_Action_Recognition", split="train")

    # ----------------------------
    # Data processing
    # ----------------------------

    dataset = dataset.shuffle(seed=617)
    dataset = dataset.train_test_split(test_size=0.2)

    labels = dataset["train"].features["labels"].names
    label2id = {x: i for i, x in enumerate(labels)}
    id2label = {i: x for i, x in enumerate(labels)}

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_ckpt = "google/vit-base-patch16-224-in21k"
    image_processor = AutoImageProcessor.from_pretrained(model_ckpt, use_fast=True)

    def process_image(batch):
        temp = image_processor(batch["image"]) 
        return temp
    
    dataset = dataset.map(process_image, batched=True, batch_size=200)


    # ----------------------------
    # Prepare training
    # ----------------------------

    accuracy = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    model = AutoModelForImageClassification.from_pretrained(
        model_ckpt, 
        num_labels=len(label2id), 
        label2id=label2id, 
        id2label=id2label
    ).to(device)
    print(model)

    args = TrainingArguments(
        output_dir = f"{tmp_path}/train_dir", 
        overwrite_output_dir = True, 
        num_train_epochs = num_train_epochs, 
        learning_rate = learning_rate, 
        per_device_train_batch_size = batch_size, 
        per_device_eval_batch_size = batch_size, 
        evaluation_strategy = "epoch"
    )

    trainer = Trainer(
        model=model, 
        args=args, 
        train_dataset=dataset["train"], 
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics, 
        processing_class=image_processor
    )


    # ----------------------------
    # Model training
    # ----------------------------

    train_output = trainer.train()
    eval_output = trainer.evaluate()
    print("Training output", train_output)
    print("Evaluate output", eval_output)
    trainer.save_model(f"{tmp_path}/{model_name}-{run_id}")


    # ----------------------------
    # Upload model to s3
    # ----------------------------

    s3_path_splits = s3_result_uri.replace("s3://", "").split("/")
    bucket_name = s3_path_splits.pop(0)
    s3_key = "/".join(s3_path_splits)

    upload_dir(bucket_name, f"{tmp_path}/{model_name}-{run_id}", f'{s3_key}/{model_name}-{run_id}')
    print("Model saved to s3", f'{s3_key}/{model_name}-{run_id}')

    return 


if __name__ == "__main__":

    TrainVitPoseClassification(
        run_id = "20250208-v2", 
        num_train_epochs = 5, 
        learning_rate = 2e-5, 
        batch_size = 256, 
        s3_result_uri = "s3://mlops-yang-l-20250206/ml-models"
    )