import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
import torch
import evaluate
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from utils.s3 import *


def TrainTinyBertDisasterTweet(
    run_id: str
    , num_train_epochs: int
    , learning_rate: float
    , batch_size: int
    , s3_result_uri: str
    , tmp_path: str = "./"
    , model_name: str = "tinybert-disaster-tweet"
):

    # ----------------------------
    # Data loading
    # ----------------------------

    df = pd.read_csv("twitter_disaster_tweets.csv", usecols=["text", "target"])

    # ----------------------------
    # Data processing
    # ----------------------------

    df = df.sample(frac=1).reset_index(drop=True)
    df = df.rename(columns={"target": "label"})
    dataset = Dataset.from_pandas(df)
    dataset = dataset.train_test_split(test_size=0.2)

    id2label = {0: "general", 1: "disaster"}
    label2id = {id2label[id]: id for id in id2label}

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_ckpt = "huawei-noah/TinyBERT_General_4L_312D"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt, use_fast=True)

    def tokenize(batch):
        temp = tokenizer(batch["text"], padding=True, truncation=True, max_length=100) 
        # truncation: if length larger than context window, truncate it
        return temp
    dataset = dataset.map(tokenize, batched=True, batch_size=2000)


    # ----------------------------
    # Prepare training
    # ----------------------------

    accuracy = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    model = AutoModelForSequenceClassification.from_pretrained(
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
        tokenizer=tokenizer  # store tokenizer together with model for later use consistency
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

    TrainTinyBertDisasterTweet(
        run_id = "20250207", 
        num_train_epochs = 5, 
        learning_rate = 1e-5, 
        batch_size = 256, 
        s3_result_uri = "s3://mlops-yang-l-20250206/ml-models"
    )