import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
import torch
import evaluate
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from utils.s3 import *


def TrainTinyBertSentimentAnalysis(
    run_id: str
    , num_train_epochs: int
    , learning_rate: float
    , batch_size: int
    , s3_result_uri: str
    , tmp_path: str = "./"
    , model_name: str = "tinybert-sentiment-analysis"
):

    # ----------------------------
    # Data loading
    # ----------------------------

    data = pd.read_csv('IMDB-Dataset.csv')
    dataset = Dataset.from_pandas(data)

    # ----------------------------
    # Data processing
    # ----------------------------

    dataset = dataset.train_test_split(test_size=0.3)
    label2id = {'negative': 0, 'positive': 1}
    id2label = {0:'negative', 1:'positive'}
    dataset = dataset.map(lambda x: {'label': label2id[x['sentiment']]})

    # Data tokenization
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model_ckpt = 'huawei-noah/TinyBERT_General_4L_312D'
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt, use_fast=True)

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

    TrainTinyBertSentimentAnalysis(
        run_id = "20250206", 
        num_train_epochs = 10, 
        learning_rate = 2e-5, 
        batch_size = 512, 
        s3_result_uri = "s3://mlops-yang-l-20250206/ml-models"
    )