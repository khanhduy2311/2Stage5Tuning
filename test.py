from importlib.metadata import files
from datasets import load_dataset, load_metric
from pprint import pprint
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import gc
import glob
import joblib
import json
import numpy as np
import os
import pandas as pd
import random
import re
import torch
import ast
import warnings

try:
    from sentence_transformers import models, SentenceTransformer, losses, InputExample
    from transformers import AutoModel, AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoConfig, AutoTokenizer, EarlyStoppingCallback
    import sentence_transformers
    import transformers
    transformers.logging.set_verbosity_error()
    import wandb
except:
    pass
def evaluate_model(data_loader, model_path, tokenizer_name='microsoft/deberta-v3-base'):
    """helper for evaluating model"""
    # Load trained model
    #model_checkpoint = "/content/drive/MyDrive/deep_learning_project/baseline_model/checkpoint-26500"
    model = AutoModelForSequenceClassification.from_pretrained(model_path,
                                                               num_labels=3)
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Define test trainer
    test_trainer = Trainer(model)
    
    # Make prediction
    raw_pred, _, _ = test_trainer.predict(data_loader)

    # Preprocess raw predictions
    y_pred = np.argmax(raw_pred, axis=1)
    y_true = pd.read_parquet(data_path)["label"].values
    print("Accuracy score: {}", round(accuracy_score(y_true, y_pred)), 3)
    print("F1 score: {}", round(f1_score(y_true, y_pred, average='macro'), 3))


evaluate_model(test_dataset, "/content/drive/MyDrive/deep_learning_project/online_contrastive_exp1/nli_model/checkpoint-60000")