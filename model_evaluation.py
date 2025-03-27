from datasets import load_dataset, load_metric
from sentence_transformers import InputExample, models, SentenceTransformer, losses
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import AutoModel, AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoConfig, AutoTokenizer
import numpy as np
import pandas as pd
import torch
import transformers
transformers.logging.set_verbosity_error()

class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        self.required_cols = ['input_ids', 'token_type_ids', 'attention_mask']

    def __getitem__(self, idx):
        item = {col: torch.tensor(self.encodings.loc[idx, col][0]) for col in self.encodings.columns}
        item['labels'] = torch.tensor(self.labels.loc[idx])
        return item

    def __len__(self):
        return len(self.labels)


def encode(examples):
    return tokenizer(examples["sentence1"],
                     examples["sentence2"],
                     padding='max_length',  # Pad to max_length
                     truncation=True,  # Truncate to max_length
                     max_length=100,
                     return_tensors='pt')


tokenizer = AutoTokenizer.from_pretrained(
    'microsoft/deberta-v3-base',
)

data_path = '/path/to/data'
test_dataset = load_dataset('parquet', data_files=data_path)
test_df = pd.DataFrame(test_dataset.map(encode)["train"])
test_dataset = CustomDataset(test_df[['input_ids', 'token_type_ids', 'attention_mask']], test_df["label"])
y_true = pd.read_parquet(data_path)