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
def set_seed(seed):
    """Set all seeds to make results reproducible"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


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
                     max_length=MAX_LENGTH,
                     return_tensors='pt')


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = load_metric("accuracy")
    f1 = load_metric("f1")
    acc = acc.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = f1.compute(predictions=predictions, references=labels, average="macro")["f1"]

    return {"accuracy": acc, "f1": f1}


def delete_path(path):
    import shutil
    shutil.rmtree(path)    

    

class ModelTrainer:
    def __init__(self,
                 experiment_params,
                 run_name):
        self.params = experiment_params
        self.run_name = run_name

        if not os.path.exists(self.params["stage1"]["model_path"]):
            os.makedirs(self.params["stage1"]["model_path"])

        if not os.path.exists(self.params["stage2"]["model_path"]):
            os.makedirs(self.params["stage2"]["model_path"])

        self.label_map = {
            0: "entailment",
            1: "neutral",
            2: "contradiction"
        }

        self.seed = 2022
        set_seed(self.seed)


    def get_loss(self):
        """helper for selecting appropriate loss function"""

        if self.params["stage1"]["loss"] == "contrastive":
            loss = losses.ContrastiveLoss(self.model)

        elif self.params["stage1"]["loss"] == "online_contrastive":
            loss = losses.OnlineContrastiveLoss(self.model)

        elif self.params["stage1"]["loss"] == "mnr":
            loss = losses.MultipleNegativesRankingLoss(self.model)

        elif self.params["stage1"]["loss"] == "cosine":
            loss = losses.CosineSimilarityLoss(self.model)

        return loss


    def load_data(self, data_path, is_nli=False, round=None):
        """helper for loading data for training & testing"""

        if not is_nli:
            df = pd.read_parquet(data_path)
            samples = []

            for i in tqdm(range(len(df)), desc="Preparing data..."):
                if self.params["stage1"]["loss"] != "mnr":
                    samples.append(InputExample(texts=[df.loc[i, "sentence1"],
                                                       df.loc[i, "sentence2"]],
                                                label=df.loc[i, "label"]))
                else:
                    samples.append(InputExample(texts=[df.loc[i, "sentence1"],
                                                       df.loc[i, "sentence2"]]))

            if self.params["stage1"]["loss"] != "mnr":
                loader = DataLoader(samples,
                                    shuffle=True,
                                    batch_size=self.params["stage1"]["batch_size"])

            else:
                loader = sentence_transformers.datasets.NoDuplicatesDataLoader(samples,
                                                                               batch_size=self.params["stage1"]["batch_size"])

        else:
            df = pd.DataFrame(load_dataset('parquet', data_files=data_path, split='train').map(encode))
            
            if round is not None:
                df = df[df["round"] == round]
                df.reset_index(drop=True, inplace=True)

            loader = CustomDataset(df[['input_ids', 'token_type_ids', 'attention_mask']], df["label"])

        return loader


    def train_model(self,
                    mode="train",
                    do_stage1=False,
                    do_stage2=True,
                    use_wandb=False):
      
        """main routine for running training session"""

        if use_wandb:
            wandb.init(project=self.params["project_name"],
                        name=self.params["run_name"],
                        entity="deep_learning_project_597_bogazici",
                        notes = self.params["run_notes"])
            
        # Part 1 - fine tuning
        
        if do_stage1:
            self.loader = self.load_data(self.params["stage1"]["data_path"])
            deberta = models.Transformer("microsoft/deberta-v3-base", 
                                         max_seq_length=MAX_LENGTH)

            pooler = models.Pooling(
                deberta.get_word_embedding_dimension(),
                pooling_mode_cls_token=True
            ) if self.params["stage1"]["pooling_mode"] == "cls_token" else models.Pooling(
                deberta.get_word_embedding_dimension(),
                pooling_mode_mean_tokens=True
            )

            self.model = SentenceTransformer(modules=[deberta, pooler])
            self.stage1_loss = self.get_loss()

            self.model.fit(
                train_objectives=[(self.loader, self.stage1_loss)],
                epochs=self.params["stage1"]["n_epochs"],
                warmup_steps=int(len(self.loader) * self.params["stage1"]["n_epochs"] * 0.1),
                output_path=self.params["stage1"]["model_path"],
                show_progress_bar=True,
                use_amp=True,
                weight_decay=self.params["stage1"]["weight_decay"]
            )

            print("{} loss fine-tuning done.".format(self.params["stage1"]["loss"]))

            del self.model, self.stage1_loss, self.loader
            gc.collect()
            torch.cuda.empty_cache()

        # Part 2 - NLI fine tuning

        if do_stage2:
            # NLI Softmax Loss read data
            self.train_dataset = self.load_data(self.params["stage2"]["train_data_path"],
                                                is_nli=True)

            self.eval_dataset = self.load_data(self.params["stage2"]["val_data_path"],
                                                is_nli=True)

            # load model
            if do_stage1:
                model_path = self.params["stage1"]["model_name"]
            else:
                model_path = self.params["stage2"]["model_name"]

            config = AutoConfig.from_pretrained(
                model_path,
                num_labels=3
            )

            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
            )

            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                config=config,
            )

            training_args = TrainingArguments(
                output_dir=self.params["stage2"]["model_path"],
                num_train_epochs=self.params["stage2"]["n_epochs"],
                per_device_train_batch_size=self.params["stage2"]["batch_size"],
                per_device_eval_batch_size=self.params["stage2"]["batch_size"],
                learning_rate=self.params["stage2"]["learning_rate"],
                warmup_ratio=self.params["stage2"]["warmup_ratio"],
                weight_decay=self.params["stage2"]["weight_decay"],
                logging_steps=self.params["stage2"]["eval_steps"],
                report_to="wandb",
                evaluation_strategy="steps",
                eval_steps=self.params["stage2"]["eval_steps"],
                run_name=self.params["run_name"],
                disable_tqdm=False,
                fp16=True,
                seed=self.seed,
                log_level='error',
                load_best_model_at_end=True,
                save_total_limit=self.params["stage2"]["save_total_limit"],
                save_strategy="steps",
                save_steps=self.params["stage2"]["eval_steps"],
                callbacks=[EarlyStoppingCallback(early_stopping_patience=1, 
                                                 early_stopping_threshold = 0.5)]
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                compute_metrics=compute_metrics,
            )

            trainer.train()
            torch.cuda.empty_cache()
            print("NLI training done.")

            if not do_stage1:
                del self.params["stage1"]
        
        if use_wandb:
            wandb.finish()
        
        # save experiment configuration

        with open(self.params["config_path"], 'w') as f:
            for key, value in self.params.items():
                f.write('%s:%s\n' % (key, value))


# environment & wandb variables
BASE_PATH = "/content/drive/MyDrive/deep_learning_project/"
project_name = "softmax"
run_name = "softmax_exp4"
run_notes = "Softmax Loss, deberta-v3-large, 3 epochs"
stage1_loss = ""
MAX_LENGTH = 128
model_name = "microsoft/deberta-v3-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# '/content/drive/MyDrive/deep_learning_project/online_contrastive_exp2/online_contrastive_loss_model'


# experiment parameters
params = {
    "project_name": project_name,
    "run_name": run_name,
    "run_notes": run_notes,
    "config_path":os.path.join(BASE_PATH, run_name, "experiment_config.txt"),
    "max_len": MAX_LENGTH,
    "dataset_metadata": "/content/drive/MyDrive/deep_learning_project/data/final_metadata.joblib",
    
  "stage1": {
      "batch_size": 32,
      "n_epochs": 5,
      "learning_rate": 2e-5,
      "loss": stage1_loss,
      "data_path": os.path.join(BASE_PATH, "data/stage1_train.parquet"),
      "model_path": os.path.join(BASE_PATH, run_name, "{}_loss_model".format(stage1_loss)),
      "maxlen": MAX_LENGTH,
      "pooling_mode": "mean_token",
      "weight_decay": 0.05,
      "model_name": model_name
  },

  "stage2": {
      "batch_size": 64, #64
      "n_epochs": 10,
      "learning_rate": 2e-5,
      "train_data_path": os.path.join(BASE_PATH, "data/stage2_train.parquet"),
      "val_data_path": os.path.join(BASE_PATH, "data/stage2_val.parquet"),
      "test_data_path": os.path.join(BASE_PATH, "data/stage2_test.parquet"),
      "model_path": os.path.join(BASE_PATH, run_name, "nli_model"),
      "warmup_ratio": 0.1,
      "weight_decay": 0.05,
      "eval_steps": 15750,
      "save_total_limit": 1,
      "model_name": model_name
  }
}

# current step: Gradual unfreezing layers for stage2 (using stage1 model directly.)
# next step: Gradual unfreezing for both models

model_trainer = ModelTrainer(params, run_name)
model_trainer.train_model(do_stage1=False, do_stage2=True, use_wandb=True)