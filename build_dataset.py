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
class DataOps:
    def __init__(self):
        self.label_map = {
            0: "entailment",
            1: "neutral",
            2: "contradiction"
        }

        self.dataset_metadata = {"stage2_datasets": ["anli",
                                                      "mnli",
                                                      "FEVER",
                                                      "conjNLI",
                                                      "EQUATE"],
                                 
                                 "stage1_dataset": ["mrpc",
                                                    "qnli",
                                                    "rte",
                                                    "stsb",
                                                    "wnli",
                                                    "paws",
                                                    "swag",
                                                    "qqp",
                                                    "art"],
                                 
                                 "label_map": self.label_map}


        self.mapper = {"premise": "sentence1", 
                       "hypothesis": "sentence2",
                       "label": "label",
                       "sentence_A": "sentence1",
                       "sentence_B": "sentence2",
                       "sentence1": "sentence1",
                       "sentence2": "sentence2",
                       "question": "sentence1",
                       "sentence": "sentence2",
                       "question1": "sentence1",
                       "question2": "sentence2"}


        self.final_col_names = ["sentence1", "sentence2", "label"]
        self.stage1_train_df = pd.DataFrame(columns=self.final_col_names)
        self.stage2_train_df = pd.DataFrame(columns=self.final_col_names)


    def get_anli_data(self):
        """helper for getting ANLI data (0-1-2)"""
        d = {}
        dataset = load_dataset('anli')
        
        # train data
        for feature in ["premise", "hypothesis", "label"]:
            data = []

            for split_type in ["train"]:
                for i in range(1, 4):
                    data += dataset[split_type + "_r" + str(i)][feature]
            
            d[self.mapper[feature]] = data
        
        self.stage2_train_df = pd.concat([self.stage2_train_df, 
                                          pd.DataFrame(d, columns=["sentence1", "sentence2", "label"])],
                                         ignore_index=True)

        # val data
        for feature in ["premise", "hypothesis", "label"]:
            data = []

            for split_type in ["dev"]:
                for i in range(1, 4):
                    data += dataset[split_type + "_r" + str(i)][feature]
            
            d[self.mapper[feature]] = data

        self.stage2_val_df = pd.DataFrame(d, columns=["sentence1", "sentence2", "label"])

         # test data
        premise_l = []
        hypo_l = []
        label_l = []

        for elem in ["test_r1", "test_r2", "test_r3"]:
            premise_l += zip(dataset[elem]["premise"], 
                             [elem for _ in range(len(dataset[elem]["premise"]))])
            
            hypo_l += dataset[elem]["hypothesis"]
            label_l += dataset[elem]["label"]


        self.stage2_test_df = pd.DataFrame(premise_l, columns= ["premise", "round"])
        self.stage2_test_df["hypothesis"] = hypo_l
        self.stage2_test_df["label"] = label_l
        self.stage2_test_df.rename(columns={"premise": "sentence1", "hypothesis": "sentence2"}, 
                                 inplace=True) 

    def get_mnli_data(self):
        """helper for getting MNLI data (0-1-2)"""
        d = {}
        dataset = load_dataset('glue', 'mnli')

        for feature in ["premise", "hypothesis", "label"]:
            data = []

            for split_type in ["train", "validation_matched", "validation_mismatched"]:
                data += dataset[split_type][feature]
            
            d[self.mapper[feature]] = data

        return pd.DataFrame(d, columns=["sentence1", "sentence2", "label"])


    def get_fever_data(self):
        """helper for getting FEVER-NLI dataset"""

        with open('/content/train_fitems.jsonl', 'r') as json_file:
            json_list = list(json_file)

        results = [json.loads(json_str) for json_str in json_list]
        fever = pd.DataFrame([(elem["context"], elem["query"], elem["label"]) for elem in results])
        fever.columns = ["sentence1", "sentence2", "label"]
        fever["label"] = fever["label"].map({'NOT ENOUGH INFO': 1, 'REFUTES': 2, 'SUPPORTS': 0})

        return fever

                                            

    def get_glue_data(self, dataset_name, feature_list, split_types):
        """helper for getting GLUE data"""
        d = {}
        dataset = load_dataset('glue', dataset_name)

        for feature in feature_list:
            data = []

            for split_type in split_types:
                data += dataset[split_type][feature]
            
            d[self.mapper[feature]] = data

        data = pd.DataFrame(d, columns=["sentence1", "sentence2", "label"])
        
        if dataset_name == "rte":
            data["label"] = [1 if elem == 0 else 0 for elem in data["label"]]

        elif dataset_name == "stsb":
            data["label"] = [0 if label <= 3 else 1 for label in data["label"]]

        return data

    
    def get_paws_data(self):
        """helper for loading PAWS data (0-1)"""
        d = {}
        dataset = load_dataset('paws', "labeled_final")

        for feature in ['sentence1', 'sentence2', 'label']:
            data = []

            for split_type in ["train", "validation", "test"]:
                data += dataset[split_type][feature]
            
            d[feature] = data

        return pd.DataFrame(d, columns=d.keys())                         


    def get_swag_data(self):
        """helper for loading swag data (0-1)"""
        d = {}
        dataset = load_dataset('swag')

        sent1_l = []
        sent2_l = []
        
        for row in dataset["train"]:
            sent1_l.append(row["sent1"])
            sent2_l.append(row["sent2"] + " " + row["ending" + str(row["label"])])

        d["sentence1"] = sent1_l
        d["sentence2"] = sent2_l

        swag_df = pd.DataFrame(d, columns=d.keys())
        swag_df["label"] = 1

        return swag_df


    def get_art_data(self):
        d = {}
        dataset = load_dataset('art')

        sent1_l = []
        sent2_l = []
        
        for row in dataset["train"]:
            sent1_l.append(row["observation_1"] + " " + row["observation_2"])
            sent2_l.append(row["hypothesis_{}".format(row["label"])])


        for row in dataset["validation"]:
            sent1_l.append(row["observation_1"] + " " + row["observation_2"])
            sent2_l.append(row["hypothesis_{}".format(row["label"])])

            
        d["sentence1"] = sent1_l
        d["sentence2"] = sent2_l

        art_df = pd.DataFrame(d, columns=d.keys())
        art_df["label"] = 1

        return art_df


    def get_conjNLI_data(self):
        """helper for getting conjNLI dataset"""
        merged_df = pd.concat([pd.read_csv("/content/adversarial_train_15k.tsv.txt", sep="\t").dropna(), 
                          pd.read_csv("/content/conj_dev.tsv.txt", sep="\t").dropna()], ignore_index=True)
        
        merged_df.columns = ["sentence1", "sentence2", "label"]        
        merged_df["label"] = merged_df["label"].map({'neutral': 1, 'contradiction': 2, 'entailment': 0})
        return merged_df


    def get_EQUATE_data(self):
        files = ["/content/AWPNLI.jsonl", "/content/NewsNLI.jsonl", "/content/RedditNLI.jsonl", "/content/RTE_Quant.jsonl", "/content/StressTest.jsonl"]
        result_df = pd.DataFrame()

        for f in tqdm(files):
            try:
                with open(f, 'r') as json_file:
                    json_list = list(json_file)

                results = [json.loads(json_str) for json_str in json_list]
                result_df = pd.concat([result_df,
                                      pd.DataFrame([(elem["sentence1"], elem["sentence2"], elem["gold_label"]) for elem in results])], ignore_index=True)
                
            except Exception as e:
                print(f)           
                print(e)

        result_df.columns = ["sentence1", "sentence2", "label"]
        result_df["label"] = result_df["label"].map({'neutral': 1, 'contradiction': 2, 'entailment': 0})
        return result_df
        

    def create_dataset(self):
        """main routine for this class."""
        # STAGE 1
        stage1_dict = {
            'mrpc': (['sentence1', 'sentence2', 'label'], ["train", "validation", "test"]), 
            "qnli": (['question', 'sentence', 'label'], ["train", "validation"]),
            "rte": (['sentence1', 'sentence2', 'label'], ["train", "validation"]),     
            'stsb': (['sentence1', 'sentence2', 'label'], ["train", "validation"]), 
            'wnli': (['sentence1', 'sentence2', 'label'], ["train", "validation"]),
            "qqp": (['question1', 'question2', 'label'], ["train", "validation"]),
        }

        for key, val in stage1_dict.items():
            self.stage1_train_df = pd.concat([self.stage1_train_df,
                                              self.get_glue_data(key, val[0], val[1])],
                                              ignore_index=True)
        
        for dataset in tqdm([self.get_paws_data(),
                        self.get_swag_data(),
                        self.get_art_data()]):        

            self.stage1_train_df = pd.concat([self.stage1_train_df, dataset],
                                             ignore_index=True)

        print("Stage 1 done.")

        self.get_anli_data()
         
        for dataset in tqdm([self.get_mnli_data(), self.get_fever_data(), self.get_EQUATE_data(), self.get_conjNLI_data()]):
            self.stage2_train_df = pd.concat([self.stage2_train_df, dataset],
                                              ignore_index=True)                                         

        print("Stage 2 done.")

        self.stage1_train_df["label"] = self.stage1_train_df["label"].astype(int)
        self.stage2_train_df["label"] = self.stage2_train_df["label"].astype(int)
        
        # save results
        self.stage1_train_df.drop_duplicates().dropna(how="any", axis=0).reset_index(drop=True)
        self.stage2_train_df.drop_duplicates().dropna(how="any", axis=0).reset_index(drop=True)

        self.stage1_train_df.to_parquet("/content/drive/MyDrive/deep_learning_project/data/stage1_train.parquet", index=False)
        self.stage2_train_df.to_parquet("/content/drive/MyDrive/deep_learning_project/data/stage2_train.parquet", index=False)
        self.stage2_val_df.to_parquet("/content/drive/MyDrive/deep_learning_project/data/stage2_val.parquet", index=False)
        self.stage2_test_df.to_parquet("/content/drive/MyDrive/deep_learning_project/data/stage2_test.parquet", index=False)

        self.dataset_metadata["concatenated"] = {}

        for key, data in {
            "stage1_train": self.stage1_train_df,
            "stage2_train": self.stage2_train_df,
            "val": self.stage2_val_df,
            "test": self.stage2_test_df}.items():

            self.dataset_metadata["concatenated"][key] = {
                "data_len": len(data),
                "avg_text_len": np.mean([len(elem.split()) for elem in data["sentence1"]] + [len(elem.split()) for elem in data["sentence2"]]),
                "label_ratio": data["label"].value_counts(normalize=True)
            }
        
        pprint(dataops.dataset_metadata)

        joblib.dump(self.dataset_metadata, "/content/drive/MyDrive/deep_learning_project/data/final_metadata.joblib")
        files.download("/content/drive/MyDrive/deep_learning_project/data/final_metadata.joblib")
        print("Done.")

    
dataops = DataOps()
dataops.create_dataset()