import torch
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from transformers import TextClassificationPipeline
import pandas as pd
import numpy as np
import tqdm
from tqdm import tqdm
import sys, os
from transformers import Pipeline
from torch import Tensor 
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
import torch.nn as nn
import sentence_transformers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_REPO = "shreyahavaldar/multilingual_politeness"
MODEL_REPO = "shreyahavaldar/xlm-roberta-politeness"
TOKENIZER_REPO = "xlm-roberta-base"

def load_data():
    hf_dataset = load_dataset(DATASET_REPO)
    return hf_dataset

def load_model():
    model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_REPO)
    model.to(device)
    return model

class PolitenessDataset(torch.utils.data.Dataset):
    def __init__(self, split, language="english"):
        dataset = load_dataset(DATASET_REPO)[split]
        dataset = dataset.filter(lambda x: x["language"] == language)
        dataset = dataset.rename_column("politeness", "label")
        self.dataset = dataset
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(TOKENIZER_REPO)
        self.max_len = max([len(text.split()) for text in dataset['Utterance']])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset["Utterance"][idx]
        label = self.dataset["label"][idx]
        encoding = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
        word_list = text.split()
        for i in range(len(word_list), self.max_len):
            word_list.append('')
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(label),
            'word_list': word_list
        }

class PolitenessClassifier(nn.Module):
    def __init__(self):
        super(PolitenessClassifier, self).__init__()
        self.model = load_model()

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask)
        logits = outputs.logits
        return logits


class Metric(nn.Module): 
    def __init__(self, model_name:str="distiluse-base-multilingual-cased"): 
        super(Metric, self).__init__()
        self.model = sentence_transformers.SentenceTransformer(model_name)
        self.centroids = self.get_centroids()
    
    def get_centroids(self):
        # read lexica files
        languages = ["english", "spanish", "chinese", "japanese"]
        lexica = {}
        for l in languages:
            filepath = f"../src/exlib/utils/politeness_lexica/{l}_politelex.csv"
            lexica[l] = pd.read_csv(filepath)

        # create centroids
        all_centroids = {}        
        for l in languages:
            categories = lexica[l]["CATEGORY"].unique()
            centroids = {}
            for c in categories:
                words = lexica[l][lexica[l]["CATEGORY"] == c]["word"].tolist()
                embeddings = self.model.encode(words)
                centroid = np.mean(embeddings, axis=0)
                centroids[c] = centroid
            assert len(categories) == len(centroids.keys())
            all_centroids[l] = centroids
            print(f"Centroids for {l} created.")
        return all_centroids

    # input: list of words
    def calculate_single_group_alignment(self, group:list, language:str="english"):
        #find max avg cos sim between word embeddings and centroids
        category_similarities = {}
        centroids = self.centroids[language]
        for category, centroid_emb in centroids.items():
            #calculate cosine similarity
            cos_sim = []
            for word in group:
                word_emb = self.model.encode(word)
                cos_sim.append(np.dot(word_emb, centroid_emb) / (np.linalg.norm(word_emb) * np.linalg.norm(centroid_emb)))
            avg_cos_sim = np.mean(cos_sim)
            category_similarities[category] = avg_cos_sim
        #return highest similarity score
        return max(category_similarities.values())

    def calculate_group_alignment(self, groups:list, language:str="english"):
        group_alignments = []
        for group in groups:
            group_alignments.append(self.calculate_single_group_alignment(group, language))
        return group_alignments



def get_politeness_scores(baselines = ['word', 'phrase', 'sentence']):
    dataset = PolitenessDataset("test")
    model = PolitenessClassifier()
    model.to(device)
    model.eval()

    metric = Metric()
    torch.manual_seed(1234)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    alignment_scores_all = {}
    for baseline in baselines:
        print(f"---- {baseline} Level Groups ----")
        
        baseline_scores = []
        for i, batch in enumerate(tqdm(dataloader)):
            word_lists = batch['word_list']
            word_lists = list(map(list, zip(*word_lists)))
            processed_word_lists = []
            for word_list in word_lists:
                processed_word_lists.append([word for word in word_list if word != ''])

            for word_list in processed_word_lists:
                groups = []
                if baseline == 'word':
                    for word in word_list:
                        groups.append([word])
                elif baseline == 'phrase':
                    #each group is 3 consecutive words
                    for i in range(0, len(word_list), 3):
                        groups.append(word_list[i:i+3])
                elif baseline == 'sentence':
                    #reconstruct sentences from word list
                    sentence = ""
                    for word in word_list:
                        sentence += word + " "
                        if word[-1] == "." or word[-1] == "!" or word[-1] == "?":
                            groups.append(sentence.split())
                            sentence = ""
                    if(len(sentence) > 0):
                        groups.append(sentence.split())
                                
                # print(groups)
                alignments = torch.tensor(metric.calculate_group_alignment(groups))
                score = alignments.mean()
                baseline_scores.append(score)
            if i > 1:
                break
        print(baseline_scores)    
        baseline_scores = torch.stack(baseline_scores)
        alignment_scores_all[baseline] = baseline_scores
    
    print(alignment_scores_all)
    return alignment_scores_all


            
                
    
