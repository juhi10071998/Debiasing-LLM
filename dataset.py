import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

class BiasDataset(Dataset):
    
    def __init__(self, data_path, n_tokens):
        #data = np.loadtxt("./data/occupation.csv", delimiter =",", skiprows = 1)
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        data = pd.read_csv(data_path)
        self.x = data.to_numpy()
        self.n_samples = len(data)
        self.n_tokens = n_tokens
        
        
        
    def __getitem__(self, index):
        sentence1, sentence2 = self.x[index][0], self.x[index][1]
       
        input_1 = self.tokenizer(sentence1, return_tensors="pt")
        input_2 = self.tokenizer(sentence2, return_tensors="pt")
        
        input_1['input_ids'] = torch.cat([torch.full((1, self.n_tokens), 50256), input_1['input_ids']], 1)
        input_1['attention_mask'] = torch.cat([torch.full((1, self.n_tokens), 1), input_1['attention_mask']], 1)
        
        
        input_2['input_ids'] = torch.cat([torch.full((1, self.n_tokens), 50256), input_2['input_ids']], 1)
        input_2['attention_mask'] = torch.cat([torch.full((1, self.n_tokens), 1), input_2['attention_mask']], 1)
        return [input_1,input_2]
        #return sentence1, sentence2
    
        
        
    def __len__(self):
        return self.n_samples

        