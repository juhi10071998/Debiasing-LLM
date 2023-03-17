import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

class BiasDataset(Dataset):
    
    def __init__(self, data_path, n_tokens):
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        data = pd.read_csv(data_path)
        
        #data = data.head(20)
        self.x = data.to_numpy()
        self.n_samples = len(data)
        self.n_tokens = n_tokens
        
        
        
    def __getitem__(self, index):
        self.tokenizer.pad_token = self.tokenizer.eos_token        
        ## return a bunch of sentences here, at any instance the dataloader gets called only once,         
        sentence1, sentence2 = self.x[index][0], self.x[index][1]   
        return sentence1, sentence2
            
        
    def __len__(self):
        return self.n_samples
    
    
    def collate_fn(self, batch):
        #separate the sentences for the two identity groups so that in a batch can send to the model.
        #returns for this, to get a tensor of lists from the list of tensors.

        batch_1, batch_2 = [ele[0] for ele in batch], [ele[1] for ele in batch]
        
        tokenized_1 = self.tokenizer.batch_encode_plus(batch_1, padding=True, truncation=True, return_tensors="pt")
        input_ids_1 = tokenized_1["input_ids"]
        attention_masks_1 = tokenized_1["attention_mask"]       
        input_ids_1 = torch.stack([torch.cat([torch.full((1, self.n_tokens), 50256)[0], input_id], 0) for input_id in input_ids_1])
        attention_masks_1 = torch.stack([torch.cat([torch.full((1, self.n_tokens), 1)[0], attention_mask], 0) for attention_mask in attention_masks_1])         
                
        tokenized_2 = self.tokenizer.batch_encode_plus(batch_2, padding=True, truncation=True, return_tensors="pt")
        input_ids_2 = tokenized_2["input_ids"]
        attention_masks_2 = tokenized_2["attention_mask"]        
        input_ids_2 = torch.stack([torch.cat([torch.full((1, self.n_tokens), 50256)[0], input_id], 0) for input_id in input_ids_2])
        attention_masks_2 = torch.stack([torch.cat([torch.full((1, self.n_tokens), 1)[0], attention_mask], 0) for attention_mask in attention_masks_2])
        
        return (input_ids_1, attention_masks_1, batch_1), (input_ids_2, attention_masks_2, batch_2)
    
    def get_dataloader(self, batch_size):
        return DataLoader(self, batch_size=batch_size, collate_fn=self.collate_fn)
    