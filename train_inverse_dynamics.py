import torch
from torch import nn 
from torch import optim 
from torch.utils.data import DataLoader 
import argparse 
from models import * 
import json
import os


def parse_args() : 
    parser = argparse.ArgumentParser(description='Training inverse dynamics model and save it')
    
    # training parameters
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    
    # paths
    parser.add_argument('--save_path', type=str, default="./preprocess/inverse_dynamics", help='Path to save the model')
    parser.add_argument('--data_path', type=str, default="./preprocess/encoder_dataset", help='Path to the dataset') 
    parser.add_argument('--item_embedding_path', type=str, default="./preprocess/item_embeddings", help='Path to the item embeddings') 
    
    # device
    parser.add_argument('--device', type=str, default="cuda", help='Device to train on')

    # model parameters 
    parser.add_argument('--encoder', type=str, default="tinybert", choices = ["tinybert", "distilbert"], help='Encoder to use')
    parser.add_argument('--expansion_ratio', type=int, default=4, help='Expansion ratio for MLP') 
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate for MLP') 
    parser.add_argument('--mode', type=str, default="cls", choices = ["cls", "average"], help='Mode for encoder inverse dynamics')  
    
    args = parser.parse_args() 
    return args 


class EncoderDataset(torch.utils.data.Dataset) : 
    def __init__(self, path, tokenizer) : 
        self.attributes = []
        self.reviews = []
        with open(os.path.join(path, "train_data.json"), "r") as f :
            sequences = json.load(f) 
            for seq in sequences : 
                assert len(seq) > 1, "Sequence length should be greater than 1" 
                self.attributes.append([review["attribue"] for review in seq])
                self.reviews.append([review["review"] for review in seq])
        self.tokenzier = tokenizer 
    def __len__(self) : 
        return len(self.attributes) 
    
    def __getitem__(self, idx) : 
        return self.attributes[idx], self.reviews[idx] 





def training(encoder, inverse_dynamics, dataset, epochs, batch_size, lr, device, args) :  
    pass 
    
    
    
def main(args) : 
    pass 
    
    
if __name__ == "__main__" :
    
    # test 
    
    path = "./preprocess/data/preprocessed/train_data.json" 
    with open(path, "r") as f :
        sequences = json.load(f) 
    print(sequences) 
    
    args = parse_args()
    main(args)








