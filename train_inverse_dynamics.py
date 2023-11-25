import json
import os

import torch
import numpy as np
import argparse 

from src.models.encoder import * 
from src.utils.utils import *

from torch import nn 
from torch import optim 
from torch.utils.data import DataLoader 


def parse_args() : 
    parser = argparse.ArgumentParser(description='Training inverse dynamics model and save it')
    
    # training parameters
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--enc_lr', type=float, default=1e-3, help='Learning rate for encoder')
    parser.add_argument('--inv_lr', type=float, default=1e-4, help='Learning rate for inverse dynamics')
    
    # paths
    parser.add_argument('--save_path', type=str, default=".output/ckpt/inverse_dynamics", help='Path to save the model')
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
    def __init__(self, path, tokenizer, asin_to_idx, max_len = 512, sep_token = "[SEP]") : 
        self.attributes = []
        self.reviews = []
        self.actions = [] 
        self.labels = [] 
        self.asin_to_idx = asin_to_idx 
        self.tokenzier = tokenizer 
        self.sep_token = sep_token
        self.max_len = max_len
        
        with open(os.path.join(path, "train_data.json"), "r") as f :
            sequences = json.load(f) 
            for seq in sequences : 
                # seq : List[dict]
                assert len(seq) > 1, "Sequence length should be greater than 1" 
                
                attribute = [s["attribute"] for s in seq] 
                review = [s["review"] for s in seq]
                concated_attribute = f" {sep_token} ".join(attribute) # concated attribute 
                
                # only add concated sentence is lower than max_len -> or truncate
                if len(tokenizer.encode(concated_attribute)) < max_len :
                    self.attributes.append(seq) 
                    self.reviews.append([0] * (max_len - len(review) + review)) 
                else :
                    # find idx that is lower than max_len from the end 
                    for idx in range(1, len(attribute), 1) : 
                        if len(tokenizer.encode(f" {sep_token} ".join(attribute[idx:]))) < max_len : 
                            self.attributes.append(attribute[idx:]) 
                            # padd the reviews
                            self.reviews.append([0] * (max_len - len(review[idx:]) +review[idx:]))
                            break
                self.actions.append(attribute[-1]) # last item in the sequence is the action
                self.labels.append(asin_to_idx[seq[-1]["asin"]]) # last item in the sequence is the action 
    def __len__(self) : 
        return len(self.attributes) 
    
    def __getitem__(self, idx) : 
        return {"state1": self.attributes[idx], "state2": self.attributes[idx][:-1], "reviews": self.reviews[idx], "action": self.actions[idx], "label": self.labels[idx]}

    def __collate_fn__(self, batch) : 
        # tokenize the state1 and state2 and action 
        state1 = [b["state1"] for b in batch]
        state2 = [b["state2"] for b in batch] 
        action = [b["label"] for b in batch] 
        label = [b["label"] for b in batch] 
        
        state1 = self.tokenizer(state1, padding = True, max_length = self.max_len, truncation = True)
        state2 = self.tokenizer(state2, padding = True, max_length = self.max_len, truncation = True)
        action = [self.asin_to_idx[a] for a in action]
        
        return {"state1": torch.tensor(state1["input_ids"]), "state2": torch.tensor(state2["input_ids"]), "action": torch.tensor(action), "label": torch.tensor(label)}
        



def training(model : nn.Module, train_dataloader, test_dataloader, enc_optimizer, inv_optimizer, loss_fn, epochs, device, save_path) -> dict():

    model.train()
    best_loss = np.inf 
    train_epoch_loss = []
    test_epcoh_loss = [] 
    
    save_file_name = os.path.join(save_path, "encoder_inverse_dynamics.pt")
    for e in epochs:
        train_loss = [] 
        for batch in train_dataloader:
            enc_optimizer.zero_grad()
            inv_optimizer.zero_grad()

            state1 = batch["state1"].to(device)
            state2 = batch["state2"].to(device)
        
            if args.target == "action" : 
                loss = loss_fn(model(state1, state2), model.get_embeddings(batch["action"].to(device))) # dot product and get the loss
            elif args.target == "label" :
                loss = loss_fn(model(state1, state2), batch["label"].to(device))
                 
            loss.backward()

            inv_optimizer.step()
            enc_optimizer.step()
            
            train_loss.append(loss.detach().item())
        
        model.eval()
        test_loss = [] 
        with torch.no_grad():
            for batch in test_dataloader:
                state1 = batch["state1"].to(device)
                state2 = batch["state2"].to(device)
                label = batch["label"].to(device) 
                loss = loss_fn(model(state1, state2), label)
                test_loss.append(loss.detach().item())
    
        print(f"Epoch {e} || Train loss: {np.mean(train_loss)} | Test loss: {np.mean(test_loss)}")
        
        train_epoch_loss.append(np.mean(train_loss))
        test_epcoh_loss.append(np.mean(test_loss))
        
        if np.mean(test_loss) < best_loss : 
            print("Saving model...") 
            torch.save(model.state_dict(), save_file_name)
            best_loss = np.mean(test_loss) 
            print("Model saved")

    print("Training complete")
    
    return {"train_loss_log": train_epoch_loss, "test_loss_log": test_epcoh_loss}

    
def main(args) : 
    # device 
    device = torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")  # use gpu if available 
    
    # seed 
    set_seed(42)
    
    # path generation 
    if not os.path.exists(args.save_path) : 
        os.makedirs(args.save_path)
    
    # model
    if args.encoder == "tinybert" : 
        encoder, tokenizer = getTinyBert()
    elif args.encoder == "distilbert" : 
        encoder, tokenizer = getDistilBert()
    else :
        raise NotImplementedError("Invalid encoder") 
        
    # define asin to idx mapping 
    with open(args.item_embedding_path, "r") as f :
        item_embeddings = json.load(f) # asin : embedding 
    asin_to_idx = {asin : idx for idx, asin in enumerate(item_embeddings.keys())}
    idx_to_asin = {idx : asin for idx, asin in enumerate(item_embeddings.keys())} 
    
    num_items = len(item_embeddings) 
    
    # define dataset 
    dataset = EncoderDataset(args.data_path, tokenizer, asin_to_idx = asin_to_idx, max_len = 512, sep_token = "[SEP]")
    
    # split dataset
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    # define dataloader
    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, collate_fn = dataset.__collate_fn__)
    test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False, collate_fn = dataset.__collate_fn__) 
    
    
    # define inverse dynamics model 
    inverse_dynamics = InverseDynamics(hidden_dim = encoder.hidden_dim, item_embedding_lookup_table = item_embeddings, expansion_ratio = args.expansion_ratio, dropout_rate = args.dropout_rate) 
    model = EncoderInverseDynamics(encoder, inverse_dynamics, item_embedding_lookup_table = item_embeddings, mode = args.mode) 
    
    # define loss function
    loss_fn = nn.CrossEntropyLoss()    
    
    # define optimizer 
    enc_optim = optim.Adam(model.encoder.parameters(), lr = args.lr)
    inv_optim = optim.Adam(model.inverse_dynamics.parameters(), lr = args.lr) 
    
    # training
    training(model, dataloader, enc_optim, inv_optim, loss_fn, args.epochs, device, args.save_path)
    
if __name__ == "__main__" :

    args = parse_args()
    main(args)







