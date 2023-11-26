import json
import os
import copy
import time 

import torch
import numpy as np
import argparse 

from src.models.encoder import * 
from src.utils.utils import *
from src.utils.metrics import *

from torch import nn 
from torch import optim 
from torch.utils.data import DataLoader 
from tqdm import tqdm


from multiprocessing import Pool
from dotenv import load_dotenv


def parse_args() : 
    parser = argparse.ArgumentParser(description='Training inverse dynamics model and save it')
    
    # training parameters
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--enc_lr', type=float, default=1e-3, help='Learning rate for encoder')
    parser.add_argument('--inv_lr', type=float, default=1e-4, help='Learning rate for inverse dynamics')
    parser.add_argument('--log_interval', type=int, default=100, help='Interval for logging')
    
    # paths
    parser.add_argument('--save_path', type=str, default="./output/ckpt/inverse_dynamics", help='Path to save the model')
    parser.add_argument('--data_path', type=str, default="./preprocess/data/preprocessed/train_data.json", help='Path to the dataset') 
    parser.add_argument('--meta_path', type=str, default="./preprocess/data/preprocessed/meta_data.json", help='Path to the meta data')
    
    # device
    parser.add_argument('--device', type=str, default="cuda", help='device to train on')

    # model parameters 
    parser.add_argument('--encoder', type=str, default="tinybert", choices = ["tinybert", "distilbert"], help='Encoder to use')
    parser.add_argument('--expansion_ratio', type=int, default=4, help='Expansion ratio for MLP') 
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate for MLP') 
    parser.add_argument('--mode', type=str, default="cls", choices = ["cls", "average"], help='Mode for encoder inverse dynamics')  
    parser.add_argument('--max_len', type=int, default=512, help='Maximum length for input sequence')
    parser.add_argument("--freeze_encoder", default= True, help= "Freeze encoder weights") 
    
    args = parser.parse_args() 
    return args 


class EncoderDataset(torch.utils.data.Dataset) : 
    def __init__(self, path, sequences, tokenizer, asin_to_idx, max_len = 512, sep_token = "[SEP]") : 
        self.attributes = []
        self.reviews = []
        self.actions = [] 
        self.labels = [] 
        self.asin_to_idx = asin_to_idx 
        self.tokenizer = tokenizer 
        self.sep_token = sep_token
        self.max_len = max_len
        
        if sequences is None :
            with open(path, "r") as f :
                sequences = json.load(f)
        
        # too long -> multiprocess 
        num_processes = 4
        
        # split sequences into num_processes
        split_sequences = [sequences[i::num_processes] for i in range(num_processes)]
        
        # copy tokenizer and asin_to_idx for each process
        tokenizer_list = [copy.deepcopy(tokenizer)] * num_processes 
        asin_to_idx_list = [asin_to_idx] * num_processes 
        
        # multiprocess
        with Pool(num_processes) as p :
            results = p.starmap(self._process_sequences, 
                                zip(split_sequences, tokenizer_list, asin_to_idx_list, [max_len] * num_processes, [sep_token] * num_processes))
        
        # concatenate results 
        for result in results : 
            self.attributes += result[0]
            self.reviews += result[1]
            self.actions += result[2]
            self.labels += result[3] 
            
        del sequences, split_sequences, tokenizer_list, asin_to_idx_list, results
            
                
    def _process_sequences(self, sequences, tokenizer, asin_to_idx, max_len, sep_token) :
        attributes = []
        reviews = []
        actions = [] 
        labels = [] 
        
        for seq in sequences:
             
            # seq : List[dict]
            assert len(seq) > 1, "Sequence length should be greater than 1" 
            
            attribute = [s["attribute"] for s in seq] 
            review = [s["review"] for s in seq]
            concated_attribute = f" {sep_token} ".join(attribute)
            
            # only add concated sentence is lower than max_len -> or truncate
            if len(tokenizer.encode(concated_attribute)) < max_len :
                attributes.append(concated_attribute)
                reviews.append([0] * (max_len - len(review)) + review)
            else :
                # find idx that is lower than max_len from the end 
                for idx in range(1, len(attribute), 1) : 
                    if len(tokenizer.encode(f" {sep_token} ".join(attribute[idx:]))) < max_len : 
                        attributes.append(f" {sep_token} ".join(attribute[idx:]))
                        # padd the reviews
                        reviews.append([0] * (max_len - len(review[idx:])) +review[idx:])
                        break
            actions.append(attribute[-1]) # last item in the sequence is the action
            labels.append(asin_to_idx[seq[-1]["asin"]]) # last item in the sequence is the action 
                
        return attributes, reviews, actions, labels 
                
    def __len__(self) : 
        return len(self.attributes) 
    
    def __getitem__(self, idx) : 
        return {"state1": self.attributes[idx], "state2": self.attributes[idx][:-1], "reviews": self.reviews[idx], "label": self.labels[idx]}

    def __collate_fn__(self, batch) : 
        # tokenize the state1 and state2 and action for one batch
        # it could be inefficient but it is okay for now since model is not that big but I will change it later @jinpil  
        state1 = [b["state1"] for b in batch]
        state2 = [b["state2"] for b in batch] 
        label = [b["label"] for b in batch] 
        
        state1 = self.tokenizer(state1, padding = True, max_length = self.max_len, truncation = True)
        state2 = self.tokenizer(state2, padding = True, max_length = self.max_len, truncation = True)
        
        return {"state1": torch.tensor(state1["input_ids"]), 
                "attn_mask1": torch.tensor(state1["attention_mask"]),
                "state2": torch.tensor(state2["input_ids"]), 
                "attn_mask2": torch.tensor(state2["attention_mask"]),
                "label": torch.tensor(label)}

def training(args, 
             model : nn.Module, 
             train_dataloader: DataLoader, 
             test_dataloader: DataLoader, 
             enc_optimizer, inv_optimizer, 
             loss_fn, 
             epochs, 
             device, 
             save_path,
             topks = [3, 5, 10, 20 ,100]) -> dict():

    start_time = time.time()
    model.to(device)
    
    best_loss = np.inf 
    train_epoch_loss = []
    test_epoch_loss = []  
    
    save_file_name = os.path.join(save_path, "encoder_inverse_dynamics.pt")
    for e in range(epochs):
        train_HR_epoch = {topk : [] for topk in topks}
        train_NDCG_epoch = {topk : [] for topk in topks}
        test_HR_epoch = {topk : [] for topk in topks}
        test_NDCG_epoch = {topk : [] for topk in topks}
        
        # training code
        model.train()
        train_loss = []
        num_train_data = 0 
        for batch in tqdm(train_dataloader):
            
            num_train_data += len(batch) 
            enc_optimizer.zero_grad()
            inv_optimizer.zero_grad()

            state1 = batch["state1"].to(device)
            state2 = batch["state2"].to(device)
            attn_mask1 = batch["attn_mask1"].to(device) 
            attn_mask2 = batch["attn_mask2"].to(device) 
            
            label = batch["label"].to(device) 
            preds = model(state1, attn_mask1, state2, attn_mask2)
            loss = loss_fn(preds, label)
            
            loss.backward()

            inv_optimizer.step()
            enc_optimizer.step()
            
            # logging -> TODO : add wandb logging 
            train_loss.append(loss.detach().item())
            if len(train_loss) % args.log_interval == 0 :
                print(f"Epoch {e} | Train loss: {np.mean(train_loss)}")
            
            # HR and NDCG 
            recommendations = torch.argsort(preds, dim = -1, descending = True).detach().cpu().numpy() # [batch_size, num_items]
            label = label.detach().cpu().numpy() # [batch_size, 1] 
            hr = HR(recommendations, label)
            ndcg = NDCG(recommendations, label)
            for topk in topks : 
                train_HR_epoch[topk].append(hr[topk] * len(batch)) 
                train_NDCG_epoch[topk].append(ndcg[topk] * len(batch)) 
            
            
        # test code 
        model.eval()
        test_loss = [] 
        with torch.no_grad():
            for batch in tqdm(test_dataloader):
                state1 = batch["state1"].to(device)
                state2 = batch["state2"].to(device)
                attn_mask1 = batch["attn_mask1"].to(device) 
                attn_mask2 = batch["attn_mask2"].to(device) 
                label = batch["label"].to(device) 
                
                preds = model(state1, attn_mask1, state2, attn_mask2) 
                loss = loss_fn(preds, label)
                
                test_loss.append(loss.detach().item())
                recommendations = torch.argsort(preds, dim = -1, descending = True).detach().cpu().numpy() # [batch_size, num_items]
                label = label.detach().cpu().numpy() # [batch_size, 1] 
                
                test_hr = HR(recommendations, label)
                test_ndcg = NDCG(recommendations, label)
                for topk in topks : 
                    test_HR_epoch[topk].append(test_hr[topk] * len(batch))
                    test_NDCG_epoch[topk].append(test_ndcg[topk] * len(batch)) 
               
        train_epoch_loss.append(np.mean(train_loss))
        test_epoch_loss.append(np.mean(test_loss))
        
        print(f"Training Report for Epoch {e}")
        print(f"Train loss: {np.mean(train_loss)} | Test loss: {np.mean(test_loss)}")
        print(" ".join([f"Train HR@{topk}: {np.sum(train_HR_epoch[topk]) / num_train_data}" for topk in topks])) 
        print(" ".join([f"Train NDCG@{topk}: {np.sum(train_NDCG_epoch[topk]) / num_train_data}" for topk in topks]))
        print(" ".join([f"Test HR@{topk}: {np.sum(test_HR_epoch[topk]) / len(test_dataloader.dataset)}" for topk in topks]))
        print(" ".join([f"Test NDCG@{topk}: {np.sum(test_NDCG_epoch[topk]) / len(test_dataloader.dataset)}" for topk in topks]))
        print("=====================================")
        
        
        # save model 
        if np.mean(test_loss) < best_loss : 
            torch.save(model.state_dict(), save_file_name)
            best_loss = np.mean(test_loss) 
            
    print("Training complete")
    
    return {"train_loss_log": train_epoch_loss, "test_loss_log": test_epoch_loss, "num_train_data": num_train_data, "time": time.time() - start_time}


def get_item_embeddings(meta_data, tokenizer, encoder : nn.Module, emb_dim, device) :
    with open(meta_data, "r") as f : 
        meta_data = json.load(f)

    encoder.to(device)
    item_embeddings = np.zeros((len(meta_data), emb_dim)) # [num_items, hidden_dim] 
    asin_to_idx = dict() # key : asin // val : idx
    idx_to_asin = dict() # key : idx // val : asin 
    
    # batch inference 
    batch_size = 512
       
    for start in tqdm(range(0, len(meta_data), batch_size)) :
        batch_asins = list(meta_data.keys())[start: start + batch_size]
        batch_attributes = [f"This product, titled '{meta_data[asin]['title']}' and branded as {meta_data[asin]['brand']}, falls under the category of {meta_data[asin]['category']}." for asin in batch_asins]
        batch_attributes = tokenizer(batch_attributes, padding = True, max_length = 512, truncation = True)
        tokens = torch.tensor(batch_attributes["input_ids"]).to(device)
        attn_mask = torch.tensor(batch_attributes["attention_mask"]).to(device) 
        batch_embeddings = encoder(tokens, attention_mask = attn_mask).last_hidden_state[:,0,...].detach().cpu().numpy()
        item_embeddings[start: start + batch_size] = batch_embeddings
        
        asin_to_idx.update({asin: idx + start for idx, asin in enumerate(batch_asins)}) 
        idx_to_asin.update({idx + start: asin for idx, asin in enumerate(batch_asins)})
    # add last batch
    if len(meta_data) % batch_size != 0 :
        start = len(meta_data) - len(meta_data) % batch_size 
        batch_asins = list(meta_data.keys())[start:]
        batch_attributes = [f"This product, titled '{meta_data[asin]['title']}' and branded as {meta_data[asin]['brand']}, falls under the category of {meta_data[asin]['category']}." for asin in batch_asins]
        batch_attributes = tokenizer(batch_attributes, padding = True, max_length = 512, truncation = True)
        tokens = torch.tensor(batch_attributes["input_ids"]).to(device)
        attn_mask = torch.tensor(batch_attributes["attention_mask"]).to(device) 
        batch_embeddings = encoder(tokens, attention_mask = attn_mask).last_hidden_state[:,0,...].detach().cpu().numpy()
        item_embeddings[start:] = batch_embeddings
 
        asin_to_idx.update({asin: idx + start for idx, asin in enumerate(batch_asins)}) 
        idx_to_asin.update({idx + start: asin for idx, asin in enumerate(batch_asins)})
        
    print(f"Number of items: {len(item_embeddings)}")
    print(f"Shape of item embeddings: {item_embeddings.shape}")
    assert len(asin_to_idx) == len(idx_to_asin) == len(item_embeddings), "Length of asin_to_idx, idx_to_asin, and item_embeddings should be the same"
    
    return asin_to_idx, idx_to_asin, item_embeddings
        
def main(args) : 
    # load env
    load_dotenv(verbose=True)
    
    # device 
    device = torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")  # use gpu if available 
    print(f"Device: {device}")
    # seed 
    set_seed(42)
    
    # path generation 
    if not os.path.exists(args.save_path) : 
        os.makedirs(args.save_path)
    
    # model
    if args.encoder == "tinybert" : 
        encoder, tokenizer, emb_dim = getTinyBert()
    elif args.encoder == "distilbert" : 
        encoder, tokenizer, emb_dim = getDistilBert()
    else : 
        raise NotImplementedError("Invalid encoder") 
    print(f"Encoder: {args.encoder} | Embedding dimension: {emb_dim}")
    
    # load each item embeddings for inverse dynamics model
    asin_to_idx, idx_to_asin, item_embeddings = get_item_embeddings(args.meta_path, tokenizer, encoder, emb_dim, device) 
    
    # save it as json file
    with open(os.path.join(args.save_path, "asin_to_idx.json"), "w") as f : 
        json.dump(asin_to_idx, f)
    with open(os.path.join(args.save_path, "idx_to_asin.json"), "w") as f :
        json.dump(idx_to_asin, f)
   
    # load dataset
    train_sequences = json.load(open(args.data_path, "r")) # List[List[dict]]
    
    # ramdomly split train_sequence into train and test
    np.random.shuffle(train_sequences)
    test_sequences = train_sequences[:int(len(train_sequences) * 0.2)]
    train_sequences = train_sequences[int(len(train_sequences) * 0.2):]
    
   
    # define dataset 
    train_dataset = EncoderDataset(None, train_sequences, tokenizer, asin_to_idx = asin_to_idx, max_len = 512, sep_token = "[SEP]")
    test_dataset  = EncoderDataset(None, test_sequences, tokenizer, asin_to_idx = asin_to_idx, max_len = 512, sep_token = "[SEP]")
        
    # define dataloader
    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, collate_fn = train_dataset.__collate_fn__)
    test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False, collate_fn = test_dataset.__collate_fn__) 
    
    
    # define inverse dynamics model 
    inverse_dynamics = InverseDynamics(emb_dim = emb_dim, 
                                       expansion_ratio = args.expansion_ratio, 
                                       dropout_rate = args.dropout_rate, 
                                       asin_to_idx = asin_to_idx, 
                                       idx_to_asin = idx_to_asin, 
                                       item_embeddings = item_embeddings
                                       )
    model = EncoderInverseDynamics(encoder, 
                                   inverse_dynamics, 
                                   item_embedding_lookup_table = item_embeddings, 
                                   mode = args.mode,
                                   max_len = args.max_len,
                                   freeze_encoder = args.freeze_encoder) 
    
    # define loss function
    loss_fn = nn.CrossEntropyLoss()    
    
    # define optimizer 
    enc_optim = optim.Adam(model.encoder.parameters(), lr = args.enc_lr)    
    inv_optim = optim.Adam(model.inverse_dynamics.parameters(), lr = args.inv_lr) 
    
    # training
    train_log = training(args, model, train_dataloader, test_dataloader, enc_optim, inv_optim, loss_fn, args.epochs, device, args.save_path) 
    
    # save training log
    with open(os.path.join(args.save_path, "train_log.json"), "w") as f : 
        json.dump(train_log, f)
    
if __name__ == "__main__" :

    args = parse_args()
    main(args)







