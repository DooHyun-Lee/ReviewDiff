import torch
import numpy as np

from torch import nn

from transformers import DistilBertModel, DistilBertTokenizer
from transformers import AutoTokenizer, AutoModel


def getDistilBert(model_name = "distilbert-base-uncased") : 
    return DistilBertModel.from_pretrained(model_name), DistilBertTokenizer.from_pretrained(model_name), 768

def getTinyBert(model_name = "bert-tiny"):
    assert model_name in ["bert-tiny", "bert-mini", "bert-small", "bert-medium"], "Invalid model name" 
    embedding_size = {"bert-tiny" : 128, "bert-mini" : 256, "bert-small" : 512, "bert-medium" : 768}
    size = embedding_size[model_name]
    model_name= f"prajjwal1/{model_name}"
    return AutoModel.from_pretrained(model_name), AutoTokenizer.from_pretrained(model_name), size

class InverseDynamics(nn.Module) :
    def __init__(self, emb_dim, expansion_ratio : int, dropout_rate : float, asin_to_idx : dict, idx_to_asin : dict, item_embeddings : np.ndarray): 
        super().__init__ () 
        
        self.item_embedding_lookup_table = item_embeddings  # key : asin , value : embedding
        self.asin_to_idx = asin_to_idx # key : asin, value : idx 
        self.idx_to_asin = idx_to_asin # key : idx, value : asin 
        
        self.emb_dim = emb_dim  
        self.expansion_ratio = expansion_ratio 
        self.dropout_rate = dropout_rate 
        self.mlp = nn.Sequential(nn.Linear(2*emb_dim, expansion_ratio*emb_dim), 
                                 nn.ReLU(), 
                                 nn.Dropout(dropout_rate),
                                 nn.Linear(expansion_ratio*emb_dim, expansion_ratio*emb_dim),
                                 nn.ReLU(),
                                 nn.Linear(expansion_ratio*emb_dim, emb_dim))
        
        self.num_items = len(self.item_embedding_lookup_table) # key : item_id, value : embedding 
        self.lookuptensor = torch.tensor(item_embeddings, dtype = torch.float32) # [N, emb_dim]
        self.lookuptensor = nn.Parameter(self.lookuptensor, requires_grad = False) # [N, emb_dim] 
        
    def get_dot_products(self, output_emb) : 
        # output_emb : [batch_size, emb_dim]
        # lookuptensor : [N, emb_dim]
        # output_emb * lookuptensor.T : [batch_size, N]
        return torch.matmul(output_emb, self.lookuptensor.T) # [batch_size, N] 
        
        
    def forward(self, emb1, emb2):
        output_emb = self.mlp(torch.cat([emb1, emb2], dim = -1)) # [batch_size, 2*h] -> [batch_size, h]
        return  self.get_dot_products(output_emb) # [batch_size, N] 
    
    
        
    def predict(self, emb1, emb2, topk = 10):
        scores = self.forward(emb1, emb2) 
        predictions = torch.topk(scores, k = topk, dim = -1) # [batch_size, topk]

        # change idx to asin
        predictions = [[self.idx_to_item[idx.item()] for idx in prediction] for prediction in predictions] # [batch_size, topk]
        return predictions

         
class EncoderInverseDynamics(nn.Module) : 
    def __init__(self, encoder, inverse_dynamics, item_embedding_lookup_table, max_len = 512, mode = "cls", freeze_encoder = True) : 
        super().__init__() 
        
        assert mode in ["last_hidden_state", "average_hidden_state", "cls"], f"Invalid mode : {mode}"
        self.mode = mode 
        self.encoder = encoder 
        self.inverse_dynamics = inverse_dynamics
        self.freeze_encoder = freeze_encoder 
        if self.freeze_encoder : 
            for param in self.encoder.parameters() : 
                param.requires_grad = False 
                
    def forward(self, state1 : torch.Tensor, attn_mask1, state2 : torch.Tensor, attn_mask2) :
        """
        Args:
            state1 (torch.Tensor): (batch_size, seq_len)  # tokenized integer sequence s_{t}
            state2 (torch.Tensor): (batch_size, seq_len)  # tokenized integer sequence with s_{t+1} 
        """

        # get the embedding for the states  
        state1 = self.encoder(state1, attn_mask1).last_hidden_state # [batch_size, seq_len, hidden_dim]
        state2 = self.encoder(state2, attn_mask2).last_hidden_state # [batch_size, seq_len, hidden_dim] 
        
        # choose what to use for total embedding for the state 
        if self.mode == "last_hidden_state" : 
            state1 = state1[:, -1, :] 
            state2 = state2[:, -1, :]
        elif self.mode == "average_hidden_state" : 
            state1 = torch.mean(state1, dim = 1) 
            state2 = torch.mean(state2, dim = 1)    
        elif self.mode == "cls" : 
            state1 = state1[:, 0, :] 
            state2 = state2[:, 0, :]
        else :
            raise NotImplementedError("Invalid mode")
        
        # get the inverse dynamics 
        return self.inverse_dynamics(state1, state2) 
        

if __name__ == "__main__" :
    model, tokenizer = getTinyBert()
    sample = "This is a sample sentence. [SEP] This is another sample sentence." 
    print(tokenizer(sample, padding='max_length', max_length = 512, truncation = False)) 
    print(type(tokenizer.encode(sample, padding = True, max_length = 512, truncation = True)))
    print(tokenizer.decode(tokenizer.encode(sample))) 
    
    model, tokenizer = getDistilBert()
    print(tokenizer(sample))
    print(tokenizer.encode(sample))
    print(tokenizer.decode(tokenizer.encode(sample)))