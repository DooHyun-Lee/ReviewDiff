import torch
from torch import nn
from transformers import DistilBertModel, DistilBertTokenizer
from transformers import AutoTokenizer, AutoModel

def getDistilBert(model_name = "distilbert-base-uncased") : 
    return DistilBertModel.from_pretrained(model_name), DistilBertTokenizer.from_pretrained(model_name) 

def getTinyBert(model_name = "bert-tiny"):
    assert model_name in ["bert-tiny", "bert-mini", "bert-small", "bert-medium"], "Invalid model name" 
    model_name= f"prajjwal1/{model_name}"
    return AutoModel.from_pretrained(model_name), AutoTokenizer.from_pretrained(model_name)  

class InverseDynamics(nn.Module) :
    def __init__(self, emb_dim, expansion_ratio : int, dropout_rate : float, item_embeddings : dict): 
        super().__init__ () 
        
        self.item_embedding_lookup_table = item_embeddings  # key : asin , value : embedding
        self.item_to_idx = {item : idx for idx, item in enumerate(self.item_embedding_lookup_table.keys())} # key : asin, value : idx 
        self.idx_to_item= {idx : item for item, idx in self.item_to_idx.items()} # key : idx, value : asin 
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
        self.lookuptensor = torch.tensor(list(self.item_embedding_lookup_table.values())) # N x H 
        
    def get_dot_products(self, output_emb) : 
        # output_emb : [batch_size, emb_dim]
        # lookuptensor : [N, emb_dim]
        # output_emb * lookuptensor.T : [batch_size, N]
        return torch.matmul(output_emb, self.lookuptensor.T) # [batch_size, N] 
        
        
    def forward(self, emb1, emb2):
        self.mlp(torch.cat([emb1, emb2], dim = -1)) # [batch_size, 2*h] -> [batch_size, h]
        scores = self.get_dot_products(emb1) # [batch_size, N] 
        return scores
    
        
    def predict(self, emb1, emb2, topk = 10):
        scores = self.forward(emb1, emb2) 
        predictions = torch.topk(scores, k = topk, dim = -1) # [batch_size, topk]

        # change idx to asin
        predictions = [[self.idx_to_item[idx.item()] for idx in prediction] for prediction in predictions] # [batch_size, topk]
        return predictions

         
class EncoderInverseDynamics(nn.Module) : 
    def __init__(self, encoder, inverse_dynamics, item_embedding_lookup_table, mode = "cls") : 
        super().__init__() 
        
        assert mode in ["last_hidden_state", "average_hidden_state"], "Invalid mode" 
        self.mode = mode 
        self.encoder = encoder 
        self.inverse_dynamics = inverse_dynamics(hidden_dim = encoder.hidden_dim, item_embedding_lookup_table = item_embedding_lookup_table)
        
    def forward(self, state1 : torch.Tensor, state2 : torch.Tensor) :
        """
        Args:
            state1 (torch.Tensor): (batch_size, seq_len)  # tokenized integer sequence s_{t}
            state2 (torch.Tensor): (batch_size, seq_len)  # tokenized integer sequence with s_{t+1} 
        """

        # get the embedding for the states  
        state1 = self.encoder(state1) # [batch_size, seq_len, hidden_dim]
        state2 = self.encoder(state2) # [batch_size, seq_len, hidden_dim] 
        
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
    print(model)
    print(tokenizer)
    
    model, tokenizer = getDistilBert()
    print(model)
    print(tokenizer) 