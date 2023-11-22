from transformers import DistilBertModel, DistilBertTokenizer

from transformers import AutoTokenizer, AutoModel


def getDistilBert(model_name = "distilbert-base-uncased") : 
    return DistilBertModel.from_pretrained(model_name), DistilBertTokenizer.from_pretrained(model_name) 


def getTinyBert(model_name = "bert-tiny"):
    assert model_name in ["bert-tiny", "bert-mini", "bert-small", "bert-medium"], "Invalid model name" 
    model_name= f"prajjwal1/{model_name}"
    return AutoModel.from_pretrained(model_name), AutoTokenizer.from_pretrained(model_name)  
    



if __name__ == "__main__" :
    model, tokenizer = getTinyBert()
    print(model)
    print(tokenizer)
    
    model, tokenizer = getDistilBert()
    print(model)
    print(tokenizer) 