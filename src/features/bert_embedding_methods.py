from transformers import AutoTokenizer, BertModel
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
from transformers import set_seed
set_seed(42)

"""
ref: https://towardsdatascience.com/3-types-of-contextualized-word-embeddings-from-bert-using-transfer-learning-81fcefe3fe6d
"""
def bert_text_preparation(
    text, 
    tokenizer, 
    max_length_tokens = 512
    ):
    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text, truncation=True, max_length=max_length_tokens, padding = True)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1]*len(indexed_tokens)

    tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')
    segments_tensors = torch.tensor([segments_ids]).to('cuda')

    return tokenized_text, tokens_tensor, segments_tensors

def get_bert_embeddings(tokens_tensor, segments_tensors, model):
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        hidden_states = outputs[2][1:]

    token_embeddings = hidden_states[-1]
    token_embeddings = torch.squeeze(token_embeddings, dim=0)
    list_token_embeddings = [token_embed.tolist() for token_embed in token_embeddings]

    return list_token_embeddings

def create_bert_embeddings_mean_from_series(
    model_name,
    text_series
):
    # Import hugginface models
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
    model = BertModel.from_pretrained(model_name, output_hidden_states=True).to('cuda')
    
    emb_len = 768
    # allocate array with the necessary shape
    emb_list = np.empty((len(text_series), emb_len))
    
    print(f'Running {model_name}. Datetime start: {datetime.today()}')
    for i, text in tqdm(enumerate(text_series), total = len(text_series)):

        # pre process data
        tokenized_text, tokens_tensor, segments_tensors = bert_text_preparation(text, tokenizer)
        
        # get embeddings
        list_token_embeddings = get_bert_embeddings(tokens_tensor, segments_tensors, model)

        # calculate mean
        bert_emb = np.array(list_token_embeddings).mean(axis=0)
        
        emb_list[i] = bert_emb

    print(f'End of Embedding. Datetime: {datetime.today()}')
    
    return emb_list