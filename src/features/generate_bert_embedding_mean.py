from transformers import AutoTokenizer, BertModel
import torch
import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm
from transformers import set_seed
set_seed(42)

################ Data Paths ################################
path_raw_data = 'data/raw/'
path_processed_data = 'data/processed/'


DATA_INPUT_PATH = path_processed_data + 'data_filtered_separated.csv'
############################################################

MAX_LENGTH_TOKENS = 512


################ Aux Functions #############################
"""
ref: https://towardsdatascience.com/3-types-of-contextualized-word-embeddings-from-bert-using-transfer-learning-81fcefe3fe6d
"""
def bert_text_preparation(text, tokenizer):
    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text, truncation=True, max_length=MAX_LENGTH_TOKENS, padding = False)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1]*len(indexed_tokens)

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    return tokenized_text, tokens_tensor, segments_tensors

def get_bert_embeddings(tokens_tensor, segments_tensors, model):
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        hidden_states = outputs[2][1:]

    token_embeddings = hidden_states[-1]
    token_embeddings = torch.squeeze(token_embeddings, dim=0)
    list_token_embeddings = [token_embed.tolist() for token_embed in token_embeddings]

    return list_token_embeddings

def main():
    
    for model_size in ['base']:
        
        # Import models
        tokenizer = AutoTokenizer.from_pretrained(f'neuralmind/bert-{model_size}-portuguese-cased', do_lower_case=False)
        model = BertModel.from_pretrained(f'neuralmind/bert-{model_size}-portuguese-cased', output_hidden_states=True)

        # Get data
        data = pd.read_csv(DATA_INPUT_PATH, index_col = 0)
        data_bert = data.copy()

        emb_list = []
        print(f'Running neuralmind/bert-{model_size}-portuguese-cased. Datetime start: {datetime.datetime.today()}')
        for i, row in tqdm(data_bert.iterrows(), total = len(data_bert)):
            text = row['comment']
            tokenized_text, tokens_tensor, segments_tensors = bert_text_preparation(text, tokenizer)
            list_token_embeddings = get_bert_embeddings(tokens_tensor, segments_tensors, model)

            bert_emb = np.array(list_token_embeddings).mean(axis=0)
            bert_emb = np.concatenate([row, bert_emb])
            emb_list.append(bert_emb)

        print(f'End of Embedding. Datetime: {datetime.datetime.today()}')

        columns = np.concatenate([data_bert.columns, [f"emb_{i + 1}" for i in range(len(emb_list[0]) - data.shape[1])]])
        df_bert = pd.DataFrame(emb_list, columns=columns)
        df_bert.to_parquet(path_processed_data + f'data_bert-{model_size}-portuguese-cased_mean__max_lenght={MAX_LENGTH_TOKENS}.parquet', index=False)

if __name__ == "__main__":
    main()