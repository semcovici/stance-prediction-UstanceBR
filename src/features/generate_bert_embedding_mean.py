from transformers import AutoTokenizer, BertModel
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
from transformers import set_seed
from itertools import product
set_seed(42)

from bert_embedding_methods import *

################ Data Paths ################################
path_raw_data = 'data/raw/'
path_processed_data = 'data/processed/'
############################################################

################ Options for generation ####################
model_names = [
    #'neuralmind/bert-large-portuguese-cased', 
    # 'neuralmind/bert-base-portuguese-cased',
    'pablocosta/bertabaporu-base-uncased'
    ]
list_target = [
    'ig', 
    'bo', 
    'cl', 
    'co', 
    'gl', 
    'lu'
    ]
list_splits = [
    'train', 
    'test'
    ]
list_datasets = [
    'top_mentioned_timelines',
    'users'
]
############################################################

def main():
    
    # iterate for the options
    for model_name, target, split, dataset in product(model_names, list_target, list_splits, list_datasets):
        
        print(f'##### START PROCESS - {datetime.today()} #####')
        print(f'''
Configuration:
    - model_name: {model_name}
    - target: {target}
    - split: {split}
    - dataset: {dataset}
              ''')
        
        if dataset == 'top_mentioned_timelines':
            #path_data_input = path_raw_data + f'{split}_r3_{target}_top_mentioned_timelines.csv'
            path_data_input = path_processed_data + f'{split}_r3_{target}_top_mentioned_timelines_processed.csv'
            path_data_output = path_processed_data + f'{split}_r3_{target}_top_mentioned_timelines_{model_name.replace("/", "_")}.parquet'
            texts_cols = ['Texts']
        if dataset == 'users':
            #path_data_input = path_raw_data + f'r3_{target}_{split}_users.csv'
            path_data_input = path_processed_data + f'r3_{target}_{split}_users_processed.csv'
            path_data_output = path_processed_data + f'r3_{target}_{split}_users_{model_name.replace("/", "_")}.parquet'
            texts_cols = ['Timeline', 'Stance']
        
        # Get data
        data = pd.read_csv(
            path_data_input, 
            sep = ';', 
            encoding='utf-8-sig'
            )
        
        for col in texts_cols:
        
            print(f'Running BERT Embedding on {col} column')
            
            emb_list = create_bert_embeddings_mean_from_series(
                model_name = model_name,
                text_series = data[col]
            )
            
            columns = [f"{col}_emb_{i + 1}" for i in range(emb_list.shape[1])]
            df_emb = pd.DataFrame(emb_list, columns=columns)
        
            data = pd.concat([data, df_emb], axis = 1)
        
        data.to_parquet(path_data_output, index=False)   
        
        print(f'##### END PROCESS - {datetime.today()} ##### \n\n\n')       
              
if __name__ == "__main__":
    main()