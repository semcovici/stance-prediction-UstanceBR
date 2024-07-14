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
path_interim_data = 'data/interim/'
check_if_file_exists = True
############################################################

n_parts = 10

################ Options for generation ####################
model_names = [
    #'neuralmind/bert-large-portuguese-cased', 
    'neuralmind/bert-base-portuguese-cased'
    ]
list_target = [
    'ig', 
    # 'bo', 
    # 'cl', 
    # 'co', 
    # 'gl', 
    # 'lu'
    ]
list_splits = [
    'train', 
    'test'
    ]
list_datasets = [
    'top_mentioned_timelines',
    # 'users'
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
            path_data_input = path_processed_data + f'{split}_r3_{target}_top_mentioned_timelines_separated_comments_unique.csv'
            path_output_base_parts = path_interim_data + f'{split}_r3_{target}_top_mentioned_timelines_separated_comments_unique_{model_name.replace("/", "_")}'
            path_data_output = path_processed_data + f'{split}_r3_{target}_top_mentioned_timelines_separated_comments_unique_{model_name.replace("/", "_")}.parquet'
            
            text_col = 'Texts'
        if dataset == 'users':
            path_data_input = path_processed_data + f'r3_{target}_{split}_users_separated_comments_unique.csv'
            path_output_base_parts = path_interim_data + f'r3_{target}_{split}_users_separated_comments_unique_{model_name.replace("/", "_")}'
            path_data_output = path_processed_data + f'r3_{target}_{split}_users_separated_comments_unique_{model_name.replace("/", "_")}.parquet'
            
            
            text_col = 'Timeline'
        
        # Get data
        data = pd.read_csv(path_data_input)
        
        print(f'##### START PROCESS EMBEDDING - {datetime.today()} #####')      
            
        # file file with embeddings                 
        create_bert_embeddings_mean_from_series_by_parts(
            model_name = model_name,
            text_series = data[text_col],
            n_parts = n_parts,
            check_if_file_exists = True,
            path_output_base = path_output_base_parts
        )
        
        print(f'##### END PROCESS EMBEDDING- {datetime.today()} #####')      
        
        gc.collect()
        
        df_emb = pd.DataFrame({})
        for part_idx in range(n_parts):
                        
            output_file = f'{path_output_base_parts}_part_{part_idx+1}.npy' 
            
            data_part = np.load(output_file)            
            
            columns = [f"{text_col}_emb_{i + 1}" for i in range(data_part.shape[1])]
            df_emb_part = pd.DataFrame(data_part, columns=columns)
            
            df_emb_part.to_parquet(f'{path_output_base_parts}_part_{part_idx+1}.parquet')
            
            df_emb = pd.concat([df_emb,df_emb_part])
            
            del df_emb_part
            del data_part
            gc.collect()
  
        data = pd.concat([
        data.reset_index(drop = True), 
        df_emb.reset_index(drop = True)
        ], axis = 1)
        
        data.to_parquet(path_data_output, index = False)
        
        print(f'##### END PROCESS - {datetime.today()} ##### \n\n\n\n')
              
if __name__ == "__main__":
    main()