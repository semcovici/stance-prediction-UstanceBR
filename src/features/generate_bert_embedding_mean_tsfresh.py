from transformers import AutoTokenizer, BertModel
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
from transformers import set_seed
from itertools import product
set_seed(42)
from tsfresh.feature_extraction import MinimalFCParameters
from tsfresh.feature_extraction import extract_features

from bert_embedding_methods import *

################ Data Paths ################################
path_raw_data = 'data/raw/'
path_processed_data = 'data/processed/'
path_interim_data = 'data/interim/'
check_if_file_exists = True
############################################################
# settings tsfresh
settings = MinimalFCParameters()
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
            path_data_input = path_processed_data + f'{split}_r3_{target}_top_mentioned_timelines_separated_comments.csv'
            path_output_base_parts = path_interim_data + f'{split}_r3_{target}_top_mentioned_timelines_separated_comments_{model_name.replace("/", "_")}_tsfresh'
            path_data_output = path_processed_data + f'{split}_r3_{target}_top_mentioned_timelines_separated_comments_{model_name.replace("/", "_")}_tsfresh.parquet'
            
            text_col = 'Texts'
        if dataset == 'users':
            path_data_input = path_processed_data + f'r3_{target}_{split}_users_separated_comments.csv'
            path_output_base_parts = path_interim_data + f'r3_{target}_{split}_users_separated_comments_{model_name.replace("/", "_")}_tsfresh'
            path_data_output = path_processed_data + f'r3_{target}_{split}_users_separated_comments_{model_name.replace("/", "_")}_tsfresh.parquet'
            
            text_col = 'Timeline'
            
        path_parts = path_output_base_parts + '_{user_id}.parquet'
        
        # Get data
        data = pd.read_csv(path_data_input)
        
        users = data.User_ID.unique()
        
        list_df_tsfresh = []
        for i, user_id in tqdm(enumerate(users), total = len(users)):
                
            data_user = data[data.User_ID == user_id].reset_index(drop = True)
            
            emb_list = create_bert_embeddings_mean_from_series(
                model_name = model_name,
                text_series = data_user[text_col],
                progress_bar = False
            )
            
            columns = [f"{text_col}_emb_{i + 1}" for i in range(emb_list.shape[1])]
            df_emb = pd.DataFrame(emb_list, columns=columns)
        
            data_user_emb = pd.concat([data_user, df_emb], axis = 1)

            # extract features
            X_tsfresh = extract_features(
                data_user_emb.drop(['Texts', 'Polarity', 'id_text'],axis =1),
                column_id='User_ID',
                default_fc_parameters= settings,
                disable_progressbar=True
            )

            # # format dataset
            X_tsfresh_cplt = X_tsfresh.reset_index(drop=False).rename(columns={'index': 'User_ID'}).merge(
                data_user_emb[['User_ID','Polarity']].drop_duplicates(),
                on = 'User_ID',
                how = 'left'
            )
            
            list_df_tsfresh.append(X_tsfresh_cplt)
            
        print(f'##### END PROCESS - {datetime.today()} #####')    
        
        df_tsfresh = pd.concat(list_df_tsfresh)
        df_tsfresh.to_parquet(path_data_output)
              
if __name__ == "__main__":
    main()