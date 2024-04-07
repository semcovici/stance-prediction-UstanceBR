##########################
# Imports 
##########################
import pandas as pd
from tsfresh.feature_extraction import MinimalFCParameters
from tsfresh.feature_extraction import extract_features
from tqdm import tqdm
tqdm.pandas() 
import numpy as np
import os
from datetime import datetime

##########################
# Definitions
##########################
# path vars
path_processed_data = 'data/processed/'
path_interim_data = 'data/interim/'

# model config
#model_name = 'neuralmind/bert-base-portuguese-cased'
model_name = 'neuralmind/bert-base-portuguese-cased'
n_parts = 10

# corpus config
corpus = 'ig'

# settings tsfresh
settings = MinimalFCParameters()

check_if_file_exists = False

##########################
# Process
##########################

# there is a problem in this code that is a problema of embedding code.
# the same id appers in different parts

def main():
    
    last_user_id = None
    for part_id in range(1,n_parts + 1):
        
        # define paths
        path_input = path_interim_data + f'train_r3_{corpus}_separated_comments_{model_name.replace('/', '_')}_part_{part_id}.parquet'
        path_output = path_interim_data + f'train_r3_{corpus}_separated_comments_{model_name.replace("/", "_")}_tsfresh_part_{part_id}.parquet'

        print(f'###### START of {corpus} part {part_id} of {n_parts} ######')
        print(f'Time start: {datetime.now()}')

        # check if file already exists
        if os.path.isfile(path_output) and check_if_file_exists:
        
            print(f'{path_output} already exists')
            
            continue


        # get data
        data = pd.read_parquet(path_input) 
        
        # problem solving
        if last_user_id != None:
            data = pd.concat([df_temp, data])
            data.sort_values('User_ID', inplace = True)
        last_user_id = data.User_ID.unique().tolist()[-1]
        df_temp = data[data.User_ID == last_user_id]
        data = data[data.User_ID != last_user_id]
        

        # drop useless columns
        data_emb = data.drop(
            ['Texts', 'Polarity'],
            axis =1
        )

        # extract features
        X_tsfresh = extract_features(
            data_emb,
            column_id='User_ID',
            default_fc_parameters= settings
        )

        # # format dataset
        X_tsfresh_cplt = X_tsfresh.reset_index(drop=False).rename(columns={'index': 'User_ID'}).merge(
            data[['User_ID','Polarity']].drop_duplicates(),
            on = 'User_ID',
            how = 'left'
        )
        # X_tsfresh_cplt = X_tsfresh.reset_index().rename(columns={'index': 'User_ID'})
        # print(X_tsfresh_cplt.User_ID.unique())

        #save part
        X_tsfresh_cplt.to_parquet(path_output,index = False)
        
        print(f'Time end: {datetime.now()}')
        print(f'###### END of {corpus} part {part_id} of {n_parts} ######')
        
        
        
        
    print(f'Concating final df. Datetime start: {datetime.today()}')
    lista_df_parts = []
    for part_id in range(1,n_parts + 1):
        
        path_output = path_interim_data + f'train_r3_{corpus}_separated_comments_{model_name.replace("/", "_")}_tsfresh_part_{part_id}.parquet'
        
        df_part = pd.read_parquet(path_output)
        
        print(df_part.shape)
        
        lista_df_parts.append(df_part)
  
    df_final = pd.concat(lista_df_parts)
    print(f'Concating final df. Datetime end: {datetime.today()}')
    
    df_final.to_parquet(path_processed_data + f'train_r3_{corpus}_separated_comments_{model_name.replace("/", "_")}_tsfresh.parquet')

if __name__ == "__main__":
    main()
    


