import pandas as pd
from tqdm import tqdm
tqdm.pandas()

import re

def omit_usernames(tweet):
    # Use regex to find all occurrences of @ followed by non-whitespace characters
    result = re.sub(r'@\w+', '@', tweet)
    return result


path_raw_data = 'data/raw/'
path_processed_data = 'data/processed/'


for split in ['train', 'test']:
    
    

    for target in tqdm([
        'bo',
        'cl',
        'co',
        'gl',
        'ig',
        'lu'
    ]):
        
        for dataset, text_col in [
            ('{split}_r3_{target}_top_mentioned_timelines', 'Texts'),
            ('r3_{target}_{split}_users', 'Timeline')
        ]:
            

            data = pd.read_csv(
                path_raw_data + dataset.format(split = split, target = target) + ".csv", 
                sep = ';', 
                encoding='utf-8-sig'
                )            
            data_filtered = data.copy()
            
            ####################
            # train and test process
            ####################
            
            # ommit @ of users
            data_filtered[text_col] = data_filtered[text_col].apply(omit_usernames)
            
            ####################
            # only train process
            ####################
            if split == 'train':

                # remove na comments from train
                data_filtered = data_filtered[~(data_filtered[text_col] == 'na')]
                
            data_filtered.to_csv(
                path_processed_data + dataset.format(split = split, target = target) + "_processed.csv",
                sep = ';', encoding='utf-8-sig')