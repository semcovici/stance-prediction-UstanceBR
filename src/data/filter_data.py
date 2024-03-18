import pandas as pd


path_raw_data = 'data/raw/'
path_processed_data = 'data/processed/'

data = pd.read_csv(
    path_raw_data + 'train_r2_bo_top_mentioned_timelines.csv', 
    sep = ';', 
    encoding='utf-8-sig'
    )


data_filtered = data.drop_duplicates(subset=['Texts'])


data_filtered.to_csv(path_processed_data + 'data_filtered.csv')

