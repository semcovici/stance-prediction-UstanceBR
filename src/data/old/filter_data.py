import pandas as pd
from tqdm import tqdm


path_raw_data = 'data/raw/'
path_processed_data = 'data/processed/'

def main():
    
    for corpus in tqdm([
        'bo',
        'cl',
        'co',
        'gl',
        'ig',
        'lu'
    ]):
    
        data = pd.read_csv(
            path_raw_data + f'train_r3_{corpus}_top_mentioned_timelines.csv', 
            sep = ';', 
            encoding='utf-8-sig'
            )
        data_filtered = data.drop_duplicates(subset=['Texts'])
        data_filtered.to_csv(path_processed_data + f'train_r3_{corpus}_filtered.csv')
    
if __name__ == "__main__":
    main()

