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
        
        # Separa os textos em linhas individuais
        df_sep_comments = data.assign(Texts=data['Texts'].str.split(' # ')).explode('Texts')

        # Reindexa o DataFrame resultante
        df_sep_comments.reset_index(drop=True, inplace = True)

        df_sep_comments.ffill(inplace = True)
        
        df_sep_comments.to_csv(path_processed_data + f'train_r3_{corpus}_separated_comments.csv', index = False)
    
if __name__ == "__main__":
    main()

