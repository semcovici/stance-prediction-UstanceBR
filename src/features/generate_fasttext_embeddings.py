##########################
# Imports 
##########################
import fasttext
from huggingface_hub import hf_hub_download
import pandas as pd
from nltk.tokenize import word_tokenize
import numpy as np
from tqdm import tqdm 
import os
tqdm.pandas()

##########################
# Aux Functions 
##########################
def get_mean_embeddings(text, model):
        
    try: 
        # tokenize
        tokens = word_tokenize(text, language='portuguese')
        
        # get mean embeddings 
        mean_embeddings = np.array([model[str(tk)] for tk in tokens]).mean(axis = 0)
    except Exception as e:
         
         print('-------------- WARNING --------------')
         print('An exception occurred')
         print(e)
         
         print('The text input is', text)
         print('-------------- WARNING --------------')
         
         print(text)
         
         
         
         mean_embeddings = None
    
    
    return mean_embeddings


##########################
# Definitions
##########################
model_name = 'facebook/fasttext-pt-vectors'

path_processed_data = 'data/processed/'

check_if_file_exists = True


##########################
# Process
##########################
def main():
    
    # load model
    model_path = hf_hub_download(repo_id="facebook/fasttext-pt-vectors", filename="model.bin")
    model = fasttext.load_model(model_path)
    
    list_corpus = ['ig','bo', 'cl', 'co', 'gl', 'lu']
    
    for i, corpus in enumerate(list_corpus):
        
        print(f'###### START {corpus} ({i + 1} of {len(list_corpus)}) ######')
        
        path_output = path_processed_data + f'train_r3_{corpus}_separated_comments_{model_name.replace('/', '_')}.parquet'
        if os.path.isfile(path_output) and check_if_file_exists: 
            
            print('The file exists')
            print("File name: ",path_output)
            
        else:
            
            # get data
            data = pd.read_csv(path_processed_data + f'train_r3_{corpus}_separated_comments.csv')

            # create column with embeddings
            data['mean_embeddings'] = data.Texts.progress_apply(lambda x: get_mean_embeddings(x, model))
            
            # save dataset
            data.to_parquet(path_output,index = False)
            
        print(f'###### END {corpus} ({i + 1} of {len(list_corpus)}) ######')


if __name__ == '__main__':
    main()