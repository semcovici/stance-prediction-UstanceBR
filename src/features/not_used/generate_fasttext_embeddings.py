##########################
# Imports 
##########################
import gc
import fasttext
from huggingface_hub import hf_hub_download
import pandas as pd
from nltk.tokenize import word_tokenize
import numpy as np
from tqdm import tqdm 
import os
from datetime import datetime, timedelta

tqdm.pandas()




##########################
# Definitions
##########################
model_name = 'facebook/fasttext-pt-vectors'
emb_dim = 300
# load model
model_path = hf_hub_download(repo_id="facebook/fasttext-pt-vectors", filename="model.bin")
model = fasttext.load_model(model_path)

path_processed_data = 'data/processed/'

check_if_file_exists = False


##########################
# Aux Functions 
##########################0.371501
def get_mean_embeddings(row):
        
    # tokenize
    tokens = word_tokenize(row.Texts, language='portuguese')
    
    # get mean embeddings 
    mean_embeddings = np.array([model[str(tk)] for tk in tokens]).mean(axis = 0)
    
    row = row._append(pd.Series(mean_embeddings, index=[f'emb_{i}' for i in range(emb_dim)]))
            
    return row


##########################
# Process
##########################
def main():
    

    
    list_corpus = ['ig','bo', 'cl', 'co', 'gl', 'lu']
    
    for i, corpus in enumerate(list_corpus):
        
        print(f'###### START {corpus} ({i + 1} of {len(list_corpus)}) ######')
        
        path_output = path_processed_data + f'train_r3_{corpus}_separated_comments_{model_name.replace("/", "_")}.parquet'
        if os.path.isfile(path_output) and check_if_file_exists: 
            
            print('The file exists')
            print("File name: ",path_output)
            
        else:
            
            # get data
            print('Getting data ', datetime.now())
            data = pd.read_csv(path_processed_data + f'train_r3_{corpus}_separated_comments.csv')

            chunk_size = 1000  # You can adjust this chunk size as needed
            chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
            
            del data
            gc.collect()
            processed_chunks = []
            for chunk in tqdm(chunks, desc='Processing chunks'):
                
                temp_df = chunk.apply(get_mean_embeddings, axis = 1)
                
                processed_chunks.append(temp_df)
                
                del temp_df
                gc.collect()
                
            data = pd.concat(processed_chunks)
            
            # save dataset
            print('saving in a file')
            data.to_parquet(path_output,index = False)
            
        print(f'###### END {corpus} ({i + 1} of {len(list_corpus)}) ######')


if __name__ == '__main__':
    main()