from transformers import AutoTokenizer, BertModel
import torch
import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm
from transformers import set_seed
import os
import gc
set_seed(42)

################ Data Paths ################################
path_raw_data = 'data/raw/'
path_processed_data = 'data/processed/'
path_interim_data = 'data/interim/'
check_if_file_exists = True
############################################################


def main():
    
    for model_name in [
        f'neuralmind/bert-base-portuguese-cased'
        ]:
        
            
        list_corpus = ['ig','bo', 'cl', 'co', 'gl', 'lu']
        
        for i, corpus in enumerate(list_corpus):
            
            print(f'###### START {corpus} ({i + 1} of {len(list_corpus)}) ######')
            
            path_output_base = path_interim_data + f'train_r3_{corpus}_separated_comments_{model_name.replace("/", "_")}'
        
            # Import models
            tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
            model = BertModel.from_pretrained(model_name, output_hidden_states=True).to('cuda')
            
            # Get data
            data = pd.read_csv(path_processed_data + f'train_r3_{corpus}_separated_comments.csv')
            data_bert = data.copy()
            
            # Split data into 10 parts
            data_parts = np.array_split(data_bert, 10)

            emb_list = []
            print(f'Running {model_name}. Datetime start: {datetime.datetime.today()}')
            for part_idx, part_data in enumerate(data_parts):
                output_file = f'{path_output_base}_part_{part_idx+1}.parquet'
                if check_if_file_exists and os.path.exists(output_file):
                    print(f'Skipping part {part_idx+1} as it already exists.')
                    continue
                
                print(f'Processing part {part_idx+1} of {len(data_parts)}')
                part_emb_list = []
                for i, row in tqdm(part_data.iterrows(), total=len(part_data)):
                    text = row['Texts']
                    tokenized_text, tokens_tensor, segments_tensors = bert_text_preparation(text, tokenizer)
                    list_token_embeddings = get_bert_embeddings(tokens_tensor, segments_tensors, model)

                    bert_emb = np.array(list_token_embeddings).mean(axis=0)
                    bert_emb = np.concatenate([row, bert_emb])
                    part_emb_list.append(bert_emb)

                columns = np.concatenate([part_data.columns, [f"emb_{i + 1}" for i in range(len(part_emb_list[0]) - data.shape[1])]])
                df_bert = pd.DataFrame(part_emb_list, columns=columns)
                df_bert.to_parquet(output_file, index=False)

                # Liberando memória
                del part_emb_list
                del part_data
                del df_bert
                gc.collect()
                
            print(f'End of Embedding. Datetime: {datetime.datetime.today()}')
            
if __name__ == "__main__":
    main()
