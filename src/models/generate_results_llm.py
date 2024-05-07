# !ollama pull llama3
# !ollama pull llama3:70b

import ollama
import pandas as pd
from tqdm import tqdm
import ast

from sklearn.metrics import classification_report, precision_score, f1_score

import sys
sys.path.append('src/')
from models.classification_methods import get_classification_report

#############################
# Definitions
#############################
random_seed = 42

raw_data_path = 'data/raw/'
processed_data_path = 'data/processed/'
results_cr_path = 'reports/classification_reports/'
test_results_path = 'reports/test_results/'
reports_path = 'reports/'

target_list = [
    'ig',
    'bo', 
    'cl', 
    'co', 
    'gl', 
    'lu'
    ]

estimator_name = 'llama3'
#estimator_name = 'llama3:70b'

dict_cp = {
    'cl':'Hidroxicloroquina',
    'lu':'Lula',
    'co':'Sinovac',
    'ig':'Church',
    'gl':'Globo TV',
    'bo':'Bolsonaro',
}

file_format_users = 'r3_{target}_test_users.csv'
file_format_tmt = 'test_r3_{target}_top_mentioned_timelines.csv'

dict_experiments = {
    'Texts': {
        "text_col": 'Texts',
        "prompts_to_test": ['prompt2_Texts'],
        "is_multi_text": True,
        "n_comments": 30,
        "file_format": file_format_tmt
    },
    'Timeline': {
        "text_col": 'Timeline',
        "prompts_to_test": ['prompt2_Timeline'],
        "is_multi_text": True,
        "n_comments": 30,
        "file_format": file_format_users
    },
    'Stance': {
        "text_col": 'Stance',
        "prompts_to_test": ['prompt2_Stance'],
        "is_multi_text": False,
        "file_format": file_format_users
    }
}

#############################
# Aux Functions
#############################
def get_response_from_llm(prompt):
    response_full = ollama.generate(model=estimator_name, prompt = prompt)
    return response_full


def format_response(
    response,
    threshold = 0.5
    ):
    
    message = response['response']
        
    try:
        # string dict to dict
        response = eval(message)
        
        if response < threshold:
            y_pred = 0
        else:
            y_pred = 1
            
    except Exception as e:
        y_pred = None 
        
    return message, y_pred

def get_prompt(prompt_name):
    
    with open(f'src/models/config/prompts/{prompt_name}.txt', 'r') as file:
        
        prompt_template = file.read()
        
    return prompt_template 




#############################
# Process
#############################


for exp_name, config in dict_experiments.items():
    
    # get configs of experiments
    text_col = config['text_col']
    prompts_to_test = config['prompts_to_test']
    is_multi_text = config['is_multi_text']
    file_format = config['file_format']
    
    data_list = []
    for target in target_list:
        
        # read data
        data_aux = pd.read_csv(raw_data_path + file_format.format(target=target), sep = ';', encoding='utf-8-sig')
        
        data_aux['target'] = target
        
        data_list.append(data_aux)
    
    # create final test df
    test_df = pd.concat(data_list)

    # test all prompts
    for prompt_name in prompts_to_test:
        
        # get prompt template from file
        prompt_template = get_prompt(prompt_name)

        dict_responses = {}

        list_results = [] 

        list_df_responses = [] 

        for target in target_list:

            df_responses = pd.DataFrame({
                "idx":[],
                "text":[],
                "target":[],
                "y_test":[],
                "y_pred":[],
                "justification":[],
                "complete_response": []
            })
            
            data = test_df[test_df['target'] == target]    
                
            for idx, row in tqdm(data.iterrows(), total = len(data), desc = target):
                
                text = row[text_col]
                target_id = target
                target = dict_cp.get(row['target'])
                polarity = row["Polarity"]
                polarity = 1 if polarity == 'for' else 0
                
                if not is_multi_text:
                
                    prompt_formated = prompt_template.format(
                    target = target,
                    text = text)
                    
                else: 
                    
                    n_comments = config['n_comments']
                    
                    # create list with comments and get the firt n comments
                    comments = text.split(' # ')
                    comments_filtered =  comments[:n_comments]
                    
                    texts = ''
                    for c in comments_filtered:
                        
                        texts += '<comment>\n'
                        texts += c
                        texts += '\n</comment>'
                                        
                    prompt_formated = prompt_template.format(
                    target = target,
                    text = texts)
                    
                
                response_full = get_response_from_llm(prompt_formated)
                
                message, y_pred = format_response(response_full)

                new_row = {
                "idx": idx,
                "text":text,
                "target":target,
                "y_test": polarity,
                "y_pred":y_pred,
                "complete_response": message
                
                }
                
                df_responses.loc[len(df_responses)] = new_row
                
            df_responses['target'] = target
            
            list_df_responses.append(df_responses)
            
        df_results_final = pd.concat(list_df_responses)   

        df_results_final.to_csv(f'{reports_path}test_results/{estimator_name}_{exp_name}_{prompt_name}_classification_report.csv')