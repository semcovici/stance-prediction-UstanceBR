# %%
import ollama

import pandas as pd
from tqdm import tqdm
import ast

from sklearn.metrics import classification_report, precision_score, f1_score

import sys
sys.path.append('src/')
from models.classification_methods import get_classification_report

random_seed = 42

raw_data_path = 'data/raw/'
processed_data_path = 'data/processed/'
results_cr_path = 'reports/classification_reports/'
test_results_path = 'reports/test_results/'
reports_path = 'reports/'

target_list = ['ig','bo', 'cl', 'co', 'gl', 'lu']

estimator_name = 'llama3'

with open('src/models/config/prompts/prompt1.txt', 'r') as file:
    
    prompt_template = file.read() 

data_list = []

for target in target_list:
    
    # read data
    data_temp = pd.read_csv(
        raw_data_path + f'r3_{target}_test_users.csv', 
        sep = ';', 
        encoding='utf-8-sig'
        )
    
    data_temp['target'] = target
    
    data_list.append(data_temp)
    
data_users = pd.concat(data_list)

dict_cp = {
    'cl':'Hidroxicloroquina',
    'lu':'Lula',
    'co':'Sinovac',
    'ig':'Church',
    'gl':'Globo TV',
    'bo':'Bolsonaro',
}

def get_response_from_llm(prompt):
    
    
    response_full = ollama.chat(model=estimator_name, messages=[{
            'role':'user',
            'content':prompt
            }])
    
    
    return response_full

def format_response(response):
    
    message = response["message"]["content"]
    
    try:
        # string dict to dict
        response = ast.literal_eval(message)
        y_pred = response['classification'].casefold()
        
        if y_pred == "against":
            y_pred = 0
        elif y_pred == "for":
            y_pred = 1
        else: y_pred = None
            
    except Exception as e:
        y_pred = None 
        
    return message, y_pred


dict_responses = {}
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


    data = data_users[data_users['target'] == target]

    text_col = 'Stance'
    
    cr_path = f"{reports_path}classification_reports/{estimator_name}_{target}_{text_col}_classification_report.csv"
    
    for idx, row in tqdm(data.iterrows(), total = len(data)):
        
        
        
        
        text = row['Stance']
        target = dict_cp.get(row['target'])
        polarity = row["Polarity"]
        polarity = 1 if polarity == 'for' else 0
        
        prompt_formated = prompt_template.format(
        target = target,
        text = text)
        
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
        
    dict_responses.update({target: df_responses})
    
    df_cr = get_classification_report(df_responses.y_test, df_responses.y_pred)
    
    df_cr.to_csv(cr_path)


