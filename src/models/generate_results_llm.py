# !ollama pull llama3
# !ollama pull llama3:70b

import ollama
import pandas as pd
from tqdm import tqdm
from ast import literal_eval
tqdm.pandas()

import os


from sklearn.metrics import classification_report, precision_score, f1_score

import sys
sys.path.append('src/')
from models.classification_methods import get_classification_report,create_test_results_df
from data.lambdas import int_to_label, label_to_int

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

file_format_users = raw_data_path +'r3_{target}_test_users.csv'
file_format_users_filtered = processed_data_path + 'r3_{target}_test_users_scored_Timeline.csv' 
file_format_tmt = raw_data_path +'test_r3_{target}_top_mentioned_timelines.csv'
file_format_tmt_filtered = processed_data_path + 'test_r3_{target}_top_mentioned_timelines_scored_Texts.csv'


dict_experiments = {
    
    'Stance': {
        "text_col": 'Stance',
        "prompts_to_test": ['prompt2_Stance'],
        "is_multi_text": False,
        "file_format": file_format_users
    },
    
    'filtered_Texts5': {
        "text_col": 'Texts',
        "prompts_to_test": ['prompt2_Texts'],
        "is_multi_text": True,
        "n_comments": 5,
        "file_format": file_format_tmt_filtered
    },
    'filteredTimeline5': {
        "text_col": 'Timeline',
        "prompts_to_test": ['prompt2_Timeline'],
        "is_multi_text": True,
        "n_comments": 5,
        "file_format": file_format_users_filtered
    },
    'filtered_Texts10': {
        "text_col": 'Texts',
        "prompts_to_test": ['prompt2_Texts'],
        "is_multi_text": True,
        "n_comments": 10,
        "file_format": file_format_tmt_filtered
    },
    'filteredTimeline10': {
        "text_col": 'Timeline',
        "prompts_to_test": ['prompt2_Timeline'],
        "is_multi_text": True,
        "n_comments": 10,
        "file_format": file_format_users_filtered
    },
    
    'filtered_Texts15': {
        "text_col": 'Texts',
        "prompts_to_test": ['prompt2_Texts'],
        "is_multi_text": True,
        "n_comments": 15,
        "file_format": file_format_tmt_filtered
    },
    'filteredTimeline15': {
        "text_col": 'Timeline',
        "prompts_to_test": ['prompt2_Timeline'],
        "is_multi_text": True,
        "n_comments": 15,
        "file_format": file_format_users_filtered
    },
    
    'filtered_Texts20': {
        "text_col": 'Texts',
        "prompts_to_test": ['prompt2_Texts'],
        "is_multi_text": True,
        "n_comments": 20,
        "file_format": file_format_tmt_filtered
    },
    'filteredTimeline20': {
        "text_col": 'Timeline',
        "prompts_to_test": ['prompt2_Timeline'],
        "is_multi_text": True,
        "n_comments": 20,
        "file_format": file_format_users_filtered
    },

    
    'filtered_Texts30': {
        "text_col": 'Texts',
        "prompts_to_test": ['prompt2_Texts'],
        "is_multi_text": True,
        "n_comments": 30,
        "file_format": file_format_tmt_filtered
    },
    'filteredTimeline30': {
        "text_col": 'Timeline',
        "prompts_to_test": ['prompt2_Timeline'],
        "is_multi_text": True,
        "n_comments": 30,
        "file_format": file_format_users_filtered
    },
    

    
    'filtered_Texts40': {
        "text_col": 'Texts',
        "prompts_to_test": ['prompt2_Texts'],
        "is_multi_text": True,
        "n_comments": 40,
        "file_format": file_format_tmt_filtered
    },
    'filteredTimeline40': {
        "text_col": 'Timeline',
        "prompts_to_test": ['prompt2_Timeline'],
        "is_multi_text": True,
        "n_comments": 40,
        "file_format": file_format_users_filtered
    }

}

#############################
# Aux Functions
#############################
def get_response_from_llm(prompt):
    response_full = ollama.generate(
        model=estimator_name, 
        prompt = prompt, 
        options = {
            'seed': random_seed,
            'temperature': 0,
            "num_gpu": 1,
            "main_gpu": 0,
            }
        )
    return response_full


def format_response(
    response,
    target,
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
        
        # when get an error, the prediction is the most common in train
        y_pred = dict_majority[target]
        
    return message, response, y_pred

def get_prompt(prompt_name):
    
    with open(f'src/models/config/prompts/{prompt_name}.txt', 'r') as file:
        
        prompt_template = file.read()
        
    return prompt_template 


check_if_already_exists = True

print("Creating dict with majority polarity in train")
dict_majority = {}
for target in tqdm(target_list):
    
    df = pd.read_csv( f"data/raw/r3_{target}_train_users.csv", sep = ';', encoding='utf-8-sig')
    majority = label_to_int(df.Polarity.value_counts().idxmax())
    dict_majority.update({target:majority})




#############################
# Process
#############################
for exp_name, config in dict_experiments.items():
    
    print(f"""####################################  
# Running {exp_name}
#####################################""")
    
    
    # get configs of experiments
    text_col = config['text_col']
    prompts_to_test = config['prompts_to_test']
    is_multi_text = config['is_multi_text']
    file_format = config['file_format']
    
    
    data_list = []
    for target in target_list:
        
        # read data
        data_aux = pd.read_csv( file_format.format(target=target), sep = ';', encoding='utf-8-sig')
        
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
        input_list = []
        output_list = [] 

        for target in target_list:
            
            target_original = target
            
            output_file = f'{reports_path}test_results/{estimator_name}_{target_original}_{exp_name}_{prompt_name}_test_results.csv'
            
            
            if os.path.isfile(output_file) and check_if_already_exists:
                print('# experiment already done')
                continue
            
            data = test_df[test_df['target'] == target]
                                    
            # if is multi_text, filter only the best n comments
            if is_multi_text:
                n_comments = config['n_comments'] 
                data[f'comments_and_scores_{text_col}'] = data[f'comments_and_scores_{text_col}'].progress_apply(lambda x: literal_eval(x))
                data[text_col] = data[f'comments_and_scores_{text_col}'].progress_apply(
                    lambda x: " # ".join([comment for score, comment in x[-n_comments:]])
                    ) 
            
            list_polarity_pred = []
            pred_proba_0 = []
            pred_proba_1 = []
            list_message = []
            for idx, row in tqdm(data.iterrows(), total = len(data), desc = target):
                
                text = row[text_col]
                
                target = dict_cp.get(row['target'])
                polarity = row["Polarity"]
                polarity = 1 if polarity == 'for' else 0
                
                
                if not is_multi_text:
                
                    prompt_formated = prompt_template.format(
                    target = target,
                    text = text)
                    
                else: 
                    
                    # create list with comments and get the firt n comments
                    try:
                        comments = text.split(' # ')
                        comments_filtered =  comments[:n_comments]
                        
                    except Exception as e:
                        
                        comments = []
                    
                    texts = ''
                    for c in comments_filtered:
                        
                        texts += '<comment>\n'
                        texts += c
                        texts += '\n</comment>'
                                        
                    prompt_formated = prompt_template.format(
                    target = target,
                    text = texts)
                    
                input_list.append(prompt_formated)
                response_full = get_response_from_llm(prompt_formated)
                output_list.append(response_full)
                
                message, response, y_pred = format_response(response_full, target)
                
                # create probas
                if y_pred is not None:
                    if y_pred == 0:
                        proba_0 = response
                        proba_1 = 1 - response
                    elif y_pred == 1:
                        proba_1 = response
                        proba_0 = 1 - response
                    else:
                        raise Exception("Erro")
                    
                else:
                    proba_1 = -1
                    proba_0 = -1
                    
                list_message.append(message)
                list_polarity_pred.append(y_pred)
                
                pred_proba_0.append(float(proba_0))
                pred_proba_1.append(float(proba_1))
                
            
            y_test = data['Polarity'].apply(lambda x: label_to_int(x)).tolist()
            
            
            # format test and pred
            y_test_formated = [int_to_label(test) for test in y_test]
            y_pred_formated = [int_to_label(pred) for pred in list_polarity_pred]
            
            # create df with results
            df_test_results = create_test_results_df(
                y_test_formated, 
                y_pred_formated, 
                pred_proba_0, 
                pred_proba_1
                )
            
            df_test_results['message'] = list_message 
            df_test_results['input'] = input_list 
            df_test_results['output'] = output_list 
            
            print("Results in ", output_file)
            
            df_test_results.to_csv(output_file, index = False)