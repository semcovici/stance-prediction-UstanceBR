######################
# Imports
######################
import pandas as pd
from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.tokenize import word_tokenize
import nltk
from unidecode import unidecode
import numpy as np

tqdm.pandas()


######################
# Definitions
######################

L = 15

path_raw_data = 'data/raw/'
path_processed_data = 'data/processed/'

######################
# Lists
######################
terms_list_ig = [
    "vaticano",
    "crisma",
    "comunhão",
    "batismo",
    "culto",
    "missa",
    "hóstia",
    "cálice",
    "crucifixo",
    "altar",
    "sacerdote",
    "papa",
    "bispo",
    "paróquia",
    "templo",
    "capela",
    "catedral",
    "pastor",
    "padre",
    "igreja"]


terms_list_cl = [
    "droga",
    "antimalárico",
    "tratamento",
    "medicamento",
    "remédio",
    "hidroxicloroquina",
    "cloroquina"]


terms_list_lu = [
    "13",
    "política",
    "governo",
    "ex-presidente",
    "luiz inácio lula da silva",
    "partido dos trabalhadores",
    "presidente",
    "pt",
    "lula"]


terms_list_co = [
    "china",
    "pandemia",
    "covid-19",
    "biontech",
    "vacinação",
    "imunização",
    "vacina",
    "vachina",
    "coronavac",
    "sinovac"]


terms_list_gl = [
    "jornalismo",
    "mídia",
    "emissora",
    "televisão",
    "tv",
    "globo"]


terms_list_bo = [
    "17",
    "22",
    "ex-presidente",
    "conservador",
    "política",
    "pl",
    "partido liberal",
    "governo bolsonaro",
    "presidente",
    "jair",
    "bolsonaro"
]

target_terms_dict = {
    'ig': [term.casefold() for term in terms_list_ig],
    'bo': [term.casefold() for term in terms_list_bo], 
    'cl': [term.casefold() for term in terms_list_cl], 
    'co': [term.casefold() for term in terms_list_co], 
    'gl': [term.casefold() for term in terms_list_gl], 
    'lu': [term.casefold() for term in terms_list_lu]
}


######################
# Functions
######################
# Function to tokenize sentences
def tokenize_sentences(sentences):
    # Split sentences into lists of words
    tokenized = np.char.split(sentences)
    return tokenized
# given comments separated by " # " and a list of terms, 
# return all coments that have at least one of terms in the terms_list
def find_relevant_comments(comments, terms_list, L=None):

    # Tokenização dos comentários
    list_comments = np.array(comments.split(' # '))
    tokenized_comments = tokenize_sentences(list_comments) 
    
    func = lambda tokenized_comment: o if (o:= np.where(np.isin(terms_list, tokenized_comment) == 1)[0].max(initial = -100000)) else -1
    vfunc = np.vectorize(func)
    score_com = vfunc(tokenized_comments)
    
    sorted_score_com = sorted(zip(score_com, list_comments))
    
    if L is not None:
        
        sorted_score_com = sorted_score_com[-L:]
    
    #sorted_scores, sorted_com=list(zip(*sorted_score_com))      
    # Concatenação dos comentários relevantes
    #str_rel_comments = ' # '.join(sorted_com) if sorted_com else ''
    return sorted_score_com

splits = [
    "test",
    "train" 
    ]
datasets = {
    "users": {
        "path_input_format":path_processed_data + 'r3_{target}_{split}_users_processed.csv', 
        "path_output_format":path_processed_data + 'r3_{target}_{split}_users_scored_Timeline.csv', 
        "path_output_format_L":path_processed_data + 'r3_{target}_{split}_users_scored_Timeline' + f'_L={L}_.csv', 
        "text_col": "Timeline"
    },
    "tmt":{
        "path_input_format":path_processed_data + '{split}_r3_{target}_top_mentioned_timelines_processed.csv',
        "path_output_format":path_processed_data + '{split}_r3_{target}_top_mentioned_timelines_scored_Texts.csv',
        "path_output_format_L":path_processed_data + '{split}_r3_{target}_top_mentioned_timelines_scored_Texts'+ f'_L={L}_.csv',
        "text_col": "Texts"
    }
}

######################
# Process
######################

dict_final = {}
for dataset_name, config in datasets.items():

    

    dict_dfs = {}
    for target, terms_list in target_terms_dict.items():
        
        print(f"""########################################
# Running dataset:{dataset_name} | target:{target}
########################################""")
        
        dict_splits = {}

        for split in splits:
            
            print(f'# {split}')
        
            path_data = config['path_input_format'].format(split = split, target = target)
            path_output_L = config['path_output_format_L'].format(split = split, target = target)
            path_output_normal = config['path_output_format'].format(split = split, target = target)
            
            # read data
            data = pd.read_csv(
                path_data,
                sep = ';', 
                encoding='utf-8-sig'
            )
            
            text_col = config['text_col']
            
            new_col = f'comments_and_scores_{config['text_col']}'
            
            data[new_col] = data[config['text_col']].progress_apply(lambda x: find_relevant_comments(x, terms_list))
            
            
            data[text_col + '_original'] = data[text_col]             
            data[text_col] = data[f'comments_and_scores_{text_col}'].progress_apply(
                lambda x: " # ".join([comment for score, comment in x[::-1]])
                ) 
            
            
            data.to_csv(
                path_output_normal,
                sep = ';', 
                encoding='utf-8-sig',
                index = False
                )
            
            
            data_L = data.copy()
            data_L[config['text_col'] + f"_L={L}"] = data_L[new_col].progress_apply(lambda x: " # ".join([comment for score, comment in x[-L:]])) 
            
            data_L.to_csv(
                path_output_L, 
                sep = ';', 
                encoding='utf-8-sig',
                index = False
                )
            
            
            
            dict_splits.update({split:data})
            
            
        dict_dfs.update({target:dict_splits})
        
    dict_final.update({dataset_name:dict_dfs})


