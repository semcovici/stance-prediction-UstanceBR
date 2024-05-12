######################
# Imports
######################

import pandas as pd
from tqdm import tqdm
import numpy as np

tqdm.pandas()

######################
# Definitions
######################

L = 30

path_raw_data = 'data/raw/'
path_processed_data = 'data/processed/'

terms_list_ig = [
    "igreja",
    "catedral",
    "capela",
    "templo",
    "paróquia",
    "basílica",
    "padre",
    "pastor",
    "bispo",
    "cardeal",
    "papa",
    "sacerdote",
    "arcebispo",
    "deão",
    "vigário",
    "altar",
    "crucifixo",
    "cálice",
    "hóstia",
    "círio",
    "batistério",
    "sacristia",
    "tabernáculo",
    "missa",
    "culto",
    "batismo",
    "comunhão",
    "confissão",
    "crisma",
    "cerimônia",
    "vaticano",
    "concílio",
    "encíclica",
    "dioceses"
]

terms_list_cl= [
    "hidroxicloroquina",
    "remédio",
    "medicamento",
    "tratamento",
    "antimalárico",
    "antimalárico sintético",
    "droga"
]

terms_list_lu = [
    "lula",
    "presidente",
    "luiz inácio lula da silva",
    "pt",
    "partido dos trabalhadores",
    "ex-presidente",
    "liderança política",
    "governo lula",
    "política"
]

terms_list_co = [
    "sinovac",
    "coronavac",
    "vacina",
    "vacina chinesa",
    "imunização",
    "vacinação",
    "biontech",
    "covid-19",
    "pandemia"
]
terms_list_gl = [
    "globo",
    "tv globo",
    "rede globo",
    "televisão",
    "emissora",
    "rede de televisão",
    "mídia",
    "jornalismo",
    "programação de tv",
    "entretenimento"
]


terms_list_bo = [
    "bolsonaro",
    "jair bolsonaro",
    "presidente",
    "presidente do brasil",
    "governo bolsonaro",
    "partido liberal",
    "política",
    "conservador",
    "ex-presidente"
]

target_terms_dict = {
    'ig': terms_list_ig,
    'bo': terms_list_bo, 
    'cl': terms_list_cl, 
    'co':terms_list_co, 
    'gl': terms_list_gl, 
    'lu': terms_list_lu
}


######################
# Functions
######################

# given comments separated by " # " and a list of terms, 
# return all coments that have at least one of terms in the list_terms
def find_relevant_comments(comments, list_terms, L = None):    
    
    list_comments = comments.split(' # ')
    
    terms_set = set(t.casefold() for t in list_terms)
    
    list_rel_comments = [
        com for com in list_comments if any(term in com.casefold() for term in terms_set)
    ]
    
    if L is not None:
        
        list_rel_comments = create_comment_list(list_comments, list_rel_comments, L)
        
    str_rel_comments = ' # '.join(list_rel_comments) if len(list_rel_comments) > 0 else ''
    
    return str_rel_comments

def create_comment_list(A, B, L):
    """
    Creates a new comment list C of size L. If B contains more than L comments,
    randomly selects L comments from B. If B contains fewer than L comments, it
    randomly selects the remaining comments from A to fill up C.

    Parameters:
        A (list): List of comments from the social network.
        B (list): List of comments from a user on the social network.
        L (int): Size of the new comment list C.

    Returns:
        list: New comment list C of size L.
    """
    C = []

    if len(B) >= L:
        # If B has at least L comments, select L comments randomly without replacement
        C = np.random.choice(B, L, replace=False)
    else:
        # If B has fewer than L comments, add all comments from B
        C.extend(B)
        remaining_comments = L - len(B)
        if remaining_comments <= len(A):
            # If there are enough comments in A to fill up C, randomly select the remaining comments from A
            A_comments = np.random.choice(A, remaining_comments, replace=False)
            C.extend(A_comments)
        else:
            pass

    return C

######################
# Process
######################

splits = ["train", "test"]
datasets = {
    "users": {
        "path_input_format":path_raw_data + 'r3_{target}_{split}_users.csv', 
        "path_output_format":path_processed_data + 'r3_{target}_{split}_users_filtered_Timeline.csv', 
        "text_col": "Timeline"
    },
    "tmt":{
        "path_input_format":path_raw_data + '{split}_r3_{target}_top_mentioned_timelines.csv',
        "path_output_format":path_processed_data + '{split}_r3_{target}_top_mentioned_timelines_filtered_Texts.csv',
        "text_col": "Texts"
    }
}

def main():
    
    for dataset_name, config in datasets.items():

        for target, terms_list in target_terms_dict.items():
            
            print(f"""
########################################
# Running dataset:{dataset_name} | target:{target}
########################################""")

            for split in ["train", "test"]:
                
                print(f'# {split}')
            
                path_data = config['path_input_format'].format(split = split, target = target)
                path_output = config['path_output_format'].format(split = split, target = target)
                
                # read data
                data = pd.read_csv(
                    path_data,
                    sep = ';', 
                    encoding='utf-8-sig'
                )
                
                data[f'filtered_{config['text_col']}'] = data[config['text_col']].progress_apply(lambda x: find_relevant_comments(x, terms_list, L))
                
                data.to_csv(path_output,index=False,sep = ';',encoding='utf-8-sig')

if __name__ == "__main__":
    main()