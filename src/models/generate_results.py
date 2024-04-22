import pandas as pd
from tqdm import tqdm
from sklearn.dummy import DummyClassifier
from joblib import Parallel, delayed
from joblib_progress import joblib_progress
from datetime import datetime
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler
import numpy as np
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline as IMBPipeline
from sklearn.feature_selection import RFE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords
from tqdm import tqdm
tqdm.pandas()
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFE
from tsfresh.feature_extraction import MinimalFCParameters
from tsfresh.feature_extraction import extract_features
tqdm.pandas()
import warnings
warnings.filterwarnings('ignore') 


# Aviso para ignorar alertas desnecessários
warnings.filterwarnings("ignore")

import sys
sys.path.append('src/')
from models.classification_methods import create_train_test_tuples, generate_results 

model_name = 'neuralmind/bert-base-portuguese-cased'

random_seed = 42

raw_data_path = 'data/raw/'
processed_data_path = 'data/processed/'
results_cr_path = 'reports/classification_reports/'
test_results_path = 'reports/test_results/'
reports_path = 'reports/'

target_list = ['ig','bo', 'cl', 'co', 'gl', 'lu']

top_ment_time_path = raw_data_path + '{}_r3_{}_top_mentioned_timelines.csv'
list_train_paths_tmt = [top_ment_time_path.format("train",t) for t in target_list]
list_test_paths_tmt = [top_ment_time_path.format("test",t) for t in target_list]

users_path = raw_data_path + 'r3_{}_{}_users.csv'
list_train_paths_users = [users_path.format(t,"train") for t in target_list]
list_test_paths_users = [users_path.format(t,"test") for t in target_list]

model_name = 'neuralmind/bert-base-portuguese-cased'

top_ment_time_emb_path = processed_data_path + '{}_r3_{}_top_mentioned_timelines_{}.parquet'
list_train_paths_tmt_emb = [top_ment_time_emb_path.format("train",t, model_name.replace("/", "_")) for t in target_list]
list_test_paths_tmt_emb = [top_ment_time_emb_path.format("test",t, model_name.replace("/", "_")) for t in target_list]


users_emb_path = processed_data_path + 'r3_{}_{}_users_{}.parquet'
list_train_paths_users_emb = [users_emb_path.format(t,"train", model_name.replace("/", "_")) for t in target_list]
list_test_paths_users_emb = [users_emb_path.format(t,"test", model_name.replace("/", "_")) for t in target_list]



clf_to_test = {
    'dummy': {
        'estimator': DummyClassifier()
    },
    'tfidf_xgb':{
        'preprocessing': TfidfVectorizer(
                    stop_words = stopwords.words('portuguese'),
                    lowercase = True,
                    # ngram_range = (1,3),
                    # max_features=50
                    
                    ),
        'scaling': MaxAbsScaler(),
        'estimator':  XGBClassifier(
                random_state = 42,
                #verbosity = 3,
                # device = 'cuda',
                # tree_method = 'hist'
                )
    }
}

clf_to_test_emb = {
    'bertimbau_xgb':{
        'scaling': MaxAbsScaler(),
        'estimator':  XGBClassifier(
                random_state = 42,
                #verbosity = 3,
                # device = 'cuda',
                # tree_method = 'hist'
                )
    }
}


# X_cols_comb: possible combinations of X_col
config_experiments_dict = {
    'top_mentioned_timelines':{
        'list_train_paths': list_train_paths_tmt,
        'list_test_paths' : list_test_paths_tmt,
        'file_type': 'csv',
        'read_data_args' : {'sep': ';', 'encoding': 'utf-8-sig'},
        'X_cols_comb': [['Texts']],
        'clf_to_test': clf_to_test
    },
    'users':{
        'list_train_paths': list_train_paths_users,
        'list_test_paths' : list_test_paths_users,
        'file_type': 'csv',
        'read_data_args' : {'sep': ';', 'encoding': 'utf-8-sig'},
        'X_cols_comb': [['Timeline'], ['Stance']],
        'clf_to_test': clf_to_test
    },
    'users_emb':{
        'list_train_paths': list_train_paths_users_emb,
        'list_test_paths' : list_test_paths_users_emb,
        'file_type': 'parquet',
        'read_data_args' : {},
        'X_cols_comb': [
            [f'Timeline_emb_{i + 1}' for i in range(768)], 
            [f'Stance_emb_{i + 1}' for i in range(768)]
            ],
        'clf_to_test': clf_to_test_emb
    },
    'top_mentioned_timelines_emb':{
        'list_train_paths': list_train_paths_tmt_emb,
        'list_test_paths' : list_test_paths_tmt_emb,
        'file_type': 'parquet',
        'read_data_args' : {},
        'X_cols_comb': [
            [f'Texts_emb_{i + 1}' for i in range(768)]
            ],
        'clf_to_test': clf_to_test_emb
    }
}

# Execução dos experimentos
for corpus, config in config_experiments_dict.items():
    
    print(f'##### Start of {corpus} #####')
    
    
    data_tuples_list = create_train_test_tuples(
        list_train_paths=config["list_train_paths"],
        list_test_paths=config["list_test_paths"],
        target_list=target_list,
        file_type=config["file_type"],
        read_data_args=config["read_data_args"],
    )

    for X_col in config["X_cols_comb"]:
        
        print(f'- Running classifier with features {X_col[0]} - {datetime.today()}')
        
        for clf_name, clf in config["clf_to_test"].items():
            generate_results(
                data_tuples_list,
                corpus,
                X_col,
                clf,
                reports_path,
                estimator_name=clf_name,
            )
            
    print(f'##### End of {corpus} #####\n\n\n')
