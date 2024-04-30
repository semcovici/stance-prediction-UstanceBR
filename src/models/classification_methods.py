import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import MaxAbsScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from tqdm import tqdm
import nltk
from imblearn.pipeline import Pipeline as IMBPipeline
import gc
import re
from sklearn.compose import ColumnTransformer
from datetime import datetime
from sklearn.dummy import DummyClassifier
from joblib_progress import joblib_progress
from joblib import Parallel, delayed
from sklearn.feature_selection import SelectPercentile


def get_classification_report(y_test, y_pred, cr_args = {}):
    '''Source: https://stackoverflow.com/questions/39662398/scikit-learn-output-metrics-classification-report-into-csv-tab-delimited-format'''
    report = classification_report(y_test, y_pred, output_dict=True, **cr_args)
    df_classification_report = pd.DataFrame(report).transpose()
    df_classification_report = df_classification_report.sort_values(by=['f1-score'], ascending=False)
    return df_classification_report

# def get_classification_report(y_test, y_pred, binary = True):
    
#     metrics = {}
#     if binary:
        
#         metrics["f1-score"] = f1_score(y_test, y_pred)
#         metrics["recall"] = precision_score(y_test, y_pred)
#         metrics["precision"] = recall_score(y_test, y_pred)
#         metrics["accuracy"] = accuracy_score(y_test, y_pred)
        
#     return metrics



# Leitura de dados
def read_pandas(path, file_type="csv", read_data_args={}):
    if file_type == "csv":
        return pd.read_csv(path, **read_data_args)
    elif file_type == "parquet":
        return pd.read_parquet(path, **read_data_args)

# Criar tuplas de treinamento e teste
def create_train_test_tuples(list_train_paths, list_test_paths, target_list, n_jobs=-1, file_type="csv", read_data_args={}):
    if len(list_train_paths) != len(list_test_paths) or len(list_train_paths) != len(target_list):
        raise ValueError("As listas não têm o mesmo comprimento")

    data_paths = zip(list_train_paths, list_test_paths, target_list)
    func_read_data = lambda a, b, c: (
        read_pandas(a, file_type, read_data_args),
        read_pandas(b, file_type, read_data_args),
        c,
    )

    with joblib_progress("Lendo dados...", total=len(list_train_paths)):
        parallel = Parallel(n_jobs=n_jobs)
        return parallel(delayed(func_read_data)(a, b, c) for a, b, c in data_paths)

# Geração de resultados
def generate_results(data_tuples_list, corpus_name, X_col, clf, reports_path, estimator_name=None):
    if estimator_name is None:
        estimator_name = clf["estimator"].__class__.__name__

    df_cr, df_test_results = process_classification(
        **clf,
        data_tuples=data_tuples_list,
        X_cols=X_col,
    )
    
    match = re.search(r'(\w+)_emb', X_col[0])
    
    if match is None:

        str_cols = "_".join(X_col)
        df_cr['preprocessing'] = clf['preprocessing']
        
    else: 
        
        str_cols = match.group(1)
        df_cr['preprocessing'] ='emb'

    cr_path = f"{reports_path}classification_reports/{estimator_name}_{corpus_name}_{str_cols}_classification_report.csv"
    test_results_path = f"{reports_path}test_results/{estimator_name}_{corpus_name}_{str_cols}_test_results.csv"

    df_cr['estimator'] = clf['estimator']

    df_cr.to_csv(cr_path)
    df_test_results.to_csv(test_results_path)

    return df_cr, df_test_results

def process_classification(
        estimator,
        data_tuples, 
        preprocessing = None,
        sampling = None, 
        selection = None,
        scaling = None,
        X_cols = ['Texts'],
        y_col = 'Polarity'
):
    
    df_cr = pd.DataFrame({})
    df_test_results = pd.DataFrame({})

    for data_train, data_test, target in data_tuples:
        
        X_train = data_train[X_cols].squeeze()
        y_train = data_train[y_col]
        
        X_test = data_test[X_cols].squeeze()
        y_test = data_test[y_col]
        
        le = LabelEncoder()
        le_trained = le.fit(y_train)
        
        y_train_enc = le_trained.transform(y_train)
        y_test_enc = le_trained.transform(y_test)
        
        
        if estimator.__class__.__name__ == 'DummyClassifier':
            preprocessing = None
            sampling = None
            selection = None
            scaling = None

        steps = [
            ('preprocessing', preprocessing),
            ('sampling', sampling),
            ('scaling', scaling),
            ('selection', selection),
            ('estimator',estimator)
        ]
        
        pipe = IMBPipeline(steps,verbose = False)

        pipe.fit(X_train, y_train_enc)
        y_pred = pipe.predict(X_test)
        
        try:
            y_pred_proba = pipe.predict_proba(X_test).tolist()
        except Exception as e:
            y_pred_proba = None
        
        
        df_classification_report = get_classification_report(y_test_enc, y_pred)
        
        df_classification_report = df_classification_report.reset_index().rename(columns = {"index": "class"})
        
        df_classification_report['corpus'] = target 

        df_cr = pd.concat([df_cr, df_classification_report])
        df_test_results = pd.concat([
            df_test_results,
            pd.DataFrame({
                'test':[list(y_test)],
                'pred':[list(y_pred)],
                'pred_proba': [y_pred_proba]    
            })
            ])
        
        del X_train, y_train, X_test, y_test, y_train_enc, y_test_enc        
        gc.collect()
        
    return df_cr, df_test_results