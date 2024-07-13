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

import sys
sys.path.append('src/')
from data.lambdas import int_to_label, label_to_int


import pandas as pd

def create_test_results_df(
    y_test: list, 
    y_pred: list, 
    y_pred_proba_0: list,
    y_pred_proba_1: list
) -> pd.DataFrame:
    """
    Creates a DataFrame from test results.

    This function receives four lists: y_test, y_pred, y_pred_proba_0, and y_pred_proba_1.
    It performs multiple checks to ensure that the DataFrame is correctly formatted:
        1. Verifies that all inputs are lists.
        2. Ensures that y_test and y_pred are lists of strings.
        3. Ensures that y_pred_proba_0 and y_pred_proba_1 are lists of floats.
        4. Checks for any null values in the resulting DataFrame.
    
    Args:
        y_test (list): List of true values (expected to be strings).
        y_pred (list): List of predicted values (expected to be strings).
        y_pred_proba_0 (list): List of predicted probabilities for class 0 (expected to be floats).
        y_pred_proba_1 (list): List of predicted probabilities for class 1 (expected to be floats).
    
    Returns:
        pd.DataFrame: A DataFrame containing the test results.
    
    Raises:
        TypeError: If any of the inputs are not lists or do not contain the expected types.
        Exception: If there are any null values in the resulting DataFrame.
    """
    
    # Check if all function arguments are lists
    if not isinstance(y_test, list):
        raise TypeError(f"The argument y_test is not a list. Found type: {type(y_test).__name__}")
    if not isinstance(y_pred, list):
        raise TypeError(f"The argument y_pred is not a list. Found type: {type(y_pred).__name__}")
    if not isinstance(y_pred_proba_0, list):
        raise TypeError(f"The argument y_pred_proba_0 is not a list. Found type: {type(y_pred_proba_0).__name__}")
    if not isinstance(y_pred_proba_1, list):
        raise TypeError(f"The argument y_pred_proba_1 is not a list. Found type: {type(y_pred_proba_1).__name__}")
    
    # Check if y_test and y_pred are lists of strings
    if not all(isinstance(item, str) for item in y_test):
        raise TypeError("The elements of y_test are not all strings")
    if not all(isinstance(item, str) for item in y_pred):
        raise TypeError("The elements of y_pred are not all strings")
    
    # Check if y_pred_proba_0 and y_pred_proba_1 are lists of floats
    if not all(isinstance(item, float) for item in y_pred_proba_0):
        raise TypeError("The elements of y_pred_proba_0 are not all floats")
    if not all(isinstance(item, float) for item in y_pred_proba_1):
        raise TypeError("The elements of y_pred_proba_1 are not all floats")
    
    # Check if the number of labels is ok
    possible_labels = ['against','for']
    check_labels_test = [False if t in possible_labels else True for t in y_test]
    check_labels_pred = [False if t in possible_labels else True for t in y_pred]
    if sum(check_labels_test) != 0:
        raise Exception(f"The labels are wrong: {set(y_test)}")
    if sum(check_labels_pred) != 0:
        raise Exception(f"The labels are wrong: {set(y_pred)}")
    
    # Create the DataFrame
    df_test_results = pd.DataFrame({
        'test': y_test,
        'pred': y_pred,
        'pred_proba_0': y_pred_proba_0,
        'pred_proba_1': y_pred_proba_1    
    })
    
    # Check for any null values in the DataFrame
    if df_test_results.isna().sum().sum() != 0:
        raise Exception("There are null values in the data")
        

    
    return df_test_results



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

    list_results = process_classification(
        **clf,
        data_tuples=data_tuples_list,
        X_cols=X_col,
    )
    
    # create str_cols (name of coluns for file_paths)
    match = re.search(r'(\w+)_emb', X_col[0])
    if match is None:
        str_cols = "_".join(X_col)
    else: 
        str_cols = match.group(1)
        
    for target, df_test_results in list_results:

        test_results_path = f"{reports_path}test_results/{estimator_name}_{target}_{corpus_name}_{str_cols}_test_results.csv"
        
        print("results in ", test_results_path)
        
        df_test_results.to_csv(test_results_path, index = False)

    return True

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
    
    list_results = []

    for data_train, data_test, target in data_tuples:
        
        gc.collect()
        
        X_train = data_train[X_cols].squeeze()
        y_train = data_train[y_col].apply(lambda x: label_to_int(x))
        
        X_test = data_test[X_cols].squeeze()
        y_test = data_test[y_col].apply(lambda x: label_to_int(x))
        
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

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        
        try:
            y_pred_proba = pipe.predict_proba(X_test).tolist()
        except Exception as e:
            y_pred_proba = None
            
            
        # create df test results
        ## format test and pred
        y_test_formated = [int_to_label(test) for test in y_test]
        y_pred_formated = [int_to_label(pred) for pred in y_pred]
        ## create list of proba of each class
        pred_proba_0 = [float(probas[0]) for probas in y_pred_proba]
        pred_proba_1 = [float(probas[1]) for probas in y_pred_proba]
        ## create df with results
        df_test_results = create_test_results_df(y_test_formated, y_pred_formated, pred_proba_0, pred_proba_1)
            
        list_results.append((target, df_test_results))
        
           
        gc.collect()
        
    return list_results