import pandas as pd

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, make_scorer
from scipy.stats import uniform, loguniform
from tqdm import tqdm
import sys
sys.path.append('src/')
from models.classification_methods import create_test_results_df    

# Constants for paths
PROCESSED_DATA_PATH = 'data/processed/'
TEST_RESULTS_PATH = 'reports/test_results/'
TRAIN_RESULTS_PATH = 'reports/train_results/'

PATH_BEST_UFT = "XGBClassifier_TfidfVectorizer_{target}_top_mentioned_timelines_Texts_{split}_results.csv"
PATH_BEST_UT = "XGBClassifier_TfidfVectorizer_{target}_users_Timeline_{split}_results.csv"
PATH_BEST_S = "bert_classifier_pablocosta_bertabaporu_base_uncased_{target}_Stance_{split}_results.csv"
PATH_USERS = PROCESSED_DATA_PATH + "r3_{target}_{split}_users_processed.csv"
PATH_TMT = PROCESSED_DATA_PATH + "{split}_r3_{target}_top_mentioned_timelines_processed.csv"

TARGET_LIST = ['ig', 'bo', 'cl', 'co', 'gl', 'lu']

def fill_missing_indices(df):
    # Encontre o índice completo esperado
    full_index = pd.RangeIndex(start=df.index.min(), stop=df.index.max() + 1)

    # Identifique os índices faltantes
    missing_index = full_index.difference(df.index)

    # Crie um DataFrame com os índices faltantes e valores NaN
    missing_df = pd.DataFrame(index=missing_index, columns=df.columns)

    # Combine os DataFrames original e faltante
    combined_df = pd.concat([df, missing_df])

    # Ordene o DataFrame pelo índice
    combined_df = combined_df.sort_index()
    
    combined_df.index= combined_df.index.astype('int')

    return combined_df

def load_and_prepare_data(target, split, path_template, path_index):
    """
    Load data, set appropriate column names, and index from a CSV file.
    """
    df = pd.read_csv(path_template.format(split=split, target=target))
    df.columns = [col + f'_{split[:1].upper()}' for col in df.columns]
    df.index = pd.read_csv(path_index.format(split=split, target=target), sep=';', encoding='utf-8-sig', index_col=0).index
    return df



target_list = [
    'ig',
    'bo', 
    'cl', 
    'co', 
    'gl', 
    'lu'
    ]


def read_data(
    results_df_path,
    text_col, 
    original_df_path
    ):
    
    df = pd.read_csv(results_df_path)
    df.columns = [col + f'_{text_col}' for col in df.columns]
    df.index  = pd.read_csv(
        original_df_path,
        sep = ';', 
        encoding='utf-8-sig',
        index_col = 0
    ).index
    
    return df

def preprocess_data(target):
    """
    Preprocess data for training and testing, fill missing indices, and combine dataframes.
    """
    
    train_UFT = read_data(
        results_df_path = TRAIN_RESULTS_PATH + PATH_BEST_UFT.format(split='train', target = target),
        text_col = "UFT", 
        original_df_path = PATH_TMT.format(split="train", target=target)
        )
    train_UT = read_data(
        results_df_path = TRAIN_RESULTS_PATH + PATH_BEST_UT.format(split='train', target = target),
        text_col = "UT", 
        original_df_path = PATH_USERS.format(split="train", target=target)
        )
    train_S = read_data(
        results_df_path = TRAIN_RESULTS_PATH + PATH_BEST_S.format(split='train', target = target),
        text_col = "S", 
        original_df_path = PATH_USERS.format(split="train", target=target)
        )
    test_UFT = read_data(
        results_df_path = TEST_RESULTS_PATH + PATH_BEST_UFT.format(split='test', target = target),
        text_col = "UFT", 
        original_df_path = PATH_TMT.format(split="test", target=target)
        )
    test_UT = read_data(
        results_df_path = TEST_RESULTS_PATH + PATH_BEST_UT.format(split='test', target = target),
        text_col = "UT", 
        original_df_path = PATH_USERS.format(split="test", target=target)
        )
    test_S = read_data(
        results_df_path = TEST_RESULTS_PATH + PATH_BEST_S.format(split='test', target = target),
        text_col = "S", 
        original_df_path = PATH_USERS.format(split="test", target=target)
        )
    
    # fill null values with -1
    filled_train_UFT = fill_missing_indices(train_UFT).fillna(-1)
    filled_train_UT = fill_missing_indices(train_UT).fillna(-1)
    filled_train_S = fill_missing_indices(train_S).fillna(-1)
    
    train = pd.concat([filled_train_UFT, filled_train_UT, filled_train_S], axis = 1)
    test = pd.concat([test_UFT, test_UT, test_S], axis = 1)

    if train.isna().sum().sum() > 0:
        raise TypeError("Null data in train")
    if test.isna().sum().sum() > 0:
        raise TypeError("Null data in test")


    if len(train[~ (train.test_UFT == train.test_UT) & ((train.test_UT !=-1) & (train.test_UFT !=-1))]) > 0: 
        raise ValueError("há valores inconsistentes para a label")

    if len(test[~ (test.test_UFT == test.test_UT) & ((test.test_UT !=-1) & (test.test_UFT !=-1))]) > 0: 
        raise ValueError("há valores inconsistentes para a label")

    # sabemos que as labels de UFT, UT e S são iguais tirando os casos onde é -1
    train_label_UFT = train.test_UFT.tolist()
    train_label_UT = train.test_UT.tolist()
    test_label_UFT = test.test_UFT.tolist()
    test_label_UT = test.test_UT.tolist()

    y_train = [train_label_UFT[i] if train_label_UFT[i] != -1 else train_label_UT[i]  for i in range(len(train_label_UFT))]
    y_test = [test_label_UFT[i] if test_label_UFT[i] != -1 else test_label_UT[i]  for i in range(len(test_label_UFT))]


    cols_to_drop = ['test_UFT',"pred_UFT",'test_UT', 'pred_UT', "pred_proba_0_UFT", "pred_proba_0_UT", "pred_proba_0_S", 'pred_S', 'test_S']
    
    X_train_full = train.drop(cols_to_drop,axis = 1)
    X_test_full = test.drop(cols_to_drop,axis = 1)
    
    return X_train_full, X_test_full, y_train, y_test

def perform_model_search(X_train, y_train, X_test, y_test, comb, results_path, target):
    """
    Perform RandomizedSearchCV on LogisticRegression and save test results.
    """
    param_dist = {
        'C': loguniform(1e-6, 1e6),
        'penalty': ['l1', 'l2', 'elasticnet', 'none'],
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        'max_iter': [100, 200, 300, 500, 1000, 2000],
        'l1_ratio': uniform(0, 1),
        'tol': loguniform(1e-5, 1e-1),
        'fit_intercept': [True, False],
        'class_weight': [None, 'balanced'],
        'intercept_scaling': uniform(0.1, 2)
    }

    model = LogisticRegression(random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=500, scoring="f1_macro", cv=cv, random_state=42, n_jobs=-1, verbose=True)

    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_

    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)

    y_test_formated = [test for test in y_test]
    y_pred_formated = [pred for pred in y_pred]
    pred_proba_0 = [float(probas[0]) for probas in y_pred_proba]
    pred_proba_1 = [float(probas[1]) for probas in y_pred_proba]

    df_test_results = create_test_results_df(y_test_formated, y_pred_formated, pred_proba_0, pred_proba_1)
    df_test_results.to_csv(results_path, index=False)


for target in tqdm(target_list):
    
    X_train_full, X_test_full, y_train, y_test = preprocess_data(target)


    for comb in [
        ('Texts', 'Timeline'),
        ('Texts', 'Stance'),
        ('Stance', 'Timeline'),
        ('Stance', 'Timeline', 'Texts')
    ]:

        str_cols = "_".join(comb)

        X_train = X_train_full.copy()
        X_test = X_test_full.copy()

        if "Texts" not in comb:
            X_train.drop([col for col in X_train.columns if "UFT" in col], axis=1, inplace=True)
            X_test.drop([col for col in X_test.columns if "UFT" in col], axis=1, inplace=True)
        if "Timeline" not in comb:
            X_train.drop([col for col in X_train.columns if "UT" in col], axis=1, inplace=True)
            X_test.drop([col for col in X_test.columns if "UT" in col], axis=1, inplace=True)
        if "Stance" not in comb:
            X_train.drop([col for col in X_train.columns if "S" in col], axis=1, inplace=True)
            X_test.drop([col for col in X_test.columns if "S" in col], axis=1, inplace=True)

        results_path = f"{TEST_RESULTS_PATH}/Ensemble_LogisticRegression_{target}_{str_cols}_test_results.csv"
        perform_model_search(X_train, y_train, X_test, y_test, comb, results_path, target)
        print("Results saved at:", results_path)