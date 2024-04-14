import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import MaxAbsScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from tqdm import tqdm
import nltk
from imblearn.pipeline import Pipeline as IMBPipeline

from sklearn.compose import ColumnTransformer
from datetime import datetime
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectPercentile


def get_classification_report(y_test, y_pred):
    '''Source: https://stackoverflow.com/questions/39662398/scikit-learn-output-metrics-classification-report-into-csv-tab-delimited-format'''
    report = classification_report(y_test, y_pred, output_dict=True)
    df_classification_report = pd.DataFrame(report).transpose()
    df_classification_report = df_classification_report.sort_values(by=['f1-score'], ascending=False)
    return df_classification_report


def process_classification(
        estimator,
        data_tuples, 
        vectorizer = None,
        sampling = None, 
        selection = None,
        scaling = None,
        X_cols = ['Texts'],
        y_col = 'Polarity'
):
    
    df_cr = pd.DataFrame({})
    df_test_results = pd.DataFrame({})

    for data_train, data_test, target in data_tuples:

        X_train = data_train[X_cols]
        y_train = data_train[y_col]
        
        X_test = data_test[X_cols]
        y_test = data_test[y_col]
        
        le = LabelEncoder()
        le_trained = le.fit(y_train)
        
        y_train_enc = le_trained.transform(y_train)
        y_test_enc = le_trained.transform(y_test)

        steps = [
            ('vectorizer', vectorizer),
            ('sampling', sampling),
            ('scaling', scaling),
            ('selection', selection),
            ('estimator',estimator)
        ]

        pipe = IMBPipeline(steps,verbose = True)

        print('Training ...')
        pipe_trained = pipe.fit(X_train, y_train_enc)

        y_pred = pipe_trained.predict(X_test)
        y_pred_proba = pipe_trained.predict_proba(X_test)
        
        df_classification_report = get_classification_report(y_test_enc, y_pred)
        
        df_classification_report = df_classification_report.reset_index().rename(columns = {"index": "class"})
        
        df_classification_report['corpus'] = target 

        df_cr = pd.concat([df_cr, df_classification_report])
        df_test_results = pd.concat([
            df_test_results,
            pd.DataFrame({
                'test':[list(y_test)],
                'pred':[list(y_pred)],
                'pred_proba': [list(y_pred_proba)]    
            })
            ])
        
    return df_cr, df_test_results