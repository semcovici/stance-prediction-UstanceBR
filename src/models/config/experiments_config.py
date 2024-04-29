# External
import sys 
from nltk.corpus import stopwords
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC, SVC
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from xgboost import XGBClassifier

estimators = [
        ('svm', LinearSVC(random_state=42, max_iter=10000)),
        ('lr_l1', LogisticRegression(random_state=42, penalty="l1", solver="liblinear"),
        ('rf'), RandomForestClassifier(random_state=42))]
stacking = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(random_state=42, penalty="l2", solver="liblinear"))

clfs = [
        # MultinomialNB(),
        # LogisticRegression(random_state=42, penalty="l1", solver="liblinear"),
        # LogisticRegression(random_state=42, penalty="l2", solver="liblinear"),
        # LinearSVC(random_state=42, max_iter=10000),
        # SVC(random_state=42),
        # DecisionTreeClassifier(random_state=42),
        # RandomForestClassifier(random_state=42),
        # AdaBoostClassifier(random_state=42),
        # GradientBoostingClassifier(random_state=42),
        XGBClassifier(random_state=42),
        DummyClassifier()
#        stacking
        ]

vectorizers = [
        # CountVectorizer(ngram_range=(1,1), analyzer="word"), 
        # CountVectorizer(ngram_range=(1,3), analyzer="word"), 
        TfidfVectorizer(ngram_range=(1,1), analyzer="word", stop_words=stopwords.words('portuguese')), 
        #TfidfVectorizer(ngram_range=(1,2), analyzer="word"), 
#        TfidfVectorizer(ngram_range=(1,3), analyzer="word")
                ]

def get_experiments_config():
        
        dict_exp = {}
        
        for clf in clfs:
            
            name = ''
            name += str(clf.__class__.__name__)
            
            
                    
                    
                
            for vect in vectorizers:
                    
                if clf.__class__.__name__ == 'DummyClassifier':
                        vect = None
                else:
                        name += '_'
                        name += str(vect.__class__.__name__)
                    
                dict_exp.update(
                    {name:{
                        'preprocessing': vect,
                        'estimator': clf
                    }}
                )
                
        return dict_exp
                        
                        