import pandas as pd
from imblearn.pipeline import Pipeline
import pickle

# save model in a file with picke 
def save_model_in_a_file(model, filename):
    
    pickle.dump(model, open(filename, 'wb'))
    
    return True
    
# train a model and save in a file
def train_model(
    X, y,
    output_model_path = None,
    text_vectorizer = None,
    sampling = None,
    scaling = None, 
    estimator = None
):    
    
    # define pipeline steps
    steps = [
        ('vectorizer', text_vectorizer),
        ('sampling', sampling),
        ('scaling', scaling),
        ('estimator', estimator)
    ]
    # create pipeline 
    pipe = Pipeline(steps, verbose=True)
    # train model
    pipe_trained = pipe.fit(X, y)
    
    # save in a file (if required)
    if output_model_path != None:
        save_model_in_a_file(pipe_trained, output_model_path)
    
    return pipe_trained
    
    
    
    
    
    
    
    
    
    
    