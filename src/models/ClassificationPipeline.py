from imblearn.pipeline import Pipeline

class ClassificationPipeline:
    
    
    def __init__(
        self,
        estimator,
        vectorizer = None,
        sampling = None, 
        selection = None,
        scaling = None
        ) -> None:
        
        steps = [
            ('vectorizer', vectorizer),
            ('sampling', sampling),
            ('scaling', scaling),
            ('selection', selection),
            ('estimator',estimator)
        ]
        
        self.pipe = Pipeline(steps=steps, verbose=True)
        
        self.pipe_trained = None
        
    def train(self, X, y):
        
        self.pipe_trained = self.pipe.fit(X,y)
        
        return self.pipe_trained
    
    def predict(self, X):
        
        if self.pipe_trained == None:
            raise Exception('There is no trained pipeline')
        else:
            
            y_pred = self.pipe_trained.predict(X)
            y_pred_proba = self.pipe_trained.predict_proba(X)
            
            return y_pred, y_pred_proba
        
    
        
        