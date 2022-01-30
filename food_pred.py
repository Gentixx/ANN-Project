from sklearn.svm import SVC
from random import randint
import numpy as np
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV
#from sklearn.datasets import load_iris

class Predictor:
    def __init__(self,data):
        self.model = SVC()
        self.data = data
        
    
    def init_train(self, X, y,use_predetermined=False):
        self.use_predetermined = use_predetermined
        if use_predetermined==True:
            self.model.fit(X,y)
        else:
            svc_tuned_parameters = [
                {"kernel": ["rbf"], "gamma": np.linspace(0.01, 10,num=100), "C": np.linspace(1, 100,num=100), "probability": [True]},
                {"kernel": ["linear"], "C": np.linspace(1, 100,num=100),"probability": [True]},
            ]
            temp_model = HalvingRandomSearchCV(self.model, svc_tuned_parameters,cv=2).fit(X, y)
            # print(temp_model.best_params_)
            # print(temp_model.best_score_)
            self.model=SVC(**temp_model.best_params_).fit(X, y)
    
    def update(self, X, y):
        self.init_train(X,y,use_predetermined=True)
    
    def get_best_random(self, n=512, threshold=0.6,max_iterations=5):
        for i in range(max_iterations):
            tem=[randint(0, len(self.data.index)-1) for i in range(n)]
            M=self.data["additives"].iloc[tem].to_numpy()
            X1=[]
            for i in M:
                X1.append(i)
            X=np.array(X1)
            probabilties=self.model.predict_proba(X)[:,np.where(self.model.classes_==1)[0][0]]
            max_value=max(probabilties)
            max_index=np.where(probabilties==max_value)[0][0]
            if max_value>threshold:
                break
        
        return max_index, max_value
    
    def get_random(self, num=1):
        return [randint(0, len(self.data.index)-1) for i in range(num)]
    
# X, y = load_iris(return_X_y=True)
# newpred=Predictor(X)
# newpred.init_train(X, y)