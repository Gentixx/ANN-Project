from sklearn.svm import SVC
class Predictor:
    def __init__(self,data,use_predetermined=True):
        self.model=SVC()
        self.data=data
        self.use_predetermined=use_predetermined
    def init_train(self, X,y):
        pass
    def update(self,X,y):
        pass
    def get_best_random(self,n,threshold):
        pass
    def get_random(self,num):
        pass 
    def __new_params(self):
        pass
#df.iloc[0]["additives"]