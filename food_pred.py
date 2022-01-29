from sklearn.svm import SVC
from random import seed
from random import randint

class Predictor:
    def __init__(self,data,use_predetermined=True):
        self.model = SVC()
        self.data = data
        self.use_predetermined = use_predetermined
    
    def init_train(self, X, y):
        pass
    
    def update(self, X, y):
        pass
    
    def get_best_random(self, n=512, threshold=0.8):
        return randint(0, len(self.data.index)-1), 0.8
    
    def get_random(self, num=1):
        return [randint(0, len(self.data.index)-1) for i in range(num)]
    
    def __new_params(self):
        pass
