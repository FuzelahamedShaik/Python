import numpy as np

class metrics:
    
    def mse(self,y,y_pred):
        return np.mean((y - y_pred) ** 2)
   
    def sst(self,y):
         y_avg = np.mean(y)
         return np.mean((y - y_avg) ** 2)
    
    def r_squared(self,y,y_pred):
        mse = np.mean((y - y_pred) ** 2)
        y_avg = np.mean(y)
        sst = np.mean((y - y_avg) ** 2)
        return (1-(mse/sst))
    
    def accuracy_score(self,y,y_pred):
        return (np.sum(y == y_pred) / len(y))

class LinearRegression:
    
    def __init__(self,lr=0.001,n_iters=1000):
        self.weights = None
        self.bias = None
        self.lr = lr
        self.n_iters= n_iters
    
    def fit(self,X,y):
        n_samples,n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.n_iters):
            y_pred = np.dot(X,self.weights) + self.bias
            dw = (1/n_samples) * np.dot(X.T,(y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr*db
    
    def predict(self,X):
         y_pred = np.dot(X,self.weights) + self.bias
         return y_pred
    
    
    
class LogisticRegression:
    
    def __init__(self,lr=0.001,n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        
    def sigmoid(self,x):
        sig =  1/(1 + np.exp(-x))
        return sig
    
    def fit(self,X,y):
        n_samples,n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iters):
            linear_term = np.dot(X,self.weights) + self.bias
            y_pred = self.sigmoid(linear_term)
            dw = (1/n_samples) * np.dot(X.T,(y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self,X):
        linear_term = np.dot(X,self.weights) + self.bias
        y_pred = self.sigmoid(linear_term)
        y_pred_cls = [1 if i>0.5 else 0 for i in y_pred]
        return y_pred_cls
    
class SupportVectorClassification:
    
    def __init__(self,lr=0.001,n_iters=1000,lambda_par=0.01):
        self.lr = lr
        self.n_iters = n_iters
        self.lambda_par = lambda_par
        self.weights = None
        self.bias = None
    
    def fit(self,X,y):
        n_samples,n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        y_ = np.where(y<=0,-1,1)
        
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.weights) - self.bias) >= 1
                if condition:
                    self.weights -= self.lr * (2*self.lambda_par*self.weights)
                else:
                    self.weights -= self.lr * ((2*self.lambda_par*self.weights) - np.dot(x_i,y_[idx]))
                    self.bias -= self.lr * y_[idx]
    
    def predcit(self,X):
        y_out = np.dot(X,self.weights) - self.bias
        return np.sign(y_out)
  


class KNearestNeighbors:
    
    def euclidean_distance(self,x1,x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def __init__(self,k=5):
        self.k = k
        
    def fit(self,X,y):
        self.X_train = X
        self.y_train = y
        
    def predict(self,X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)
    
    def _predict(self,x):
        distance = [self.euclidean_distance(x,x_train) for x_train in self.X_train]
        k_indices = np.argsort(distance)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    