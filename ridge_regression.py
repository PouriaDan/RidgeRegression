import numpy as np

class RidgeRegression():
    """
    least squares Linear Regression with L2 Regularization
    a linear model to minimize the residual sum of squares between 
    the ground truths and predictions by gradient descent

    params:
        num_iters: maximum numbers of iteration for updating model weights according to the gradient descent
        learning_rate: coefficient of gradients used in updating model's weights
        reg_coef: regularization coef, controls the regularization weight in cost function
    """
    def __init__(self, num_iters=None, normalize=False, learning_rate=0.01, reg_coef=0):
        self.reg_coef = reg_coef
        self.normalize = normalize
        self.num_iters = num_iters
        self.learning_rate = learning_rate
        self.COST_DELTA_THRESHOLD = 1e-7
        
        self.params = None
        self.coef_ = None
        self.intercept_ = None
        self.cost = None
        
        self.params_hist = []
        self.cost_hist = []
    
    def fit(self, X, Y):
        Y = Y.reshape(-1,1)
        if X.ndim==1:
            X = X.reshape(-1,1)
        num_enteries = X.shape[0]
        
        if self.normalize:
            self.mean_ = np.mean(X, axis=0, keepdims=True)
            self.std_ = np.std(X, axis=0, keepdims=True)
            X = (X-self.mean_)/self.std_
            
        X_extended = np.append(np.ones((num_enteries,1)), X, axis=1)
        
        self.params = np.ones((X_extended.shape[1], 1))
        self.cost = self._compute_cost(X_extended, Y, self.params)

        self.params_hist.append(self.params)
        self.cost_hist.append(self.cost)
        
        n_iter=0
        while True:
            prev_cost = self.cost
            gradients = self._gradients(X_extended, Y, self.params)

            self.params = self.params - self.learning_rate*gradients
            self.cost = self._compute_cost(X_extended, Y, self.params)

            self.params_hist.append(self.params)
            self.cost_hist.append(self.cost)

            n_iter+=1
            if self.cost>prev_cost:
                raise ValueError("Model is diverging, prevent this behavior by adjusting hyperparameters")
                break
            if self.num_iters is not None and n_iter>=self.num_iters:
                break
            if self.num_iters is None and abs(self.cost-prev_cost) < self.COST_DELTA_THRESHOLD:
                break
                    
        self.coef_ = self.params[1:].ravel()
        self.intercept_ = self.params[0].ravel()[0]
        self.params_hist = np.array(self.params_hist)
        return self
    
    def _compute_cost(self, X_extended, Y, params):
        M = X_extended.shape[0]
        params = params.reshape(-1, 1)
        
        mse = np.square((X_extended @ params) - Y).sum()
        reg = np.power(params[1:] ,2).sum()
        cost = (mse+self.reg_coef*reg)/(2*M)
        return cost
    
    def _gradients(self, X_extended, Y, params):
        M = X_extended.shape[0]
        params = params.reshape(-1, 1)
        params_copy = np.copy(params)
        params_copy[0] = 0
        
        x_sum = X_extended.T @ ((X_extended @ params) -Y)
        gradients = np.add(x_sum, self.reg_coef*params_copy)*(1/M)
        return gradients
    
    def predict(self, X):
        if X.ndim==1:
            X = X.reshape(-1,1)
        if self.normalize:
            X = (X-self.mean_)/self.std_
        M = X.shape[0]
        X_extended = np.append(np.ones((M,1)), X, axis=1)
        return np.matmul(X_extended, self.params)