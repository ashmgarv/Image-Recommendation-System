import numpy as np 
# Formulation of gradient descent algorithm referred from https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47 
class SVM:
    def __init__(self):
       self.w = None
       self.b = None
    
    def fit(self, x ,y):
        # varaible step size for solving the gradient descent.
        learning_rate = [0.01, 0.001, 0.0001]
        # regularization parameter to act as a trade off between misclassification and size 
        # of the margin, higher the value more weightage is given to misclassifcation.
        # Referred from https://datascience.stackexchange.com/questions/4943/intuition-for-the-regularization-parameter-in-svm
        reg_param = 0.01
        # number of epochs across the training data set.
        iter = 1000
        self.b = 0
        num_sample, num_attr = x.shape
        # initializing the decision boundary vector to the size of the vector length.
        self.w = np.zeros(num_attr)
        for learn_rate in learning_rate:
            c = 0
            while c <= iter:
                for i, row in enumerate(x):
                    cost_cnd = y[i] * (np.dot(row, self.w) - self.b)
                    if cost_cnd >= 1:
                        # if signs match, signifies that the the classification was correct.
                        self.w = self.w - (learn_rate * (2 * reg_param * self.w))
                    else:
                        self.w = self.w - (learn_rate * (2 * reg_param * self.w - np.dot(row, y[i])))
                        self.b = self.b - (learn_rate * y[i])
                c += 1
        
    def predict(self, x):
        # solving the equation w*x + b and then taking out the sign to predict classes for the given data.
        return np.sign(np.dot(x, self.w) - self.b)