import numpy as np
class Activation:

    @staticmethod
    def sigmoid( x):
        return 1 / (1 + np.e ** -x)
    
    @staticmethod
    def reLU(x):
        return np.maximum(0.0, x)

    @staticmethod
    def softmax( x):
        sum_exp = np.sum(np.exp(x), axis=0)
        if sum_exp == 0:
            return x 
        smax = np.exp(x)/sum_exp
        return smax