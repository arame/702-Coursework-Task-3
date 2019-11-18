import numpy as np
class Softmax:

    @staticmethod
    def calc( x):
        sum_exp = np.sum(np.exp(x), axis=0)
        if sum_exp == 0:
            return x 
        smax = np.exp(x)/sum_exp
        return smax
