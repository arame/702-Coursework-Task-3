import numpy as np
class Softmax:

    @staticmethod
    def softmax( x):
        sum_exp = np.sum(np.exp(x), axis=0)
        if sum_exp == 0:
            sump_exp += 0.0001  # Make sure 
        smax = np.exp(x)/sum_exp
        return smax