import numpy as np
class Softmax:

    @staticmethod
    def calc( x):
        sum_exp = np.sum(np.exp(x), axis=0)
        if sum_exp == 0:
            sum_exp = 0.0001  # Make sure division not by zero
        smax = np.exp(x)/sum_exp
        return smax     # returns a vector of the same shape as x