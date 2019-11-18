import numpy as np
class Softmax:

    @staticmethod
    def calc( x, i):
        return np.exp(x[i])/sum(np.exp(x))