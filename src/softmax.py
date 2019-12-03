import numpy as np
class Softmax:

    @staticmethod
    def calc( x):
        e_x = np.exp(x - np.max(x))
        softmax = e_x / e_x.sum(axis=0)
        return softmax