import numpy as np
from activation import Activation

class Derivative:
    @staticmethod
    def sigmoid(x):
        return x * (1.0 - x)
    
    @staticmethod
    def reLU(arr):
        arr1 = np.where(arr <= 0, 0, 1)
        return arr1

    @staticmethod
    def hyperbolic(x):
        return (1-np.square(x))

    @staticmethod
    def leakyReLU(x):
        leaky_slope = 0.1
        d=np.zeros_like(x)
        d[x<=0]=leaky_slope
        d[x>0]=1
        return d