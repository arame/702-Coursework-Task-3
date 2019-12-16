import numpy as np
class Activation:

    @staticmethod
    def sigmoid( x):
        return 1 / (1 + np.e ** -x)
    
    @staticmethod
    def reLU(x):
        return np.maximum(0.0, x)
    
    @staticmethod
    def hyperbolic(x):
        return np.tanh(x)

    @staticmethod
    def leakyReLU(x):
        leaky_slope = 0.1
        return np.maximum(leaky_slope*x,x)

