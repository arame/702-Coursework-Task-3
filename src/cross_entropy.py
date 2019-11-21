import numpy as np
from softmax import Softmax

class Cross_Entropy:

    @staticmethod
    def calc(output_network, target_vector):
        shape = target_vector.shape[0]
        smax = Softmax.calc(output_network)
        
        log_likelihood = -np.log(smax[range(shape),target_vector.argmax(axis=1)])
        loss = np.sum(log_likelihood) / shape
        #loss = log_likelihood / shape
        return loss

    @staticmethod
    def derived_calc(output_network,target_vector):
        shape = target_vector.shape[0]
        gradient = Softmax.calc(output_network)
        gradient[range(shape),target_vector.argmax(axis=1)] -= 1
        gradient = gradient/shape
        return gradient
