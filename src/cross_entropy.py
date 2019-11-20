import numpy as np
from softmax import Softmax

class Cross_Entropy:

    @staticmethod
    def calc(output_network, target_vector):
        shape = target_vector.shape[0]
        temp = softmax(output_network)
        log_likelihood = -np.log(temp[range(shape),target_vector])
        loss = np.sum(log_likelihood) / m
        return loss

    @staticmethod
    def derived_calc(output_network,target_vector):
        shape = target_vector.shape[0]
        gradient = softmax(output_network)
        gradient[range(shape),target_vector] -= 1
        gradient = gradient/m
        return gradient
