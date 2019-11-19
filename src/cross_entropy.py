import numpy as np
from activation import Activation

class Cross_Entropy:

    @staticmethod
    def calc(output_network, target_vector):
        m = target_vector.shape[0]
        log_likelihood = -np.log(output_network[range(m),target_vector])
        loss = np.sum(log_likelihood) / m
        return loss