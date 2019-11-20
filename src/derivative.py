

class Derivative:
    @staticmethod
    def sigmoid(x):
        return x * (1.0 - x)
    
    @staticmethod
    def reLU(arr):
        arr1 = np.where(arr <= 0, 0, 1)
        return arr1

    @staticmethod
    def softmax(output_errors, target_vector):
        m = target_vector.shape[0]
        output_errors[range(m),target_vector] -= 1
        output_network = output_errors/m
        return output_network