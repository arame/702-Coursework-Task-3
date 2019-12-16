import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from activation import Activation
from derivative import Derivative
from cross_entropy import Cross_Entropy
from scipy.stats import truncnorm

class NeuralNetwork:
    # intialization method (constructor)
    def __init__(self, 
                 no_of_in_nodes, 
                 no_of_out_nodes, 
                 no_of_hidden_nodes,
                 learning_rate):
        self.no_of_in_nodes = no_of_in_nodes
        self.no_of_out_nodes = no_of_out_nodes
        self.no_of_hidden_nodes = no_of_hidden_nodes
        self.learning_rate = learning_rate 
        self.create_weight_matrices()

# Code for training  

    def truncated_normal(self, mean=0, sd=1, low=0, upp=10):
        return truncnorm((low - mean) / sd, 
                     (upp - mean) / sd, 
                     loc=mean, 
                     scale=sd)

    def create_weight_matrices(self):
        """ 
        A method to initialize the weight 
        matrices of the neural network
        """
        rad = 1 / np.sqrt(self.no_of_in_nodes)
        X = self.truncated_normal(mean=0, 
                             sd=1, 
                             low=-rad, 
                             upp=rad)
        self.weights_in_hidden = X.rvs((self.no_of_hidden_nodes, self.no_of_in_nodes))
        rad = 1 / np.sqrt(self.no_of_hidden_nodes)
        X = self.truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_hidden_output = X.rvs((self.no_of_out_nodes, self.no_of_hidden_nodes))
        
    def train_single(self, input_vector, target_vector):
        """"
            Forward Propagation
            input:  input_vector  [784], target_vector [10]
            H1:     output_vector [100], weights_in_hidden[100, 784], output_hidden[100]
            output: output_vector2 [10], weight_hidden_output[10, 100], output_network[10]
            loss scalar 2.3

            Backward Propagation
            gradient [10], derived [10], tmp2 [10], hidden_errors [100,10], tmp5 [100,10]
            
        """
        input_vector = np.array(input_vector, ndmin=2).T
        target_vector = np.array(target_vector, ndmin=2).T
        
        output_vector1 = np.dot(self.weights_in_hidden, input_vector)
        output_hidden = Activation.leakyReLU(output_vector1)
        
        output_vector2 = np.dot(self.weights_hidden_output, output_hidden)
        output_network = Activation.leakyReLU(output_vector2)

        loss = Cross_Entropy.calc(output_network, target_vector)
        gradient = Cross_Entropy.derived_calc(output_network, target_vector)
        # update the weights:
        derived1 = Derivative.leakyReLU(gradient)
        tmp2 = derived1 * (loss * gradient)
        # calculate hidden errors:
        hidden_errors = np.dot(self.weights_hidden_output.T, loss * derived1)
        # update the weights:
        tmp5 = hidden_errors * Derivative.leakyReLU(output_hidden)
        self.weights_hidden_output += self.learning_rate  * np.dot(tmp2, output_hidden.T)
        self.weights_in_hidden += self.learning_rate * np.dot(tmp5, input_vector.T)
        
    def train(self, data_array, 
              labels_one_hot_array,
              epochs=1,
              intermediate_results=False):
        intermediate_weights = []
        for _ in range(epochs):  
            print("*", end="")
            for i in range(len(data_array)):
                self.train_single(data_array[i], labels_one_hot_array[i])
            if intermediate_results:
                intermediate_weights.append((self.weights_in_hidden.copy(), 
                                             self.weights_hidden_output.copy()))
        return intermediate_weights        

 # Code for testing       
    
    def run(self, input_vector):
        # input_vector can be tuple, list or ndarray
        input_vector = np.array(input_vector, ndmin=2).T
        # 1st layer
        output_vector = np.dot(self.weights_in_hidden, input_vector)
        output_vector = Activation.leakyReLU(output_vector)
        # 2nd layer
        output_vector = np.dot(self.weights_hidden_output, output_vector)
        output_vector = Activation.leakyReLU(output_vector)
    
        return output_vector
            
    def confusion_matrix(self, data_array, labels):
        cm = np.zeros((10, 10), int)
        for i in range(len(data_array)):
            res = self.run(data_array[i])
            res_max = res.argmax()
            target = labels[i][0]
            cm[res_max, int(target)] += 1
        return cm    
    def precision(self, label, confusion_matrix):
        col = confusion_matrix[:, label]
        calc_value = confusion_matrix[label, label] / col.sum()
        return calc_value
    
    def recall(self, label, confusion_matrix):
        row = confusion_matrix[label, :]
        calc_value = confusion_matrix[label, label] / row.sum()
        return calc_value
    
    def evaluate(self, data, labels):
        corrects, wrongs = 0, 0
        for i in range(len(data)):
            res = self.run(data[i])
            res_max = res.argmax()
            if res_max == labels[i]:
                corrects += 1
            else:
                wrongs += 1
        return corrects, wrongs