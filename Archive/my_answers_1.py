# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 10:35:54 2022

@author: Shafufu
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 15:49:43 2022

@author: Shafufu
"""


import numpy as np
import pandas as pd
import os

class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.w1 = np.random.normal(0.0, self.input_nodes**-0.5, (self.input_nodes, self.hidden_nodes))
        self.b1 = np.random.normal(0.0, self.input_nodes**-0.5,self.hidden_nodes)        
        self.w2 = np.random.normal(0.0, self.hidden_nodes**-0.5,(self.hidden_nodes, self.output_nodes))
        self.b2 = np.random.normal(0.0, self.hidden_nodes**-0.5, self.output_nodes)        
        self.lr = learning_rate
        self.activation_function = lambda x : 1 / (1 + np.exp(-x))  # Replace 0 with your sigmoid calculation.
 
    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        '''
        if not isinstance(features,np.ndarray) :
            features=features.to_numpy()
        if type(targets) is not np.ndarray :
            targets = np.array(targets) #targets.to_numpy()
        n_records = features.shape[0]
        dw1 = np.zeros(self.w1.shape)
        db1 = np.zeros(self.b1.shape)
        dw2 = np.zeros(self.w2.shape)
        db2 = np.zeros(self.b2.shape)
        
        for X, y in zip(features, targets):#pass
            
            final_outputs, hidden_outputs = self.forward_pass_train(X)  # Implement the forward pass function below
            # Implement the backproagation function below
            dw1,db1,dw2,db2 = \
            self.backpropagation(final_outputs, hidden_outputs, X, y, dw1,db1,dw2,db2)
        self.update_weights(dw1,db1,dw2,db2, n_records)


    def forward_pass_train(self, X):
        ''' Implement forward pass here 
         
            Arguments
            ---------
            X: features batch

        '''
        #### Implement the forward pass here ####
        ### Forward pass ###
        # TODO: Hidden layer - Replace these values with your calculations.
        hidden_inputs = np.dot(X, self.w1) + self.b1 # signals into hidden layer   # HL X.T==X
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer

        # TODO: Output layer - Replace these values with your calculations.
        final_inputs = hidden_outputs # signals into final output layer
        final_outputs = np.dot(final_inputs, self.w2) + self.b2 # signals from final output layer
        
        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, dw1,db1,dw2,db2):
        ''' Implement backpropagation
         
            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''
        #### Implement the backward pass here ####
        ### Backward pass ###

        # TODO: Output error - Replace this value with your calculations.
        error = -(y - final_outputs)# Output layer error is the difference between desired target and actual output.
        
        # TODO: Calculate the hidden layer's contribution to the error
        hidden_error = error
        
        # TODO: Backpropagated error terms - Replace these values with your calculations.
        output_error_term = error
        
        hidden_error_term = hidden_error*self.w2.T*hidden_outputs*(1-hidden_outputs)
        
        # Weight step (hidden to output)
        dw2 += (hidden_error*hidden_outputs).reshape(dw2.shape)
        db2 += hidden_error.reshape(db2.shape)
        # Weight step (input to hidden)
        dw1 += np.dot(X.reshape(-1,1),hidden_error_term).reshape(dw1.shape)
        db1 += hidden_error_term.reshape(db1.shape)
        return dw1,db1,dw2,db2

    def update_weights(self, dw1,db1,dw2,db2, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        self.w1 -= self.lr * dw1/n_records
        self.b1 -= self.lr * db1/n_records
        self.w2 -= self.lr * dw2/n_records
        self.b2 -= self.lr * db2/n_records

    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        
        #### Implement the forward pass here ####
        # TODO: Hidden layer - replace these values with the appropriate calculations.
        X=features
        hidden_inputs = np.dot(X, self.w1) + self.b1 # signals into hidden layer   # HL X.T==X
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer

        # TODO: Output layer - Replace these values with your calculations.
        final_inputs = hidden_outputs # signals into final output layer
        final_outputs = np.dot(final_inputs, self.w2) + self.b2 # signals from final output layer
        
        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 3600
learning_rate = 0.1
hidden_nodes = 20
output_nodes = 1
