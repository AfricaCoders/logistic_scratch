from turtle import shape
import pandas as pd
import numpy as np

def train_test_split(data):
    train = data.iloc[:300]
    test = data.iloc[300:]
    
    #Train
    Y_train = train["diagnosis"]
    Y_train = np.array(Y_train)
    Y_train = Y_train.reshape(Y_train.shape[0],-1).T

    X_train = train.loc[:, train.columns != "diagnosis"]
    X_train = np.array(X_train)
    X_train = X_train.reshape(X_train.shape[0], -1).T
    
    #Test
    X_test = test.loc[:, test.columns != "diagnosis"]
    X_test = np.array(X_test)
    X_test = X_test.reshape(X_test.shape[0], -1).T

    Y_test = test["diagnosis"]
    Y_test = np.array(Y_test)
    Y_test = Y_test.reshape(Y_test.shape[0], -1).T
    
    return X_train, Y_train , X_test, Y_test

def sigmoid(z):

    sigmoid = 1 / (1 + np.exp(-z))

    return sigmoid

def initialize_parameters(dims):

    w = np.zeros((dims, 1))
    b = 0

    return w, b

def propagate(W, b, X_train, Y_train):

    # number of training examples
    m = X_train.shape[1]
    
    # activation function
    A = sigmoid(np.dot(W.T, X_train) + b)
    
    # cost function
    cost  = -1/m *np.sum(np.dot(Y_train, np.log(A.T) + np.dot(1-Y_train, np.log(1 - A.T))))

    dw = (1/m)*
    db = 