import copy
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

def propagate(w, b, X_train, Y_train):

    # number of training examples
    m = X_train.shape[1]
    
    # activation function
    A = sigmoid(np.dot(w.T, X_train)+ b)
    print(A)
    # cost function
    cost  = -1/m *np.sum(np.dot(Y_train, np.log(A.T) + np.dot(1-Y_train, np.log(1 - A.T))))
    dw = (1/m)* np.dot(X_train, (A-Y_train).T)
    db = (1/m) * np.sum(A-Y_train)
    
    cost = np.squeeze(np.array(cost))
    grads = {
        "dw": dw,
        "db": db
    }

    return grads, cost

def optimize(w, b, X_train, Y_train, learning_rate =0.009 , num_iterations=100, print_cost=False):
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)

    costs = []

    for i in range(num_iterations):
        grads, cost = propagate(w, b, X_train, Y_train)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate*dw
        b = b - learning_rate*db


        if i % 100 == 0:
            costs.append(cost)
            print(f"Cost after iteration {i}: {cost}.")
    
    params = {"w": w, "b": b}

    grads = {"dw": dw, "db": db}

    return params, grads, costs

def predict(w, b, X_train):
    
    m = X_train.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X_train.shape[0], 1)
    print(w.shape)

    A = sigmoid(np.dot(w.T, X_train) + b)
    for i in range(A.shape[1]):
        if A[0, i] > 0.5:
            Y_prediction[0,i] = 1
        else:
            Y_prediction[0, i] = 0

    return Y_prediction


