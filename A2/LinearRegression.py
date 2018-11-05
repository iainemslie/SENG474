#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

def main():
    #Extend the linear regression algorithm for Python on the slides to :
    #1. Read the data from a file(regdata.csv)
    #2. Scale the attributes
    #3. Compute the error at each iteration and save the error values in a list
    #4. Plot the error list as a curve in the end
    #5. Find a good learning rate based on the error curve

    #Hints
    #1. y in the slides of linear regression is a row matrix. So, when extracting it
    # from the dataset be careful to turn it into a row matrix
    #2. Do not forget to add the "dummy" attribute (all ones) to X
    #4 For this dataset the kappa should be quite small in order to have a decresing E
    data = pd.read_csv('regdata.csv', header=None, names = ['GPA', 'Years of Experience', 'Salary'])
    data.head()

    result = prepare(data)
    X = result[0]
    y = result[1]

    w = np.ones((1,X.shape[1]))

    w,E = fit(X,y,0.1,100)

    print(w)
    plt.plot(E)
    plt.show()

def error(x,y,w):
    return (y-w@x.T)**2

def error_mean(X,y,w):
    return (1/(2*len(X[0])))*np.sum(error(X,y,w))

def grad(x,y,w):
    return (y-w@x.T).T*x

def grad_mean(X,y,w):
    return np.sum(grad(X,y,w), axis=0, keepdims=True)/len(X[0])

def fit(X,y,kappa,iter):
    w = np.zeros((1,X.shape[1]))
    E = []
    for i in range(iter):
        E.append(error_mean(X,y,w))
        w = w + kappa*(grad_mean(X,y,w))
    return w,E

def prepare(data):
    X = data.iloc[:,0:2].values
    y = data.iloc[:,-1].values
    avgX = np.average(X, axis=0)
    maxX = np.max(X, axis=0)
    minX = np.min(X, axis=0)
    avgY = np.average(y, axis=0)
    maxY = np.max(y, axis=0)
    minY = np.min(y, axis=0)
    X = (X-avgX)/(maxX-minX)
    y = (y-avgY)/(maxY-minY)
    X = np.insert(X, 0, 1, axis=1)
    result = []
    result.append(X)
    result.append(y)
    return result

if __name__ == "__main__":
    main()
