# Created on October 26, 2019 by Michell Li
# Implementation of a Neural Network
# One hidden layer
# with Stochastic Gradient Descent
# Loss Function: Mean Cross Entropy

import numpy as np
import scipy as sp
import sys

# read in data
# function takes in a csv file and returns the parsed x and y vector
def read_data(file):

    data = np.loadtxt(file,delimiter=',',skiprows=0)   #split rows to have columns

    y = data[:,0].astype(int) # assign label set

    x = data[:,1:] #create features array

    # fold in bias term
    bias = [[1] for x in range(x.shape[0])]
    x = np.append(bias,x,axis=1)

    return x, y


def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def cross_entropy(probs, y):
    l = probs[y]
    return -np.log(l)

def initalize_hyperparameters(init_flag, hidden_units, X_train):

    if init_flag == '1':
        #random uniform dist over [-0.1,0.1]
        alpha = np.random.random_sample((hidden_units,X_train.shape[1]))/5 - 0.1

        # set bias column to 0
        alpha[:,0] = 0
    else:
        #all zeros
        alpha = np.zeros((hidden_units,X_train.shape[1]))

    beta = np.zeros((10,hidden_units+1))

    return alpha, beta

def mean_cross_entropy(X, y, alpha, beta):
    pred = []
    entropy = 0
    for a, ex in enumerate(X):

        yhat, z, probs = forward(ex, y[a], alpha, beta)

        entropy += cross_entropy(probs, y[a])

        pred.append(yhat)

    return entropy/X.shape[0], pred

# forward computation
def forward(x, y, alpha, beta):

    a = np.dot(alpha,x.T) # take linear combination
    #print("a", a)

    z = sigmoid(a)
    z = np.insert(z,0,1) # fold in z_0 = 1 to first entry of z
    #print("z", z)

    b = np.dot(beta,z)
    #print("b", b)

    probs = softmax(b.T) # get the predicted probabilities
    #print("probs", probs)
    yhat = np.argmax(probs) #return the index of prediction
    #print("yhat", yhat)
    #J = cross_entropy_forward(y,yhat)

    return yhat, z, probs

# backpropagation
def backward(x, y, alpha, beta, probs,z):

#     g_b (10,)
#     g_beta (10, 5)
#     g_z (1, 4)
#     g_a (1, 4)
#     g_alpha (4, 129)

    g_b = np.copy(probs) # create a new copy of probabilities

    g_b[y] = g_b[y] - 1 # yhat - y (?)
    g_b = np.reshape(g_b,(-1,len(g_b)))

    #print("g_b: ",g_b.shape)

    z = np.reshape(z,(-1,len(z))) #reshape from dim = 0 to dim = 1
    g_beta = np.dot(g_b.T,z)

    #print("g_beta: ", g_beta.shape)

    beta_star = beta[:,1:] # (10x4)
    g_z = np.dot(g_b,beta_star) # (1x4)

    #print("g_z: ", g_z.shape)

    z_star = z[0][1:] #(1x4)

    g_a = g_z * z_star * (1-z_star) #sigmoid backward, (1x4)

    #print("g_a: ", g_a.shape)

    x = np.reshape(x,(-1,len(x)))
    g_alpha = np.dot(g_a.T, x) #linearbackward
    #print("g_alpha: ", g_alpha.shape)

    return g_alpha, g_beta #parameter gradients

def error(y_pred,y):

    return sum(y_pred != y) / len(y_pred)

# Algorithm 1: Stochastic Gradient Descent (SGD) without shuffle (for purposes of assignment)
def sgd(X_train, y_train, X_test, y_test, alpha, beta, epochs, learning_rate):
    output = []
    for i in range(int(epochs)):

        for idx,X in enumerate(X_train):

            # forward propagation
            yhat, z, probs = forward(X, y_train[idx], alpha, beta)

            # backprop
            g_alpha, g_beta = backward(X, y_train[idx], alpha, beta, probs, z)

            #update parameters
            alpha = alpha - float(learning_rate) * g_alpha
            beta = beta - float(learning_rate) * g_beta

        # evaluate mean cross entropy (J_D(alpha,beta))
        J_D_train, y_pred_train = mean_cross_entropy(X_train,y_train,alpha,beta)
        J_D_test, y_pred_test = mean_cross_entropy(X_test, y_test, alpha, beta)

        str_train = "epoch:"+ str(i+1) + " crossentropy (train): " + str(J_D_train)
        str_test = "epoch:" + str( i+1) + " crossentropy (test): " + str(J_D_test)
        output.append(str_train)
        output.append(str_test)

    return y_pred_train, y_pred_test, output

if __name__ == '__main__':

    # take in command line inputs
    train_data, test_data = sys.argv[1], sys.argv[2] #datasets
    train_out, test_out, metrics_out = sys.argv[3], sys.argv[4], sys.argv[5] #output files
    epochs, hidden_units, init_flag, learning_rate = sys.argv[6], sys.argv[7], sys.argv[8], sys.argv[9] #hyperparameters

    # clean data sets
    X_train, y_train = read_data(train_data)
    X_test, y_test = read_data(test_data)

    # initalize hyperparameters
    alpha, beta = initalize_hyperparameters(init_flag, int(hidden_units), X_train)

    # train and predict
    y_pred_train, y_pred_test,output = sgd(X_train, y_train, X_test, y_test, alpha, beta, epochs, learning_rate)

    # calculate error
    train_error = error(y_pred_train,y_train)
    test_error = error(y_pred_test, y_test)

    # write metrics file
    metrics_out = open(metrics_out,'w')
    metrics_out.write("\n".join(output))
    metrics_out.write("\n")
    metrics_out.writelines("error (train): " + str(train_error) + "\n")
    metrics_out.writelines("error (test): " + str(test_error))
    metrics_out.close()

    # write predictions
    train_out = open(train_out,'w').write("\n".join([str(x) for x in y_pred_train]))
    test_out = open(test_out,'w').write("\n".join([str(x) for x in y_pred_test]))
