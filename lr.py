# Created on October 10, 2019 by Michell Li
# First run feature.py to get sparse representation of features
# Implementation of Logistic Regression
# with Stochastic Gradient Descent
# Loss Function: Cross Entropy
#
import numpy as np
import scipy as sp
import sys

# function to parse dictionary
def read_dictionary(filename):
    #create index mapping
    dict_file = open(filename).read().split("\n")
    dict_idx_pairs = {}

    for word in dict_file[:-1]:
        split = word.split(" ")
        dict_idx_pairs[split[0]] = split[1]

    return dict_idx_pairs

#parse data
# function takes in a file name and returns the parsed x and y vector
def read_data(file):

    filename = open(file,'r').read().split("\n") #load files

    data = np.array([row.strip().split('\t') for row in filename]) #split rows to have columns

    y = np.array([x[0] for x in data if x[0] != '']) #assign label set

    # x vector will be a sparse array of dictionaries as each row
    x = []
    for row in data[:-1]:
        row_dict = {}
        for word in row[1:]:
            word = word.split(":")
            row_dict[int(word[0])] = int(word[1]) # (key, value) -> (index[word], 1)

        #add in bias term
        row_dict[-1] = 1

        x.append(row_dict)

    print('x: ', len(x))
    print('y: ', y.shape)

    return x, y


def train(X, y, dictionary, epochs):
    # initialize parameters to 0
    theta = np.zeros((len(dictionary),1)) #fold in the bias term
    learning_rate = 0.1

    # train
    for i in range(epochs):
            for idx, row in enumerate(X):

                indices_used = [x for x in row] #record the theta indices that need to be updated

                theta_sparse = theta[indices_used]

                theta_T_x = sum(theta_sparse)  # take sumproduct of row to obtain theta.T*X

                J_theta = -(int(y[idx])*theta_T_x) + np.log(1+np.exp(theta_T_x)) #negative conditional log likelihood

                # dimension of (jx1), j = # theta params
                # partial derivative of negative log-likelihood J(theta)
                dJ_dtheta = (int(y[idx]) - (np.exp(theta_T_x)/(1+np.exp(theta_T_x))))

                theta[indices_used] = theta_sparse + learning_rate * dJ_dtheta   # update theta (weights)

    return theta

# make predictions
def predict(X, y, theta):

    y_hat = []
    for row in X:

        indices = [x for x in row]

        thetas_needed = theta[indices]

        u = sum(thetas_needed)  # (theta.T)X

        p_y_1 = 1 / (1+np.exp(-u)) # (P(Y=1|X, theta))

        p_y_0 = 1 - p_y_1 # (P(Y=0|X, theta))

        # the classification goes to the class w/ highest probability
        if p_y_1 > p_y_0:
            y_hat.append("1")
        else:
            y_hat.append("0")

    return y_hat

def evaluate(y_hat, y):
    #compute the error
    return sum(y_hat != y) / len(y)

if __name__ == '__main__':

    # take in command line inputs
    train_data, valid_data, test_data = sys.argv[1], sys.argv[2], sys.argv[3]
    dictionary = sys.argv[4]
    train_out, test_out, metrics_out = sys.argv[5], sys.argv[6], sys.argv[7] #output files
    epochs = sys.argv[8]

    # parse files
    train_x, train_y = read_data(train_data)
    valid_x, valid_y = read_data(valid_data)
    test_x, test_y = read_data(test_data)
    dictionary = read_dictionary(dictionary)

    # train dataset
    theta = train(train_x, train_y, dictionary, int(epochs))

    # make predictions
    y_hat_train = predict(train_x, train_y, theta)
    y_hat_valid = predict(valid_x, valid_y, theta)
    y_hat_test = predict(test_x, test_y, theta)

    #evaluate
    train_error = evaluate(y_hat_train, train_y)

    valid_error = evaluate(y_hat_valid, valid_y)
    test_error = evaluate(y_hat_test, test_y)

    # write .labels file
    train_out = open(train_out,'w').write("\n".join(y_hat_train))
    test_out = open(test_out,'w').write("\n".join(y_hat_test))

    #write metrics file
    metrics_out = open(metrics_out,'w')
    metrics_out.writelines("error (train): " + str(train_error) + "\n")
    metrics_out.writelines("error (test): " + str(test_error))
    metrics_out.close()
