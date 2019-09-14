# Created by Michell Li, on September 14, 2019
import numpy as np
import scipy as sp
import sys

# this function takes in a filename and returns
# a variable for features, labels, and column names
def read_data(file):

    filename = open(file,'r').readlines() #load files

    data = np.array([row.strip().split('\t') for row in filename]) #split rows to have columns

    col_names = [x.strip(' ') for x in data[0,:-1]] #get column names

    y = data[1:,-1] #assign label set

    x = data[1:,:-1] #delete header row and label column

    indices = [int(idx) for idx,x in enumerate(y)]
    x = np.insert(x,x.shape[1],indices,axis=1)

    #print('x: ', x.shape)
    #print('y: ', y.shape)

    return x, y, col_names

# creates tree object
class Tree(object):
    def __init__(self):
        self.left = None
        self.right = None
        self.X = None
        self.y = None
        self.attribute = None
        self.label = None

# train decision tree
def train(X, y, column_names, max_depth=3):
    root = Tree()
    root.X = X
    root.y = y
    train_tree(root, max_depth, 0, column_names)
    return root

# recursively create tree
def train_tree(tree, max_depth, depth, remaining_features):

    #stopping criteria
    node_error = error(tree.y)

    # If data is perfectly classified, or there is no more features to split on
    # or if the max depth has been reached
    if node_error == 0 or len(remaining_features) == 0 or depth >= max_depth:
        tree.label = get_label(tree.y)
        return

    # Else recurse on each of the descendants
    idx_best_attribute = mutual_information(tree.X, tree.y, remaining_features)
    tree.attribute = remaining_features[idx_best_attribute]

    # Partition node's data into its descendants
    attribute_classes = np.unique(tree.X[:,idx_best_attribute]) # get unique options for the attribute

    tree.left = Tree() #create left branch
    tree.left.X = np.delete(tree.X[np.where(tree.X[:,idx_best_attribute]==attribute_classes[0])],idx_best_attribute,axis=1) #subset of data in left branch
    tree.left.y = tree.y[np.where(tree.X[:, idx_best_attribute]==attribute_classes[0])]

    tree.right = Tree() #create right branch
    tree.right.X = np.delete(tree.X[np.where(tree.X[:,idx_best_attribute]==attribute_classes[1])],idx_best_attribute,axis=1)#subset of data in right branch
    tree.right.y = tree.y[np.where(tree.X[:, idx_best_attribute]==attribute_classes[1])]

    #remove used attribute
    updated_remaining_features = list(remaining_features)
    updated_remaining_features.remove(updated_remaining_features[idx_best_attribute])

    # recurse down branch
    train_tree(tree.left, max_depth, depth + 1, updated_remaining_features)
    train_tree(tree.right, max_depth, depth + 1, updated_remaining_features)

# calculate entropy for node splits
def entropy(y): #need to handle inf/0 case

    # H(Y) = -summation(P(Y=y)log(Y=y))
    classes = np.unique(y) #get total number of classes

    if len(classes) < 2:
        #print(classes[0],len(y), "total", len(y))
        #print()
        return 0
    else:
        class_0 = len(np.where(y==classes[0])[0]) #np.where returns a tuple, so get the first element
        class_1 = len(np.where(y==classes[1])[0]) # set the second class to 0 if there is only 1

    total = len(y)
    #print(classes[0],class_0, classes[1], class_1, "total ", total)

    h_y = (-1) * ((class_0/total)*np.log2(class_0/total) + (class_1/total)*np.log2(class_1/total))
    #print()
    return h_y

def mutual_information(X, y, remaining_features):
    mutual_info_arr = []

    h_y = entropy(y)

    for i,feature in enumerate(remaining_features):

        attribute_classes = np.unique(X[:,i]) # get unique options for the attribute

        # H(Y|X=0) 0 = No
        #print('No')
        y_zero = y[np.where(X[:,i]==attribute_classes[0])]
        y_X0 = entropy(y_zero)

        #print('Yes')
        # H(Y|X=1) 1 = Yes
        if len(attribute_classes) == 2:
            y_one = y[np.where(X[:,i]==attribute_classes[1])]
            y_X1 = entropy(y_one)
        else:
            y_one = []
            y_X1 = 0

        # P(X=0)
        p_X0 = len(y_zero) / len(y)

        # P(X=1)
        p_X1 = len(y_one) / len(y)

        # I(Y;X) = H(Y) - H(Y|X)
        i_YX = h_y - p_X0*y_X0 - p_X1*y_X1

        mutual_info_arr.append(i_YX)
        #print("Summary for feature", feature)
        #print("No: ", len(y_zero), " Yes: ", len(y_one), " Mutual Info: ", i_YX)
        #print()

    idx_best_feature = np.argmax(mutual_info_arr)
    #print(remaining_features[idx_best_feature])
    #print(mutual_info_arr)
    return idx_best_feature

def error(y):

    # error rate = label with least examples (majority vote)
    classes = np.unique(y) #get total number of classes

    if len(classes) < 2:
        return 0

    class_0 = len(np.where(y==classes[0])[0]) #np.where returns a tuple, so get the first element
    class_1 = len(np.where(y==classes[1])[0])
    total = len(y)

    if class_0 > class_1:
        error = class_1/total
    else:
        error = class_0/total

    return error

def get_label(y):

    classes = np.unique(y) #get total number of classes
    if len(classes) < 2:
        return classes[0]

    class_0 = len(np.where(y==classes[0])[0]) #np.where returns a tuple, so get the first element
    class_1 = len(np.where(y==classes[1])[0])

    if class_0 > class_1:
        return classes[0]
    else:
        return classes[1]

def evaluate(predictions,y_true):

    error = sum(predictions != y_true) / len(y_true)

    return error

def pretty_print(X, column_names, tree, depth = 1):

    if tree:

        #get binary classes used
        classes = np.unique(tree.y)

        #print root node
        if tree.attribute != None and depth==1:

            class_0 = len(np.where(tree.y==classes[0])[0])
            class_1 = len(np.where(tree.y==classes[1])[0])
            class_output = '[' + str(class_0) + ' ' + str(classes[0]) + ' /' + str(class_1) + " " + str(classes[1]) +']'
            print(class_output)

        # print for 'no' branches
        if tree.attribute != None:
            attribute_classes = get_classes(X[:,column_names.index(tree.attribute)])
            class_0 = len(np.where(tree.left.y==classes[0])[0])
            class_1 = len(np.where(tree.left.y==classes[1])[0])
            class_output = '[' + str(class_0) + ' ' + str(classes[0]) + ' /' + str(class_1) + " " + str(classes[1]) +']'

            print("| " * depth, tree.attribute, ' = ', str(attribute_classes[0])+': ', end='')
            print(class_output)

        pretty_print(X, column_names,tree.left, depth+1)

        # print for 'yes' branches
        if tree.attribute != None:
            attribute_classes = get_classes(X[:,column_names.index(tree.attribute)])
            class_0 = len(np.where(tree.right.y==classes[0])[0])
            class_1 = len(np.where(tree.right.y==classes[1])[0])
            class_output = '[' + str(class_0) + ' ' + str(classes[0]) + ' /' + str(class_1) + " " + str(classes[1]) +']'
            print("| " * depth,tree.attribute, ' = ',str(attribute_classes[1])+': ', end='')
            print(class_output)

        pretty_print(X, column_names,tree.right, depth+1)

def get_leaves(tree):
    if tree is None:
        return 0
    if(tree.left is None and tree.right is None): #leaf
        for index in tree.X:
            output_labels[int(index)] = tree.label

        return 1
    else:
        return get_leaves(tree.left) + get_leaves(tree.right)

#make predictions on unseen examples
def predict(tree, X_test, y_test, col_names_test):
    predictions = []
    for row in X_test:
        prediction = traverse(tree,row,col_names_test,X_test)
        predictions.append(prediction)

    return predictions

#send unseen examples down tree to make classification
def traverse(tree, data_point, cols_names_test,X_test):

    if tree.attribute is None:
        return tree.label

    classes = np.unique(X_test[:,cols_names_test.index(tree.attribute)])

    if (tree.left is None and tree.right is None): #leaf
        return tree.label
    elif data_point[cols_names_test.index(tree.attribute)] == classes[0]: # left branch
        return traverse(tree.left, data_point, cols_names_test, X_test)
    else:
        return traverse(tree.right, data_point, cols_names_test, X_test) #right branch

# create .label files for predictions
def write_labels_files(trained_DT):
    output_labels = ['' for idx,x in enumerate(trained_DT.y)]
    total_leaves = get_leaves(trained_DT)

def get_classes(column):

    classes = np.unique(column)

    return classes

if __name__ == '__main__':

    train_input, test_input, max_depth, train_out, test_out, metrics_out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]

    # pass data through pipeline
    X_train, y_train, col_names = read_data(train_input)
    X_test, y_test, col_names_test = read_data(test_input)

    #train decision tree
    trained_DT = train(X_train, y_train, col_names, max_depth = int(max_depth))

    #make predictions
    y_pred_train = predict(trained_DT, X_train, y_train, col_names)
    y_pred_test = predict(trained_DT, X_test, y_test, col_names_test)

    #write predictions to .labels file
    train_out = open(train_out,'w').write("\n".join(y_pred_train))
    test_out = open(test_out,'w').write("\n".join(y_pred_test))

    #evaluate predictions
    error_train = evaluate(y_pred_train, y_train)
    error_test = evaluate(y_pred_test,y_test)

    #write error to metrics out file
    metrics_out = open(metrics_out,'w')
    metrics_out.writelines("error (train): " + str(error_train) + "\n")
    metrics_out.writelines("error (test): " + str(error_test))

    #print decision tree to cmd prompt
    pretty_print(X_train,col_names,trained_DT)
