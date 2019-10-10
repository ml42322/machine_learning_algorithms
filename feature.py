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

# function takes in a file name and returns the parsed x and y vector
def read_data(file):

    filename = open(file,'r').readlines() #load files

    data = np.array([row.strip().split('\t') for row in filename]) #split rows to have columns

    y = np.array([x[0] for x in data]) #assign label set

    x = np.array([x[1] for x in data]) # assign feature set

    print('x: ', x.shape)
    print('y: ', y.shape)

    return x, y

def create_clean_data(dataset, y, dictionary, feature_flag, output_file):

    output_file = open(output_file,'w')
    tuple_word_list = []

    #loop through ever row in teh dataset
    for idx, document in enumerate(dataset):
        model_tuple_word_list = {}
        example = document.split(" ")

        # loop through every word in the document
        for word in example:
            try:
                #if the word is in the dictionary
                mapping = dictionary[word]
            except:
                continue

            # if the feature_flag = 1 (model 1)
            if feature_flag == "1":

                # then create a feature for the document: (index[word], 1)
                model_tuple_word_list[mapping] = 1

            # if the feature_flag = 2 (model 2)
            elif feature_flag == "2":
                #if the word is already in the new word list then increase the value count
                if mapping in model_tuple_word_list:

                    model_tuple_word_list[mapping] += 1
                else:
                    # if the word is not yet in the new word list, create it
                    model_tuple_word_list[mapping] = 1


        # append the document to a clean data list
        tuple_word_list.append(model_tuple_word_list)

        if feature_flag == "1":
            # write that document to the output .tsv file
            x = '\t'.join([str(x)+":"+str(model_tuple_word_list[x]) for x in model_tuple_word_list])

        # if the count is < 4, then include the word as a feature
        elif feature_flag == "2":
            x = '\t'.join([str(x)+":1" for x in model_tuple_word_list if model_tuple_word_list[x] < 4])

        output_file.writelines(y[idx] + '\t' + x + '\n')

    #print(len(tuple_word_list))
    output_file.close()

if __name__ == '__main__':

    # take in command line inputs
    train_data, valid_data, test_data = sys.argv[1], sys.argv[2], sys.argv[3]
    dictionary = sys.argv[4]
    formatted_train, formatted_valid, formatted_test = sys.argv[5], sys.argv[6], sys.argv[7] #output files
    feature_flag = sys.argv[8]

    # parse files
    train_x, train_y = read_data(train_data)
    valid_x, valid_y = read_data(valid_data)
    test_x, test_y = read_data(test_data)
    dictionary = read_dictionary(dictionary)

    #create cleaned datasets
    create_clean_data(train_x, train_y, dictionary, feature_flag, formatted_train)
    create_clean_data(valid_x, valid_y, dictionary, feature_flag, formatted_valid)
    create_clean_data(test_x, test_y, dictionary, feature_flag, formatted_test)
