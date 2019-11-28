# Created on November 25, 2019 by Michell Li
# Viterbi Algorithm
# predicts the most probable label sequence y1,...,yT given
# the observation x1,..,xT
# takes in test dataset, index to work, index to tag as .txt files
# takes in prior, emission, and transition probabilities from learnhmm.py as .txt files
# outputs the accuracy and predicted sequence

import sys
import numpy as np

# take in command line inputs
test_input = sys.argv[1]
index_to_word =  sys.argv[2]
index_to_tag = sys.argv[3]

pi = np.loadtxt(sys.argv[4])
B = np.loadtxt(sys.argv[5])
A =  np.loadtxt(sys.argv[6])

predicted_file, metric_file = sys.argv[7], sys.argv[8]  #output files

#read in data
test_data = open(test_input).read().strip().split("\n")
index_to_tag = open(index_to_tag).read().strip().split("\n")
index_to_word = open(index_to_word).read().strip().split("\n")

#data preprocessing
word_data = []
word_words = []

for line in test_data:

    #create separate tag and word arrays
    word_arr = [x.split('_')[0] for x in line.split()]
    word_words.append(word_arr)

    #convert tags to index values
    word_arr_idx = [index_to_word.index(x) for x in word_arr]

    #append to final array
    word_data.append(word_arr_idx)

tags = [i for i in range(len(index_to_tag))]

#initialize t=1
# w1_j = pi_j * b_jx1
total_predictions = []

# loop through every row in dataset
for line in word_data:

    w = np.zeros((len(tags),len(line)))
    b = np.zeros((len(tags),len(line)-1))

    # loop through tags
    for j in tags:
        w[j][0] = np.log(pi[j]) + np.log(B[j][line[0]])

    # loop through time t
    for t in range(1,len(line)):

        for j in tags:

            #product of probabilities  through path y_1, ..., y_t-1
            p_sj = np.log(B[j][line[t]]) + np.log(np.array(A[:,j])) + w[:,t-1]

            # store probability of the most probable state
            w[j,t] = np.max(p_sj)

            # store pointer
            b[j,t-1] = np.argmax(p_sj)

    #make prediction
    yhat_T = np.argmax(w[:,-1])

    prediction = [yhat_T]

    pred_idx = 0

    #trace sequence backwards from yT,...y1
    for t in range(len(line) - 2, -1, -1):
        last_state = prediction[pred_idx]
        yhat_tm1 = b[int(last_state),t]
        prediction.append(yhat_tm1)
        pred_idx += 1

    # order sequence y1,...yT
    prediction.reverse()

    total_predictions.append(prediction)

# final output
output = []
for row_idx, row in enumerate(total_predictions):
    for idx, y in enumerate(row):
        y_tag = index_to_tag[int(y)]
        word_words[row_idx][idx] = word_words[row_idx][idx] + "_" + y_tag
    output.append(' '.join(word_words[row_idx]))

test_data_ravel = []
predictions_ravel = []
for test, word in zip(test_data,word_words):
    test = test.split()
    test_data_ravel.extend(test)
    predictions_ravel.extend(word)

#accuracy
accuracy = sum(np.array(predictions_ravel)==np.array(test_data_ravel))/len(test_data_ravel)

#write output files
metric_file = open(metric_file,'w')
metric_file.write("Accuracy: " + str('{:0.4f}'.format(accuracy)) + "\n")
np.savetxt(predicted_file,output,fmt='%s')
