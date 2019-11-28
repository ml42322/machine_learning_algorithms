# Created on November 24, 2019 by Michell Li
# Algorithm to learn the hidden Markov model parameters
# needed to apply the viterbi algorithm
# takes in training data, index to word, index to tag as .txt files
# outputs .txt files for prior, emission probabilities and transition probabilities

import numpy as np
import sys

# take in command line inputs
train = sys.argv[1]
index_to_word =  sys.argv[2]
index_to_tag = sys.argv[3]
hmmprior, hmmemit, hmmtrans = sys.argv[4], sys.argv[5], sys.argv[6] #output files

#read in data
train = open(train).read().strip().split("\n")
index_to_tag = open(index_to_tag).read().strip().split("\n")
index_to_word = open(index_to_word).read().strip().split("\n")

#data preprocessing
tag_data = []
word_data = []

# loop through every sentence in the training set 
for line in train:

    #create separate tag and word arrays
    word_arr = [x.split('_')[0] for x in line.split()]
    tag_arr = [x.split('_')[1] for x in line.split()]

    #convert tags to index values
    word_arr_idx = [index_to_word.index(x) for x in word_arr]
    tag_arr_idx = [index_to_tag.index(x) for x in tag_arr]

    #append to final array
    word_data.append(word_arr_idx)
    tag_data.append(tag_arr_idx)

#flatten arraylist and get unique tags (j)
tags = list(set([x for sublist in tag_data for x in sublist]))
tags.sort()

#initialization probabilities
# get the first column
first_col = [row[0] for row in tag_data]

# get N (count) for each j
priors =  np.ones((len(index_to_tag),1))

for each in first_col:
    priors[each] += 1

priors = np.array([x/(sum(priors)) for row in priors for x in row])

#transition probabilities
#a_jk
#create matrix
A = np.ones((len(tags),len(tags)))

# count the pairs
for row in tag_data:

    for (j,k) in zip(row,row[1:]):

        # get numerator N_jk
        # number of times state s_j is followed by s_k
        A[j][k] += 1

# divide numerator/denominator
A = np.array([[f/sum(row) for f in row] for row in A])

#emission probabilities
#b_jk

#create matrix (include pseudocount)
B = np.ones((len(tags),len(index_to_word)))

# count the pairs
for word_row,tag_row in zip(word_data,tag_data):

    for (j,k) in zip(tag_row,word_row):
        # get numerator N_jk
        # number of times state s_j is associated with word k
        B[j][k] += 1

# divide numerator/denominator
B = np.array([[f/sum(row) for f in row] for row in B])

#write to files
np.savetxt(hmmprior,priors)
np.savetxt(hmmemit, B)
np.savetxt(hmmtrans, A)
