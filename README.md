# machine_learning_algorithms
My own implementation of various machine learning algorithms

################## DECISION TREE ######################

add text 





################ LOGISTIC REGRESSION ##################

feature.py
- args: <train_data.tsv> <valid_data.tsv> <test_data.tsv> <dictionary.txt> <formatted_train.tsv> <formatted_valid.tsv> <formatted_test.tsv> model_type
- feature.py is the preprocessing pipeline. It takes in 8 arguments and returns the formatted .tsv files
- the tsv files are then fed into lr.py
- .tsv files: each data point (line) is label\tword1 word2 word3 ... wordN\n
- features are converted into sparse representations based on the model choice
- Model 1: specify model_type = 1; uses bag of words feature vector
- Model 2: specify model_type = 2; uses trimmed bag of word feature vector. Trims based on count of word less than threshold t=4
- dictionary.txt: dictionary of corpus where (key=word : value=index)

lr.py
- args: <formatted_train.tsv> <formatted_valid.tsv> <formatted_test.tsv> <dictionary.txt> <train_out.labels> <test_out.labels> <metrics_out.txt> epochs
- Binary Logistic Regression implementation using stochastic gradient descent with cross entropy
- takes in formatted .tsv files from feature.py
- learning_rate = 0.1 
