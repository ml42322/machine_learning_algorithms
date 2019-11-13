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

################ NEURAL NETWORK ##################
- Neural network implementation using SGD and one hidden layer.
- Loss fn used is mean cross entropy

Example: 
- Epochs: 2
- Hidden units: 4
- Initialization: 2 weights (init_flag=='1' for random weight initialization, init_flag=='2' for zero weights)
- Learning rate: 0.1

Args:
python neuralnet.py <train.csv> <test.csv> <model1train_out.labels> <model1test_out.labels> <model1metrics_out.txt> 2 4 2 0.1

############# REINFORCEMENT LEARNING ###############
My implementation of the Mountain Car problem using Q-learning with epsilon-greedy action selection.

https://gym.openai.com/envs/MountainCar-v0/

- args: <mode> <weight_out> <returns_out> <episodes> <max_iterations> <epsilon> <gamma> <learning_rate>
 
 Example:
 
 Mode- raw or tile 
 
 python q_learning.py raw weight.out returns.out 4 200 0.05 0.99 0.01

