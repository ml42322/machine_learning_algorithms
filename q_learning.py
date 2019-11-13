# Created by Michell Li on November 10, 2019

from environment import MountainCar
import numpy as np
import scipy as sp
import sys

def q_learning(env, mode, episodes, max_iterations, epsilon, gamma, learning_rate):

    #initialize weights
    # state_space = 2 or 2048 (raw, tile), action_space = 3
    w = np.zeros((env.state_space, env.action_space))
    b = 0
    rewards = []

    for e in range(episodes):

        # updates per episode
        total_reward = 0
        state = env.reset()
        s = get_state(env, state, mode)
        #print("current state: ", s)

        for i in range(max_iterations):
            #print()
            #print("iteration ", i)
            # select action a and execute
            action = get_action(env, epsilon, s, w, b)
            #print("action: ", action)

            # receive reward
            state_prime, reward, done = env.step(action)
            total_reward += reward

            #print("s_prime: ",state_prime,reward,done)

            # observe new state s'
            s_prime = get_state(env, state_prime, mode)
            a_prime = get_action(env, epsilon, s_prime, w, b)
            #print("s_prime, a_prime: ", s_prime, a_prime)

            # update rule
            q = np.matmul(s.T,w[:,action]) + b
            q_prime = np.matmul(s_prime.T,w[:,a_prime]) + b
            delta = q - (reward + gamma * q_prime)
            dq_dw = np.zeros((env.state_space,env.action_space))
            dq_dw[:,action] = s[:,0]

            w = w - learning_rate * delta * dq_dw
            b = b - learning_rate * delta
            #print("w: ", w)
            #print("b: ", b)
            # set up next iteration
            s = s_prime
            # if done, reset the environment and end the episode
            if done:
                env.reset()
                break

        rewards.append(total_reward)
    print("w: ", w)
    print("b: ", b)
    return w, b, rewards

def get_action(env, epsilon, s, w, b):
    # epsilon-greedy action selection selects optimal actions w/ probability 1-epsilon
    probability_of_selection = np.random.random()
    #print(probability_of_selection, epsilon)

    # get optimal action
    if(probability_of_selection > epsilon):
        # q(s,a;w) = s.T*w + b
        action = np.argmax(np.matmul(s.T,w) + b)

    #select random action
    else:
        action = np.random.randint(0,env.action_space)

    return action

def get_state(env, state, mode):

    state_key = list(state.keys())

    s = [0] * env.state_space

    if mode == "raw":
        # format is {0 -> position, 1 -> velocity}
        s = np.array(list(state.values()))
        s = s.reshape(len(s),1)

    elif mode =="tile":
        # format is: {tile index -> 1} (sparse)
        for each in state_key:
            s[each] = 1
        s = np.array(s).reshape(len(s),1)

    return s


if __name__ == "__main__":

    # take in command line inputs
    mode, weight_out = sys.argv[1], sys.argv[2] #datasets
    returns_out, episodes, max_iterations = sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
    epsilon, gamma, learning_rate = float(sys.argv[6]), float(sys.argv[7]), float(sys.argv[8])

    # instantiate a new instance of Mountain Car with selected mode
    env = MountainCar(mode=mode)
    env.reset()

    #learn weights
    w, b, rewards = q_learning(env, mode, episodes, max_iterations, epsilon, gamma, learning_rate)

    #write output
    w_ravel = np.array(np.ravel(w))
    open(weight_out,'w').write(str(b[0]) + "\n" + "\n".join([str(x) for x in w_ravel]))
    open(returns_out,'w').write("\n".join([str(x) for x in rewards]))
