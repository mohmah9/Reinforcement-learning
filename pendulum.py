import numpy as np
import math
import gym
# from gym import wrappers

number_of_states = 20
slices_of_action=30
iteration_maximum = 15000
initial_learning_rate = 0.9
min_learning_rate = 0.03
gamma = 0.85
epsilon = 0.09
episode_steps = 2000  # indicate each episode step (length of episode)

def run_episode(env, policy=None, render=False):
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    while True:
        if render:
            env.render()
        if policy is None:
            action = env.action_space.sample()
        else:
            deg , vel = obs_to_state(env, obs)
            action = (policy[deg][vel]/float(slices_of_action/4))-2
            action=[action]
        obs, reward, done, info = env.step(action)
        total_reward += gamma ** step_idx * reward
        step_idx += 1
        if done:
            # env.close()
            if render:
                print("score would be :" ,total_reward)
            break

def obs_to_state(env, obs):
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    tan=obs[1]/float(obs[0])
    deg=np.degrees(np.arctan(tan))
    if obs[0] < 0 :
        deg = 180 + deg
    if deg < 0 :
        deg = 360 + deg
    # print("low " ,env.observation_space.low )
    # print("high" ,env.observation_space.high)
    # print (obs)
    env_dx = (env_high - env_low) / number_of_states
    # tan_dx = 360 
    # print (deg)
    deg = int(deg/18)
    # print(deg)
    vel = int((obs[2] - env_low[2])/env_dx[2])
    return deg, vel

if __name__ == '__main__':
    env_name = 'Pendulum-v0'
    env = gym.make(env_name)
    env.seed(0)
    np.random.seed(0)
    print ('START Q-LEARNING')
    q_table = np.zeros((20, number_of_states, slices_of_action))
    for i in range(iteration_maximum):
        obs = env.reset()
        total_reward = 0
        ## eta: learning rate is decreased at each step
        alpha = max(min_learning_rate, initial_learning_rate * (0.85 ** (i//100)))
        for j in range(episode_steps):
            deg, vel = obs_to_state(env, obs)
            if np.random.uniform(0, 1) < epsilon:
                # get random action
                action2 = [np.random.choice(slices_of_action)]
            # print(deg , vel)
            else:
                logits = q_table[deg][vel]
                # print(logits)
                logits_exp = np.exp(logits)
                probs = logits_exp / np.sum(logits_exp)
                action = np.random.choice(slices_of_action, p=probs)
                # action = np.argmax(logits)
                action2 = (action/float(slices_of_action/4))-2
                # print(action2)
                action2 = [action2]
                # print(action ,action2)
            obs, reward, done, info = env.step(action2)
                # total_reward += reward
                # update q table
            deg_, vel_ = obs_to_state(env, obs)
            # print ("a = ", a , "b =" , b)
            # print ("a_ = ", a_ , "b_ =" , b_)
            # print ("q_table[a][b][action]" , q_table[a][b][action] , "eta " , eta)
            # print("np.max(q_table[a_][b_])" , np.max(q_table[a_][b_]))
            # print("np.max(q_table[a_][b_]) - q_table[a][b][action]" , np.max(q_table[a_][b_]) - q_table[a][b][action])
            q_table[deg][vel][action] = q_table[deg][vel][action] + alpha * (reward + gamma *  np.max(q_table[deg_][vel_]) - q_table[deg][vel][action])
            if done:
                # env.close() 
                break
        if i % 100 == 0:
            print('Iteration #%d' %(i))
    # print(q_table)
    solution_policy = np.argmax(q_table, axis=2)
    run_episode(env, solution_policy, True)
