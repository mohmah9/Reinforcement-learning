import numpy as np
import gym
from gym import wrappers

number_of_states = 40
iteration_maximum = 5000
initial_learning_rate = 0.9 
min_learning_rate = 0.003
gamma = 1.0
episode_steps = 200  # indicate each episode step (length of episode)

def run_episode(env, policy=None, render=False):
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    for _ in range(episode_steps):
        if render:
            env.render()
        if policy is None:
            action = env.action_space.sample()
        else:
            a,b = obs_to_state(env, obs)
            action = policy[a][b]
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
    # print("low " ,env.observation_space.low )
    # print("high" ,env.observation_space.high)
    # print (obs)
    env_dx = (env_high - env_low) / number_of_states
    pos = int((obs[0] - env_low[0])/env_dx[0])
    vel = int((obs[1] - env_low[1])/env_dx[1])
    return pos, vel

if __name__ == '__main__':
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)
    env.seed(0)
    np.random.seed(0)
    print ('START Q-LEARNING')
    q_table = np.zeros((number_of_states, number_of_states, 3))
    for i in range(iteration_maximum):
        obs = env.reset()
        total_reward = 0
        ## eta: learning rate is decreased at each step
        alpha = max(min_learning_rate, initial_learning_rate * (0.85 ** (i//100)))
        for j in range(episode_steps):
            pos, vel = obs_to_state(env, obs)
            logits = q_table[pos][vel]
            # print(logits)
            logits_exp = np.exp(logits)
            probs = logits_exp / np.sum(logits_exp)
            action = np.random.choice(env.action_space.n, p=probs)
            obs, reward, done, info = env.step(action)
            pos_, vel_ = obs_to_state(env, obs)
            q_table[pos][vel][action] = q_table[pos][vel][action] + alpha * (reward + gamma *  np.max(q_table[pos_][vel_]) - q_table[pos][vel][action])
            if done:
                # env.close() 
                break
        if i % 100 == 0:
            print('Iteration #%d' %(i))
    solution_policy = np.argmax(q_table, axis=2)
    run_episode(env, solution_policy, True)
