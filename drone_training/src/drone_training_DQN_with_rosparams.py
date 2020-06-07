#!/usr/bin/env python
import rospy
import time
# Inspired by https://keon.io/deep-q-learning/
import random
import gym
import math
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import myquadcopter_env

class DQNDroneSolver():
    def __init__(self, n_episodes=100, n_win_ticks=-1800, min_episodes= 50, max_env_steps=None, gamma=1.0, epsilon=1.0, epsilon_min=0.01, epsilon_log_decay=0.995, alpha=0.01, alpha_decay=0.01, batch_size=64, monitor=False, quiet=False):
        self.memory = deque(maxlen=100000)
        self.env = gym.make('QuadcopterLiveShow-v0')
        #if monitor: self.env = gym.wrappers.Monitor(self.env, '../data/cartpole-1', force=True)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_log_decay
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.n_episodes = n_episodes
        self.n_win_ticks = n_win_ticks
        self.min_episodes = min_episodes
        self.batch_size = batch_size
        self.quiet = quiet
        self.max_env_steps = max_env_steps
        #if max_env_steps is not None: self.env._max_episode_steps = max_env_steps

        # Init model
        self.model = Sequential()
        self.model.add(Dense(24, input_dim=2, activation='tanh'))
        self.model.add(Dense(48, activation='tanh'))
        self.model.add(Dense(4, activation='linear'))
        self.model.compile(loss='mse', optimizer=Adam(lr=self.alpha, decay=self.alpha_decay))
        
        # Set up ROS related variables
        self.episode_num = 0
        self.cumulated_episode_reward = 0
        

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state, epsilon):
        return self.env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(self.model.predict(state))

    def get_epsilon(self, t):
        return max(self.epsilon_min, min(self.epsilon, 1.0 - math.log10((t + 1) * self.epsilon_decay)))

    def preprocess_state(self, state):
        return np.reshape(state, [1, 2])

    def replay(self, batch_size):
        x_batch, y_batch = [], []
        minibatch = random.sample(
            self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in minibatch:
            y_target = self.model.predict(state)
            y_target[0][action] = reward if done else reward + self.gamma * np.max(self.model.predict(next_state)[0])
            x_batch.append(state[0])
            y_batch.append(y_target[0])
        
        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def run(self):
        
        scores = deque(maxlen=100)

        for e in range(self.n_episodes):
            state = self.preprocess_state(self.env.reset())
            done = False
            i = 0
            while not done:
                # openai_ros doesnt support render for the moment
                #self.env.render()
                action = self.choose_action(state, self.get_epsilon(e))
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.preprocess_state(next_state)
		if i == self.max_env_steps:
			reward = -1000
			done = True
                self.remember(state, action, reward, next_state, done)
                state = next_state
                self.cumulated_episode_reward += reward
                i += 1
                # To get a render of around 30 FPS
                time.sleep(0.03)
                

            scores.append(self.cumulated_episode_reward)
            mean_score = np.mean(scores)
            if mean_score >= self.n_win_ticks and e >= self.min_episodes:
                if not self.quiet: print('Ran {} episodes. Solved after {} trials'.format(e, e - self.min_episodes))
                return e - self.min_episodes
            if e % 1 == 0 and not self.quiet:
                print('[Episode {}] - Mean survival time over last {} episodes was {} ticks.'.format(e, self.min_episodes ,mean_score))

            self.replay(self.batch_size)
            
            self._update_episode()
            

        if not self.quiet: print('Did not solve after {} episodes'.format(e))
        return e
        
    def _publish_reward_topic(self, reward, episode_number=1):
        """
        This function publishes the given reward in the reward topic for
        easy access from ROS infrastructure.
        :param reward:
        :param episode_number:
        :return:
        """
        #reward_msg = RLExperimentInfo()
        #reward_msg.episode_number = episode_number
        #reward_msg.episode_reward = reward
        #self.reward_pub.publish(reward_msg)
        
    def _update_episode(self):
        """
        Publishes the cumulated reward of the episode and 
        increases the episode number by one.
        :return:
        """
        self._publish_reward_topic(
                                    self.cumulated_episode_reward,
                                    self.episode_num
                                    )
        self.episode_num += 1
        self.cumulated_episode_reward = 0
        
if __name__ == '__main__':
    rospy.init_node('drone_training_DQN', anonymous=True)
   
     
    n_episodes = rospy.get_param('/drone_hex/episodes_training')
    n_win_ticks = rospy.get_param('/drone_hex/n_win_ticks')
    min_episodes = rospy.get_param('/drone_hex/min_episodes')
    max_env_steps = rospy.get_param("/drone_hex/max_env_steps")
    gamma =  rospy.get_param('/drone_hex/gamma')
    epsilon = rospy.get_param('/drone_hex/epsilon')
    epsilon_min = rospy.get_param('/drone_hex/epsilon_min')
    epsilon_log_decay = rospy.get_param('/drone_hex/epsilon_decay')
    alpha = rospy.get_param('/drone_hex/alpha')
    alpha_decay = rospy.get_param('/drone_hex/alpha_decay')
    batch_size = rospy.get_param('/drone_hex/batch_size')
    monitor = rospy.get_param('/drone_hex/monitor')
    quiet = rospy.get_param('/drone_hex/quiet')
    
    
    agent = DQNDroneSolver(n_episodes,
			   n_win_ticks,
			   min_episodes,
			   max_env_steps,
			   gamma,
			   epsilon,
			   epsilon_min,
			   epsilon_log_decay,
			   alpha,
			   alpha_decay,
			   batch_size,
			   monitor,
			   quiet)

    agent.run()
