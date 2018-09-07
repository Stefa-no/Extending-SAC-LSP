import numpy as np
import time

from rllab.misc import logger


def rollout(env, policy, path_length, render=False, speedup=None):
    Da = env.action_space.flat_dim
    Do = env.observation_space.flat_dim

    observation = env.reset()
    policy.reset()

    observations = np.zeros((path_length + 1, Do))
    actions = np.zeros((path_length, Da))
    terminals = np.zeros((path_length, ))
    rewards = np.zeros((path_length, ))
    agent_infos = []
    env_infos = []

    t = 0
    for t in range(path_length):

        action, agent_info = policy.get_action(observation)
        next_obs, reward, terminal, env_info = env.step(action)

        agent_infos.append(agent_info)
        env_infos.append(env_info)

        actions[t] = action
        terminals[t] = terminal
        rewards[t] = reward
        observations[t] = observation

        observation = next_obs

        if render:
            env.render()
            time_step = 0.05
            time.sleep(time_step / speedup)

        if terminal:
            break

    observations[t + 1] = observation

    path = {
        'observations': observations[:t + 1],
        'actions': actions[:t + 1],
        'rewards': rewards[:t + 1],
        'terminals': terminals[:t + 1],
        'next_observations': observations[1:t + 2],
        'agent_infos': agent_infos,
        'env_infos': env_infos
    }

    return path


def rollouts(env, policy, path_length, n_paths):
    paths = [
        rollout(env, policy, path_length)
        for i in range(n_paths)
    ]

    return paths


class Sampler(object):
    def __init__(self, max_path_length, min_pool_size, batch_size):
        self._max_path_length = max_path_length
        self._min_pool_size = min_pool_size
        self._batch_size = batch_size

        self.env = None
        self.policy = None
        self.pool = None

    def initialize(self, env, env_copy, policy, pool):
        self.env = env
        self.env_copy = env_copy
        self.policy = policy
        self.pool = pool

    def sample(self):
        raise NotImplementedError

    def batch_ready(self):
        enough_samples = self.pool.size >= self._min_pool_size
        return enough_samples

    def random_batch(self):
        return self.pool.random_batch(self._batch_size)

    def terminate(self):
        self.env.terminate()

    def log_diagnostics(self):
        logger.record_tabular('pool-size', self.pool.size)


class SimpleSampler(Sampler):
    def __init__(self, **kwargs):
        super(SimpleSampler, self).__init__(**kwargs)

        self._path_length = 0
        self._path_return = 0
        self._last_path_return = 0
        self._max_path_return = -np.inf
        self._n_episodes = 0
        self._current_observation = None
        self._total_samples = 0

    def sample(self):
        if self._current_observation is None:
            self._current_observation = self.env.reset()

        action, _ = self.policy.get_action(self._current_observation)
        next_observation, reward, terminal, info = self.env.step(action)
        self._path_length += 1
        self._path_return += reward
        self._total_samples += 1

        self.pool.add_sample(
            observation=self._current_observation,
            action=action,
            reward=reward,
            terminal=terminal,
            next_observation=next_observation)

        if terminal or self._path_length >= self._max_path_length:
            self.policy.reset()
            self._current_observation = self.env.reset()
            self._path_length = 0
            self._max_path_return = max(self._max_path_return,
                                        self._path_return)
            self._last_path_return = self._path_return

            self._path_return = 0
            self._n_episodes += 1

        else:
            self._current_observation = next_observation

    def log_diagnostics(self):
        super(SimpleSampler, self).log_diagnostics()
        logger.record_tabular('max-path-return', self._max_path_return)
        logger.record_tabular('last-path-return', self._last_path_return)
        logger.record_tabular('episodes', self._n_episodes)
        logger.record_tabular('total-samples', self._total_samples)


class Episode_experience():
    def __init__(self):
        self.memory = []
        
    def add(self, observation, action, reward, next_observation, terminal, goal):
        self.memory += [(observation, action, reward, next_observation, terminal, goal)]
        
    def clear(self):
        self.memory = []


class PrioritizedSampler(Sampler):
    def __init__(self, **kwargs):
        super(PrioritizedSampler, self).__init__(**kwargs)

        self._path_length = 0
        self._path_return = 0
        self._last_path_return = 0
        self._max_path_return = -np.inf
        self._n_episodes = 0
        self._current_observation = None
        self._current_goal = None
        self._total_samples = 0
        #new stuff
        self.alpha = 0.5
        self.ep_experience = Episode_experience()
        self.K = 4
        self.her_type = 0  #with this I control what experiment to run, should become param
        self.goal_sampl_type =0 #with this I control what type of goal sampling to do
        self.longest_path = 0

    def sample(self, total_step):
        if self._current_observation is None:
            self._current_observation = self.env.reset()
            self._current_goal =  np.random.uniform(low=-10.0, high=10.0, size=len(self._current_observation))

        action, _ = self.policy.get_action(self._current_observation, self._current_goal)
        next_observation, reward, terminal, info = self.env.step(action)
        self._path_length += 1
        self._path_return += reward
        self._total_samples += 1
        if self._path_length > self.longest_path:
            self.longest_path = self._path_length
            self.latest_state = self._current_observation
	
            
        #what if while playing the goal (which is a future state) is reached? that would make the 
        #transaction from that point on tend to go back to that specific goal istead of going forward
        #should we then sample a new goal once the goal is reached?
	if self.her_type is 0:
	    #normal reward for standard experience, goal based reard for the hindsight replay
	if self.her_type is 1:
	    #complete goal based
	    distance = np.linalg.norm(next_observation_ , goal_)
            reward = 0.5 if  distance <= self.threshold else -0.5
            if self.reset_on_reached and  distance <= self.threshold:
            	if self.goal_sample_size is 0:
                    self._current_goal = self.pool.prioritized_batch(1)
                else:
                    self._current_goal = self.latest_state
	if self.her_type is 2:
	    #proportional to reward
	    distance = np.linalg.norm(next_observation_ , goal_)
            reward += reward*0.1 if  distance <= self.threshold else -0.1*reward
            if self.reset_on_reached and  distance <= self.threshold:
            	if self.goal_sample_size is 0:
                    self._current_goal, _, _ = self.pool.prioritized_batch(1)
                else:
                    self._current_goal = self.latest_state
	if self.her_type is 3:
	    #proportional to distance from goal?
            reward_ -= 0.1* np.linalg.norm(next_observation_ , goal_) 
        self.pool.add_sample(
            observation=self._current_observation,
            action=action,
            reward=reward,
            terminal=terminal,
            next_observation=next_observation,
            goal=self._current_goal)
            
        self.ep_experience.add(
            self._current_observation, 
            action, 
            reward, 
            next_observation,
            terminal,  
            self._current_goal)    
            
        if total_step % 100 == 0 or terminal or self._path_length >= self._max_path_length:
            for t in range(len(self.ep_experience.memory)):
                for _ in range(self.K):
                    future = np.random.randint(t, len(self.ep_experience.memory))
                    goal_ = self.ep_experience.memory[future][3] # next_observation of future
                    observation_ = self.ep_experience.memory[t][0]
                    action_ = self.ep_experience.memory[t][1]
                    next_observation_ = self.ep_experience.memory[t][3]
                    terminal_ = self.ep_experience.memory[t][4]
	            if self.her_type is 0:
	                #normal reward for standard experience, goal based reard for the hindsight replay
                        reward_ = 0.5 if np.linalg.norm(next_observation_ , goal_) <= self.threshold else -0.5
	            if self.her_type is 1:
	                #complete goal based
                        reward_ = 0.5 if np.linalg.norm(next_observation_ , goal_) <= self.threshold else -0.5
	            if self.her_type is 2:
	                #proportional to reward
                        reward_ += reward_*0.1 if np.linalg.norm(next_observation_ , goal_) <= self.threshold else -0.1*reward
	            if self.her_type is 3:
	                #proportional to distance?
                        reward_ -= 0.1* np.linalg.norm(next_observation_ , goal_) 
                    self.pool.add_sample(
                        observation=observation_, 
                        action=action_, 
                        reward=reward_, 
                        next_observation=next_observation_, 
                        terminal=terminal_, 
                        goal=goal_)
            self.ep_experience.clear()

        if terminal or self._path_length >= self._max_path_length:
            self.policy.reset()
            self._current_observation = self.env.reset()
            
            if self.goal_sample_size is 0:
                self._current_goal, _, _ = self.pool.prioritized_batch(1)
            else:
                self._current_goal = self.latest_state #np.random.uniform(low=-10.0, high=10.0, size=len(self._current_observation))
            
            self._path_length = 0
            self._max_path_return = max(self._max_path_return,
                                        self._path_return)
            self._last_path_return = self._path_return

            self._path_return = 0
            self._n_episodes += 1

        else:
            self._current_observation = next_observation
            
	
    def update(self, idx, errors):
        #print(errors)
        priorities = (np.abs(np.abs(errors)) + 1e-6) ** self.alpha
        for i in range(len(idx)):
            self.pool.update(idx[i], priorities[i])   
       
    def prioritized_batch(self, beta):
        data, idx, priorities = self.pool.prioritized_batch(self._batch_size)
        probs = priorities / self.pool.total()
        weights = (self.pool._size * probs) ** -beta
        weights /= np.max(weights)
        return data, idx, weights         


    def log_diagnostics(self):
        super(PrioritizedSampler, self).log_diagnostics()
        logger.record_tabular('max-path-return', self._max_path_return)
        logger.record_tabular('last-path-return', self._last_path_return)
        logger.record_tabular('episodes', self._n_episodes)
        logger.record_tabular('total-samples', self._total_samples)


class DummySampler(Sampler):
    def __init__(self, batch_size, max_path_length):
        super(DummySampler, self).__init__(
            max_path_length=max_path_length,
            min_pool_size=0,
            batch_size=batch_size)

    def sample(self):
        pass
