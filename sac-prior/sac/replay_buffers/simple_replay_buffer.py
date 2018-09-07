import numpy as np

from rllab.core.serializable import Serializable

from .replay_buffer import ReplayBuffer


class SimpleReplayBuffer(ReplayBuffer, Serializable):
    def __init__(self, env_spec, max_replay_buffer_size):
        super(SimpleReplayBuffer, self).__init__()
        Serializable.quick_init(self, locals())

        max_replay_buffer_size = int(max_replay_buffer_size)

        self._env_spec = env_spec
        self._observation_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim
        self._max_buffer_size = max_replay_buffer_size
        self._observations = np.zeros((max_replay_buffer_size,
                                       self._observation_dim))
        # It's a bit memory inefficient to save the observations twice,
        # but it makes the code *much* easier since you no longer have to
        # worry about termination conditions.
        self._next_obs = np.zeros((max_replay_buffer_size,
                                   self._observation_dim))
        self._actions = np.zeros((max_replay_buffer_size, self._action_dim))
        self._rewards = np.zeros(max_replay_buffer_size)
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.zeros(max_replay_buffer_size, dtype='uint8')
        self._top = 0
        self._size = 0

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        self._observations[self._top] = observation
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._next_obs[self._top] = next_observation

        self._advance()

    def terminate_episode(self):
        pass

    def _advance(self):
        self._top = (self._top + 1) % self._max_buffer_size
        if self._size < self._max_buffer_size:
            self._size += 1

    def random_batch(self, batch_size):
        indices = np.random.randint(0, self._size, batch_size)
        return dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
        )

    @property
    def size(self):
        return self._size
    
    def __getstate__(self):
        d = super(SimpleReplayBuffer, self).__getstate__()
        d.update(dict(
            o=self._observations.tobytes(),
            a=self._actions.tobytes(),
            r=self._rewards.tobytes(),
            t=self._terminals.tobytes(),
            no=self._next_obs.tobytes(),
            top=self._top,
            size=self._size,
        ))
        return d

    def __setstate__(self, d):
        super(SimpleReplayBuffer, self).__setstate__(d)
        self._observations = np.fromstring(d['o']).reshape(
            self._max_buffer_size, -1
        )
        self._next_obs = np.fromstring(d['no']).reshape(
            self._max_buffer_size, -1
        )
        self._actions = np.fromstring(d['a']).reshape(self._max_buffer_size, -1)
        self._rewards = np.fromstring(d['r']).reshape(self._max_buffer_size)
        self._terminals = np.fromstring(d['t'], dtype=np.uint8)
        self._top = d['top']
        self._size = d['size']




class PrioritizedReplayBuffer(ReplayBuffer, Serializable):
    def __init__(self, env_spec, max_replay_buffer_size):
        super(PrioritizedReplayBuffer, self).__init__()
        Serializable.quick_init(self, locals())

        self.max_replay_buffer_size = int(max_replay_buffer_size)

        self._env_spec = env_spec
        self._observation_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim
        self._max_buffer_size = max_replay_buffer_size
        self._observations = np.zeros((self.max_replay_buffer_size,
                                       self._observation_dim))
        # It's a bit memory inefficient to save the observations twice,
        # but it makes the code *much* easier since you no longer have to
        # worry about termination conditions.
        self._next_obs = np.zeros((self.max_replay_buffer_size,
                                   self._observation_dim))
        self._actions = np.zeros((self.max_replay_buffer_size, self._action_dim))
        self._rewards = np.zeros(self.max_replay_buffer_size)
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.zeros(self.max_replay_buffer_size, dtype='uint8')
        self._top = 0
        self._size = 0
        #new stuff from SumTree
     #   self.capacity = capacity # this should be equivalent to max_replay_buffer_size
        self.sum_tree = np.zeros(2 * self.max_replay_buffer_size - 1)
        self.max_tree = np.ones(2 * self.max_replay_buffer_size - 1)
     #   self.write = 0	# this should be equivalent to _top
     #   self.n_entries = 0 # this should be equivalent to _size
    
    
    def _propagate(self, idx, value):
        parent = (idx - 1) // 2
        left = 2 * parent + 1
        right = left + 1

        self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
        self.max_tree[parent] = max(self.max_tree[left], self.max_tree[right])

        if parent != 0:
            self._propagate(parent, value)

    def _retrieve(self, idx, value):
        left = 2 * idx + 1

        if left >= 2 * self.max_replay_buffer_size - 1:
            return idx

        if value <= self.sum_tree[left]:
            return self._retrieve(left, value)
        else:
            right = left + 1
            return self._retrieve(right, value - self.sum_tree[left])

    def total(self):
        return self.sum_tree[0]

    def max(self):
        return self.max_tree[0]
        
    # Seems like data is the combo of all the obs,act,rew,etc. 
    # those values are put in a linear array, and value is the thing
    # actually put in the sumtree, and later used a sort of index to
    # retrieve the info in data    
    def add(self, value, data):
        self.data[self.write] = data				#this part is the equivalent of addsample
        self.update(self.write + self.capacity - 1, value)	#this part is new, as it the calls _propagate
        self.write = (self.write + 1) % self.capacity		#this and the following lines are equivalents to _advance
        self.n_entries = min(self.n_entries + 1, self.capacity)
    
    def update(self, idx, value):
        self.sum_tree[idx] = self.max_tree[idx] = value
        self._propagate(idx, value)
     
    def get(self, value):
        idx = self._retrieve(0, value)
        data_idx = idx - self.max_replay_buffer_size + 1
     #   data = dict(
     #       observations=self._observations[data_idx],
     #       actions=self._actions[data_idx],
     #       rewards=self._rewards[data_idx],
     #       terminals=self._terminals[data_idx],
     #       next_observations=self._next_obs[data_idx],
     #   )
        
        return data_idx, idx, self.sum_tree[idx]

    def prioritized_batch(self, batch_size):
        batch_idx = [None] * batch_size
        batch_priorities = [None] * batch_size
        batch = [None] * batch_size
        data_idx = [None] * batch_size
        segment = self.total() / batch_size

        a = [segment*i for i in range(batch_size)]
        b = [segment * (i+1) for i in range(batch_size)]
        s = np.random.uniform(a, b)

        for i in range(batch_size):
            (data_idx[i], batch_idx[i], batch_priorities[i]) = self.get(s[i])
        
        batch = dict(
            observations=self._observations[data_idx],
            actions=self._actions[data_idx],
            rewards=self._rewards[data_idx],
            terminals=self._terminals[data_idx],
            next_observations=self._next_obs[data_idx],
        )

        return batch, batch_idx, batch_priorities   
        
    def add_sample(self, observation, action, reward, terminal,
                   next_observation, value=None, **kwargs):
        self._observations[self._top] = observation
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._next_obs[self._top] = next_observation

        if value==None:
            value= self.max()
        self.update(self._top + self.max_replay_buffer_size - 1, value)	#this part is new, as it the calls _propagate
        self._top = (self._top + 1) % self.max_replay_buffer_size		#this and the following lines are equivalents to _advance
        self._size = min(self._size + 1, self.max_replay_buffer_size)

    #    self._advance()

    def terminate_episode(self):
        pass


    def _advance(self):
        self._top = (self._top + 1) % self._max_buffer_size
        if self._size < self._max_buffer_size:
            self._size += 1

    def random_batch(self, batch_size):
        indices = np.random.randint(0, self._size, batch_size)
        return dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
        )

    @property
    def size(self):
        return self._size
    #SimpleReplayBuffer works even without this two methods. I'll keep them 
    #for now, eventually I'll delete them if needed
    def __getstate__(self):
        d = super(PrioritizedReplayBuffer, self).__getstate__()
        d.update(dict(
            o=self._observations.tobytes(),
            a=self._actions.tobytes(),
            r=self._rewards.tobytes(),
            t=self._terminals.tobytes(),
            no=self._next_obs.tobytes(),
            top=self._top,
            size=self._size,
        ))
        return d

    def __setstate__(self, d):
        super(PrioritizedReplayBuffer, self).__setstate__(d)
        self._observations = np.fromstring(d['o']).reshape(
            self._max_buffer_size, -1
        )
        self._next_obs = np.fromstring(d['no']).reshape(
            self._max_buffer_size, -1
        )
        self._actions = np.fromstring(d['a']).reshape(self._max_buffer_size, -1)
        self._rewards = np.fromstring(d['r']).reshape(self._max_buffer_size)
        self._terminals = np.fromstring(d['t'], dtype=np.uint8)
        self._top = d['top']
        self._size = d['size']
        
    def __repr__(self):
        s = ""
        last_line = " ".join([str(self.sum_tree[i+self.capacity-1]).center(5) for i in range(len(self.data))])
        for i in range(4):
            line = self.sum_tree[2**i-1:2**i-1+2**i]
            s += (" "*(len(last_line)//2**(i+1))).join([str(line[i]).center(5) for i in range(len(line))]).center(len(last_line)) + "\n"
        s += last_line
        return s
