import abc
import gtimer as gt
import tensorflow as tf
import time

import numpy as np

import joblib

from rllab.misc import logger
from rllab.algos.base import Algorithm

from sac.core.serializable import deep_clone
from sac.misc import tf_utils
from sac.misc.sampler import rollouts
from rllab.misc.tensor_utils import flatten_tensors, unflatten_tensors

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)

class RLAlgorithm(Algorithm):
    """Abstract RLAlgorithm.

    Implements the _train and _evaluate methods to be used
    by classes inheriting from RLAlgorithm.
    """

    def __init__(
            self,
            sampler,
            worker_hosts,
            ps_hosts,
            job_name,
            task_index,
            n_epochs=1000,
            n_train_repeat=1,
            epoch_length=1000,
            eval_n_episodes=10,
            eval_deterministic=True,
            eval_render=False,
            control_interval=1
    ):
        """
        Args:
            n_epochs (`int`): Number of epochs to run the training for.
            n_train_repeat (`int`): Number of times to repeat the training
                for single time step.
            epoch_length (`int`): Epoch length.
            eval_n_episodes (`int`): Number of rollouts to evaluate.
            eval_deterministic (`int`): Whether or not to run the policy in
                deterministic mode when evaluating policy.
            eval_render (`int`): Whether or not to render the evaluation
                environment.
        """
        self.sampler = sampler
        self.worker_hosts = worker_hosts
        self.ps_hosts = ps_hosts
        self.job_name = job_name
        self.task_index = task_index

        self._n_epochs = n_epochs
        self._n_train_repeat = n_train_repeat
        self._epoch_length = epoch_length
        self._control_interval = control_interval

        self._eval_n_episodes = eval_n_episodes
        self._eval_deterministic = eval_deterministic
        self._eval_render = eval_render
        
        #config = tf.ConfigProto().gpu_options.per_process_gpu_memory_fraction = 0.4
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=0.08, allow_growth=True)
        sess_cfg = tf.ConfigProto(
            intra_op_parallelism_threads=8,
            inter_op_parallelism_threads=8,
            #device_filters=['/job:ps', '/job:worker/task:%d' % args.task_index],
            gpu_options=gpu_options)
        self._sess = tf_utils.get_default_session()
        #self._sess = 

        self._env = None
        self._policy = None
        self._pool = None
        #new stuff
        self.beta = 0.4
        self.beta_stop = 1
        self.beta_incr = (self.beta_stop - self.beta) / self._n_epochs + 1
        self._current_observation = None
        self.global_step = tf.Variable(0, trainable=False)
        self._current_observation = None
        self._current_epoch = 0
        self.last_global_step = 0
        self.repeated_step = 0
        self.logdir = 'sac_' + time.strftime('%Y%m%d_%H%M') + '/'
        self.summary_writer = None
        self.summaries = []

    def _train(self, env, policy, pool):
        """Perform RL training.

        Args:
            env (`rllab.Env`): Environment used for training
            policy (`Policy`): Policy used for training
            pool (`PoolBase`): Sample pool to add samples to
        """

        self._init_training(env, policy, pool)
        self.sampler.initialize(env, policy, pool)

    
        # Comma-separated lists of hosts passed in from the command-line
        ps_hosts = self.ps_hosts.split(',')
        worker_hosts = self.worker_hosts.split(',')
        logdir = '/tmp/train_logs/sac_distrib_' + time.strftime('%Y%m%d_%H%M') + '/'
        # Cluster specification
        cluster = tf.train.ClusterSpec({'ps': ps_hosts, 'worker': worker_hosts})
        # Session specification
        # As per https://github.com/tensorflow/tensorflow/issues/12381 the sess
        # config needs to be passed to tf.train.Server to limit GPU usage for each
        # worker, as opposed to the normal usage of passing to tf.Session()
        # I might also try autogrowth or something like that
        gpu_options = tf.GPUOptions(allow_growth=True)
#            per_process_gpu_memory_fraction=0.08)#, allow_growth=True)
        sess_cfg = tf.ConfigProto(
            intra_op_parallelism_threads=8,
            inter_op_parallelism_threads=8,
            #device_filters=['/job:ps', '/job:worker/task:%d' % args.task_index],
            gpu_options=gpu_options)
        # Server specification
        server = tf.train.Server(
            cluster, job_name=self.job_name, task_index=self.task_index, config=sess_cfg)
            
        a_dim = env.action_space.shape[0]
        s_dim = env.observation_space.shape[0]
        a_mult = env.action_space.high
        total_num_steps = self._n_epochs * self._epoch_length + 1
        with tf.device('/job:ps/task:0'):
        
            q_p = tf.placeholder(tf.int64, [None], 'queue_pri')
            q_s0 = tf.placeholder(tf.float32, [None, s_dim], 'queue_s0')
            q_a0 = tf.placeholder(tf.float32, [None, a_dim], 'queue_a')
            q_r0 = tf.placeholder(tf.float32, [None, 1], 'queue_r')
            q_s1 = tf.placeholder(tf.float32, [None, s_dim], 'queue_s1')
            q_done = tf.placeholder(tf.float32, [None, 1], 'queue_done')
            q_loss = tf.placeholder(tf.float32, [None, 1], 'queue_loss')

            shared_queue = tf.PriorityQueue(
                capacity=10000,
                types=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32),
                shapes=([s_dim], [a_dim], [1], [s_dim], [1], [1]),
                names=('weight', 'observation', 'action', 'reward', 'next_observation', 'terminal', 'loss'),
                shared_name='shared_queue',
                name='shared_queue')

            enqueue_op = shared_queue.enqueue_many({
                'weight': q_p, 'observation': q_s0, 'action': q_a0, 'reward': q_r0,
                'next_observation': q_s1, 'terminal': q_done, 'loss': q_loss})

            
        # Here, the next step depends on whether this is a PS, Actor, or Learner
        if self.job_name == 'ps':
            server.join()

        elif self.job_name == 'worker' and self.task_index > 0: # Actors start environments and collect data
            with tf.device(tf.train.replica_device_setter(
                ps_device='/job:ps/task:0',
                worker_device='/job:worker/task:%d' % self.task_index)):
                
                # Boilerplate stuff
                hooks=[tf.train.StopAtStepHook(last_step=total_num_steps)]                
                # Model
                self.global_step = tf.train.get_or_create_global_step()
                increment_global_step_op = tf.assign(
                    self.global_step, self.global_step + 1)
                merge_op = tf.summary.merge_all()
                
                with tf.train.MonitoredTrainingSession(
                    save_summaries_secs=None, save_summaries_steps=None, # Manually do summaries
                    master=server.target, is_chief=False,
                    checkpoint_dir=logdir,
                    # put none for now
                    hooks=hooks, config=sess_cfg) as mon_sess:
                    
                    self._sess = mon_sess
                    self._policy.set_sess(mon_sess)
              #      self._env.set_low_policy_sess(mon_sess)
              #      self._eval_env.set_low_policy_sess(mon_sess)
                    
                    gt.rename_root('RLAlgorithm')
                    gt.reset()
                    gt.set_def_unique(False)
                    
                    while not mon_sess.should_stop():
                        target_par = mon_sess.run(self._qf_target_params)
                        #print("WORKER "+ str(self.task_index) +"   "+str(target_par))
                        
                        global_step = mon_sess.run(self.global_step)
                        if global_step >0: 
                            if self.repeated_step>0:
                                while self.last_global_step == global_step:
                                    #print("WORKER "+ str(self.task_index) +" waiting at step " + str(global_step))
                                    time.sleep(0.1)
                                    global_step = mon_sess.run(self.global_step)
                                self.repeated_step=0
                            if self.last_global_step == global_step:
                                self.repeated_step+=1    
                        self.last_global_step = global_step
                        """
                        if self._current_observation is None:
                            self._current_observation = env.reset()

                        action, _ = self._policy.get_action(self._current_observation)
                        next_observation, reward, terminal, info = env.step(action)
                        transition=dict(
                            observation= np.expand_dims(self._current_observation, axis=0),
                            action= np.expand_dims(action, axis=0),
                            reward= np.expand_dims(reward, axis=0),
                            terminal= np.expand_dims(terminal, axis=0),
                            next_observation= np.expand_dims(next_observation, axis=0))

                        if terminal: #or self._path_length >= self._max_path_length:
                            self._policy.reset()
                            self._current_observation = env.reset()
                        """
                        transition = self.sampler.sample()
                        
                        loss = self._get_loss(
                            iteration=global_step,#t + epoch * self._epoch_length,# probably I should actually use global_step
                            batch=transition, 
                            weights=transition['terminal'])
                        transition['loss'] = loss
                        
                        for key in transition:
                            if type(transition[key]) is not np.ndarray:
                                transition[key] = np.asarray(transition[key]) 
                            if len(transition[key].shape)==1 and key!="terminal":
                                transition[key] = np.expand_dims(transition[key], axis=0)
                        feed_dict = {
                            'queue_pri:0': transition['terminal'],
                            'queue_s0:0': transition['observation'],
                            'queue_a:0': transition['action'],
                            'queue_r:0':transition['reward'], 
                            'queue_s1:0': transition['next_observation'],
                            'queue_done:0': np.expand_dims(transition['terminal'], axis=0),
                            'queue_loss:0': transition['loss']}
                            
                        #mon_sess.run(increment_global_step_op)
                        mon_sess.run(enqueue_op, feed_dict=feed_dict)   
                       
                        #if global_step >0:
                        #    print ("WORKER "+ str(self.task_index))   
                    self.sampler.terminate()
                        
        elif self.job_name == 'worker' and self.task_index == 0: # Learner continually updates policy, value fns
            with tf.device(tf.train.replica_device_setter(
                ps_device='/job:ps/task:0',
                worker_device='/job:worker/task:%d' % self.task_index)):
                
                # Boilerplate stuff
                hooks=[tf.train.StopAtStepHook(last_step=total_num_steps)]  
                # Get data from shared priority queue
                dequeue_op = shared_queue.dequeue_many(self.sampler._batch_size)  
                qsize_op = shared_queue.size()            
                # Model
                self.global_step = tf.train.get_or_create_global_step()
                increment_global_step_op = tf.assign(
                    self.global_step, self.global_step + 1)
                merge_op = tf.summary.merge_all()
                
                if not os.path.exists('summaries'):
                    os.mkdir('summaries')
                if not os.path.exists(os.path.join('summaries',self.logdir)):
                    os.mkdir(os.path.join('summaries',self.logdir))
                self.summary_writer = tf.summary.FileWriter(os.path.join('summaries',self.logdir), tf.get_default_graph())
                
                with tf.train.MonitoredTrainingSession(
                    save_summaries_secs=None, save_summaries_steps=None, # Manually do summaries
                    master=server.target, is_chief=(self.task_index == 0),
                    checkpoint_dir=logdir, save_checkpoint_secs=600,
                    hooks=hooks, config=sess_cfg) as mon_sess:

                    self._sess = mon_sess
                    self._policy.set_sess(mon_sess)
                #    self._env.set_low_policy_sess(mon_sess)
                #    self._eval_env.set_low_policy_sess(mon_sess)
                    
                    print('Sleeping until samples become available...')
                    qsize = 0
                    while qsize < 10000:
                        qsize = mon_sess.run(qsize_op)
                        time.sleep(0.1)
                    print('Samples in queue:', qsize)

                    gt.rename_root('RLAlgorithm')
                    gt.reset()
                    gt.set_def_unique(False)
                    
                    while not mon_sess.should_stop():
                        
                        learner_loop = gt.timed_loop(save_itrs=True)
                        global_step = mon_sess.run(self.global_step)
                        epoch = int(global_step / self._epoch_length)
                        logger.push_prefix('Epoch #%d | ' % epoch)
                        for num_updates in range(total_num_steps):
                        #while global_step < total_num_steps:
                            global_step = mon_sess.run(self.global_step)
                            #print("LEARNER")
                            epoch = int(global_step / self._epoch_length)
                            #print("epoch " +str(epoch))
                            #Here It would be probably better to see if current epoch > last epoch
                            
                             # I shouldn't need this in the learner
                            #  transition = self.sampler.sample()
                            # if not self.sampler.batch_ready():
                            #    continue
                            # gt.stamp('sample')
                            qsize = mon_sess.run(qsize_op)
                            #print("qsize "+ str(qsize))
                            if qsize > 5000 or self.sampler._total_samples < self.sampler._batch_size:
                                data = mon_sess.run(dequeue_op)
                                #print("dequeued ")
                                # I am not sure weather this is the best idea, and weather I can even do it, so not for now
                                #losses = np.full(data['loss'].shape, np.median(memory.loss_mem))
                                for i in range(len(data['terminal'])):
                                    transition = dict(
                                        observation= data['observation'][i],
                                        action= data['action'][i],
                                        reward= data['reward'][i],
                                        terminal= data['terminal'][i],
                                        next_observation= data['next_observation'][i],
                                        loss= data['loss'][i])
                                    self.sampler.add_sample(transition)
                            #print("batching ")
                            
                            
                            batch, idx, weights = self.sampler.prioritized_batch(self.beta)
                            #batch, idx, weights = self.sampler.random_batch()
                            loss = self._do_training(
                                iteration=global_step,#t + epoch * self._epoch_length,
                                batch=batch,
                                weights=weights)
                            batch['loss'] = loss
                            self.sampler.update(idx, loss) 
                            #print("pre "+ str(global_step))
                            mon_sess.run(increment_global_step_op)
                            global_step = mon_sess.run(self.global_step)
                            #print("post "+ str(global_step))
                            gt.stamp('train')

                            
                            if epoch > self._current_epoch:
                                #print( "EVAL")
                                # Incr beta
                                if self.beta <= self.beta_stop:
                                    self.beta += self.beta_incr   
                                self._evaluate(epoch)
                                
                                params = self.get_snapshot(epoch)
                                logger.save_itr_params(epoch, params)
                                times_itrs = gt.get_times().stamps.itrs

                               # eval_time = times_itrs['eval'][-1] if epoch > 1 else 0
                                total_time = gt.get_times().total
                               # logger.record_tabular('time-train', times_itrs['train'][-1])
                                #logger.record_tabular('time-eval', eval_time)
                               # logger.record_tabular('time-sample', times_itrs['sample'][-1])
                                logger.record_tabular('time-total', total_time)
                                logger.record_tabular('epoch', epoch)
                                logger.record_tabular('num-updates', num_updates)
                                logger.record_tabular('num-epoch-updates', int(num_updates/self._epoch_length))

                                self.sampler.log_diagnostics()

                                logger.dump_tabular(with_prefix=False)

                                gt.stamp('eval')

                                logger.pop_prefix()
                                
                                logger.push_prefix('Epoch #%d | ' % epoch)
                                learner_loop.next()
                                self._current_epoch = epoch
                                
                                                            
                            if epoch %100 == 0:
                                if not os.path.exists('sac/data/new'):
                                    os.mkdir('sac/data/new')
                                file_name = 'sac/data/new/itr_%d.pkl' % epoch
                                p_params = self._policy.get_params()
                                v_params = self._vf.get_params()
                                q_params = self._qf.get_params()
                                p_param_values, v_param_values, q_param_values = mon_sess.run( (p_params, v_params, q_params) )
                                mon_sess.graph._unsafe_unfinalize()
                                self._policy.set_param_values(flatten_tensors(p_param_values))
                                self._vf.set_param_values(flatten_tensors(v_param_values))
                                self._qf.set_param_values(flatten_tensors(q_param_values))
                                snapshot = {
                                    'epoch': epoch,
                                    'policy': self._policy,
                                    'qf': self._qf,
                                    'vf': self._vf,
                                    'env': self._env,
                                }
                                joblib.dump(snapshot, file_name, compress=3)
                            
                        print(" I-M OUT ")
                        #self.sampler.terminate()


    def add_to_summaries(self, value, name):
        value_summary = tf.Summary.Value(tag=name, simple_value=value)
        self.summaries.append(value_summary)

    def _evaluate(self, epoch):
        """Perform evaluation for the current policy.

        :param epoch: The epoch number.
        :return: None
        """

        if self._eval_n_episodes < 1:
            return

        #N.B. _max_path_lenght must probably be moved from sampler to base or something like that
        with self._policy.deterministic(self._eval_deterministic):
            paths = rollouts(self._eval_env, self._policy,
                             self.sampler._max_path_length, self._eval_n_episodes,
                            )

        total_returns = [path['rewards'].sum() for path in paths]
        episode_lengths = [len(p['rewards']) for p in paths]

        logger.record_tabular('return-average', np.mean(total_returns))
        logger.record_tabular('return-min', np.min(total_returns))
        logger.record_tabular('return-max', np.max(total_returns))
        logger.record_tabular('return-std', np.std(total_returns))
        logger.record_tabular('episode-length-avg', np.mean(episode_lengths))
        logger.record_tabular('episode-length-min', np.min(episode_lengths))
        logger.record_tabular('episode-length-max', np.max(episode_lengths))
        logger.record_tabular('episode-length-std', np.std(episode_lengths))

        self._eval_env.log_diagnostics(paths)
        if self._eval_render:
            self._eval_env.render(paths)

        iteration = epoch*self._epoch_length
        batch, idx, weights = self.sampler.prioritized_batch(self.beta)
        self.log_diagnostics(iteration, batch, weights)
        
        #tensorboard
        self.add_to_summaries(np.mean(total_returns), "return_average")
        c = tf.Summary(value= self.summaries)
        self.summary_writer.add_summary(c, epoch)
        self.summaries = []


    @abc.abstractmethod
    def log_diagnostics(self, iteration, batch):
        raise NotImplementedError

    @abc.abstractmethod
    def get_snapshot(self, epoch):
        raise NotImplementedError

    @abc.abstractmethod
    def _do_training(self, iteration, batch):
        raise NotImplementedError

    @abc.abstractmethod
    def _init_training(self, env, policy, pool):
        """Method to be called at the start of training.

        :param env: Environment instance.
        :param policy: Policy instance.
        :return: None
        """

        self._env = env
        if self._eval_n_episodes > 0:
            with tf.variable_scope("low_level_policy", reuse=False):
                self._eval_env = deep_clone(env)
        self._policy = policy
        self._pool = pool

    @property
    def policy(self):
        return self._policy

    @property
    def env(self):
        return self._env

    @property
    def pool(self):
        return self._pool
