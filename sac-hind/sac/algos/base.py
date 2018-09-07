import abc
import gtimer as gt

import numpy as np
import time
import os
import tensorflow as tf

from rllab.misc import logger
from rllab.algos.base import Algorithm

from sac.core.serializable import deep_clone
from sac.misc import tf_utils
from sac.misc.sampler import rollouts


class RLAlgorithm(Algorithm):
    """Abstract RLAlgorithm.

    Implements the _train and _evaluate methods to be used
    by classes inheriting from RLAlgorithm.
    """

    def __init__(
            self,
            sampler,
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

        self._n_epochs = n_epochs
        self._n_train_repeat = n_train_repeat
        self._epoch_length = epoch_length
        self._control_interval = control_interval

        self._eval_n_episodes = eval_n_episodes
        self._eval_deterministic = eval_deterministic
        self._eval_render = eval_render

        self._sess = tf_utils.get_default_session()

        self._env = None
        self._policy = None
        self._pool = None
        #new stuff
        self.beta = 0.4
        self.beta_stop = 1
        self.beta_incr = (self.beta_stop - self.beta) / self._n_epochs + 1
        self.logdir = 'sac_' + time.strftime('%Y%m%d_%H%M') + '/'
        self.summary_writer = None
        self.summaries = []

    def _train(self, env, env_copy, policy, pool):
        """Perform RL training.

        Args:
            env (`rllab.Env`): Environment used for training
            policy (`Policy`): Policy used for training
            pool (`PoolBase`): Sample pool to add samples to
        """

        self._init_training(env, policy, pool)
        self.sampler.initialize(env, env_copy, policy, pool)
        
        if not os.path.exists('summaries'):
            os.mkdir('summaries')
        if not os.path.exists(os.path.join('summaries',self.logdir)):
            os.mkdir(os.path.join('summaries',self.logdir))
        self.summary_writer = tf.summary.FileWriter(os.path.join('summaries',self.logdir), tf.get_default_graph())

        with self._sess.as_default():
            gt.rename_root('RLAlgorithm')
            gt.reset()
            gt.set_def_unique(False)

            for epoch in gt.timed_for(range(self._n_epochs + 1),
                                      save_itrs=True):
                logger.push_prefix('Epoch #%d | ' % epoch)

                for t in range(self._epoch_length):
                    # TODO.codeconsolidation: Add control interval to sampler
                    self.sampler.sample(t + epoch * self._epoch_length)
                    if not self.sampler.batch_ready():
                        continue
                    gt.stamp('sample')

                    for i in range(self._n_train_repeat):
                        batch, idx, weights =self.sampler.prioritized_batch(self.beta)
                        loss = self._do_training(
                            iteration=t + epoch * self._epoch_length,
                            batch=batch,
                            weights=weights)
                        self.sampler.update(idx, loss)
                            
                    gt.stamp('train')

                
                # Incr beta
                if self.beta <= self.beta_stop:
                    self.beta += self.beta_incr   
                self._evaluate(epoch)

                params = self.get_snapshot(epoch)
                logger.save_itr_params(epoch, params)
                times_itrs = gt.get_times().stamps.itrs

                eval_time = times_itrs['eval'][-1] if epoch > 1 else 0
                total_time = gt.get_times().total
                logger.record_tabular('time-train', times_itrs['train'][-1])
                logger.record_tabular('time-eval', eval_time)
                logger.record_tabular('time-sample', times_itrs['sample'][-1])
                logger.record_tabular('time-total', total_time)
                logger.record_tabular('epoch', epoch)

                self.sampler.log_diagnostics()

                logger.dump_tabular(with_prefix=False)
                logger.pop_prefix()

                gt.stamp('eval')

            self.sampler.terminate()

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
            # TODO: This is horrible. Don't do this. Get rid of this.
            import tensorflow as tf
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
