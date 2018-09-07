from numbers import Number

import numpy as np
import tensorflow as tf

from rllab.core.serializable import Serializable
from rllab.misc import logger
from rllab.misc.overrides import overrides

from .base import RLAlgorithm

EPS = 1E-6


class SAC(RLAlgorithm, Serializable):
    """Soft Actor-Critic (SAC)

    Example:
    ```python

    env = normalize(SwimmerEnv())

    pool = SimpleReplayPool(env_spec=env.spec, max_pool_size=1E6)

    base_kwargs = dict(
        min_pool_size=1000,
        epoch_length=1000,
        n_epochs=1000,
        batch_size=64,
        scale_reward=1,
        n_train_repeat=1,
        eval_render=False,
        eval_n_episodes=1,
        eval_deterministic=True,
    )

    M = 100
    qf = NNQFunction(env_spec=env.spec, hidden_layer_sizes=(M, M))

    vf = NNVFunction(env_spec=env.spec, hidden_layer_sizes=(M, M))

    policy = GMMPolicy(
        env_spec=env.spec,
        K=2,
        hidden_layers=(M, M),
        qf=qf,
        reg=0.001
    )

    algorithm = SAC(
        base_kwargs=base_kwargs,
        env=env,
        policy=policy,
        pool=pool,
        qf=qf,
        vf=vf,

        lr=3E-4,
        discount=0.99,
        tau=0.01,

        save_full_state=False
    )

    algorithm.train()
    ```

    References
    ----------
    [1] Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine, "Soft
        Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning
        with a Stochastic Actor," Deep Learning Symposium, NIPS 2017.
    """

    def __init__(
            self,
            base_kwargs,

            env,
            policy,
            qf,
            vf,
            pool,
            plotter=None,

            lr=3e-3,
            scale_reward=1,
            discount=0.99,
            tau=0.01,
            target_update_interval=1,
            action_prior='uniform',

            save_full_state=False,
    ):
        """
        Args:
            base_kwargs (dict): dictionary of base arguments that are directly
                passed to the base `RLAlgorithm` constructor.

            env (`rllab.Env`): rllab environment object.
            policy: (`rllab.NNPolicy`): A policy function approximator.
            qf (`ValueFunction`): Q-function approximator.
            vf (`ValueFunction`): Soft value function approximator.
            pool (`PoolBase`): Replay buffer to add gathered samples to.
            plotter (`QFPolicyPlotter`): Plotter instance to be used for
                visualizing Q-function during training.

            lr (`float`): Learning rate used for the function approximators.
            scale_reward (`float`): Scaling factor for raw reward.
            discount (`float`): Discount factor for Q-function updates.
            tau (`float`): Soft value function target update weight.

            save_full_state (`bool`): If True, save the full class in the
                snapshot. See `self.get_snapshot` for more information.
        """

        Serializable.quick_init(self, locals())
        super(SAC, self).__init__(**base_kwargs)

        self._env = env
        self._policy = policy
        self._qf = qf
        self._vf = vf
        self._pool = pool
        self._plotter = plotter

        self._policy_lr = lr
        self._qf_lr = lr
        self._vf_lr = lr
        self._scale_reward = scale_reward
        self._discount = discount
        self._tau = tau
        self._target_update_interval = target_update_interval
        self._action_prior = action_prior

        self._save_full_state = save_full_state

        self._Da = self._env.action_space.flat_dim
        self._Do = self._env.observation_space.flat_dim
        
        # From D4PG
        self.MIN_Q = -100
        self.MAX_Q = 5000
        self.NB_ATOMS = 51
        self.delta_z = (self.MAX_Q - self.MIN_Q) / (self.NB_ATOMS - 1)
        self.z = tf.range(self.MIN_Q, self.MAX_Q + self.delta_z, self.delta_z)
        # needed for D4PG
        self.batch_size = base_kwargs['sampler']._batch_size
        self.not_done_ph = tf.placeholder(dtype=tf.float32, shape=[None], name='not_done') 
        self.not_done = tf.expand_dims(self.not_done_ph, 1)
        self.next_distrib_qf_t = tf.placeholder(dtype=tf.float32, shape=[None], name='next_distrib_qf_t') 

        self._training_ops = list()

        self._init_placeholders()
        next_actions = self._policy.actions_for(observations=self._next_observations_ph,
                                                   with_log_pis=False)
        self.next_distrib_qf_t = self._qf.get_output_for(self._next_observations_ph, 
                                                   next_actions, reuse=False, keep_distrib=True)
        self._init_actor_update()
        self._init_critic_update()
        self._init_target_ops()

        
        # Initialize all uninitialized variables. This prevents initializing
        # pre-trained policy and qf and vf variables.
        uninit_vars = []
        for var in tf.global_variables():
            try:
                self._sess.run(var)
            except tf.errors.FailedPreconditionError:
                uninit_vars.append(var)
        self._sess.run(tf.variables_initializer(uninit_vars))


    @overrides
    def train(self):
        """Initiate training of the SAC instance."""

        self._train(self._env, self._policy, self._pool)

    def _init_placeholders(self):
        """Create input placeholders for the SAC algorithm.

        Creates `tf.placeholder`s for:
            - observation
            - next observation
            - action
            - reward
            - terminals
        """
        self._iteration_pl = tf.placeholder(
            tf.int64, shape=None, name='iteration')

        self._observations_ph = tf.placeholder(
            tf.float32,
            shape=(None, self._Do),
            name='observation',
        )

        self._next_observations_ph = tf.placeholder(
            tf.float32,
            shape=(None, self._Do),
            name='next_observation',
        )
        self._actions_ph = tf.placeholder(
            tf.float32,
            shape=(None, self._Da),
            name='actions',
        )

        self._rewards_ph = tf.placeholder(
            tf.float32,
            shape=(None, ),
            name='rewards',
        )
        self._rewards = tf.expand_dims(self._rewards_ph, 1)
        self._terminals_ph = tf.placeholder(
            tf.float32,
            shape=(None, ),
            name='terminals',
        )

    @property
    def scale_reward(self):
        if callable(self._scale_reward):
            return self._scale_reward(self._iteration_pl)
        elif isinstance(self._scale_reward, Number):
            return self._scale_reward

        raise ValueError(
            'scale_reward must be either callable or scalar')
                    
    def project_TZ (self, distrib_qf_t):    
        # Extend the support for the whole batch (i.e. with batch_size lines)
        zz = tf.tile(self.z[None], [self.batch_size, 1])

        # Compute the projection of Tz onto the support z
        #I added the stop gradient, before it was all together. Not sure wether this is right
        Tz = tf.stop_gradient(self._rewards * self.scale_reward + self._discount * self.not_done * zz)
        Tz = tf.clip_by_value(Tz,
                              self.MIN_Q, self.MAX_Q - 1e-4)
        bj = (Tz - self.MIN_Q) / self.delta_z
        l = tf.floor(bj)
        u = l + 1
        l_ind, u_ind = tf.to_int32(l), tf.to_int32(u)

        # Initialize the critic loss
        critic_loss = tf.zeros([self.batch_size])

        for j in range(self.NB_ATOMS):
            # Select the value of Q(s_t, a_t) onto the atoms l and u and clip it
            l_index = tf.stack((tf.range(self.batch_size), l_ind[:, j]), axis=1)
            u_index = tf.stack((tf.range(self.batch_size), u_ind[:, j]), axis=1)

            main_Q_distrib_l = tf.gather_nd(distrib_qf_t, l_index)
            main_Q_distrib_u = tf.gather_nd(distrib_qf_t, u_index)

            main_Q_distrib_l = tf.clip_by_value(main_Q_distrib_l, 1e-10, 1.0)
            main_Q_distrib_u = tf.clip_by_value(main_Q_distrib_u, 1e-10, 1.0)

            # loss +=   Q(s_{t+1}, a*) * (u - bj) * log Q[l](s_t, a_t)
            #         + Q(s_{t+1}, a*) * (bj - l) * log Q[u](s_t, a_t)
            critic_loss += (#self.next_distrib_qf_t[:, j] * (
                (u[:, j] - bj[:, j]) * tf.log(main_Q_distrib_l) +
                (bj[:, j] - l[:, j]) * tf.log(main_Q_distrib_u))
        return critic_loss
    
    def _init_critic_update(self):
        """Create minimization operation for critic Q-function.

        Creates a `tf.optimizer.minimize` operation for updating
        critic Q-function with gradient descent, and appends it to
        `self._training_ops` attribute.

        See Equation (10) in [1], for further information of the
        Q-function update rule.
        """
	#get_output_for returns the output of a mlp
      #  self._qf_t = self._qf.get_output_for(
       #     self._observations_ph, self._actions_ph, reuse=True)     # N 
        
        # For now, we keep the distributed one separated
        distrib_qf_t = self._qf.get_output_for(
            self._observations_ph, self._actions_ph, reuse=True, keep_distrib=True)
        
        self._qf_t = self._qf.get_output_for(
            self._observations_ph, self._actions_ph, reuse=True)
            
        # not_done == 1 - batch['terminals']
        # Is left to understand what with Q_distrib_next -> I ended up recalculating it to be sure. should not
        # be trainable since they are targets, I'll probably need to use tf.stop_gradient
        # Calculating this here is probably wrong, as I think it does it every time
        
        critic_loss = self.project_TZ( distrib_qf_t )
        # Here critic loss should be multiplied by next_distrib_qf_t to obtain the cross-entropy
          
        # Take the mean loss on the batch
        critic_loss = tf.negative(critic_loss)
    #    critic_loss = tf.reduce_mean(critic_loss)
        critic_vars = self._qf.get_params_internal()
        reg = 0
        for var in critic_vars:
            if not 'bias' in var.name:
                reg += 1e-6 * tf.nn.l2_loss(var)
    #    critic_loss += reg        
     #   self._td_loss_t = critic_loss
        
        # Gradient descent
   #     critic_trainer = tf.train.AdamOptimizer(self._qf_lr)
   #     self.critic_train_op = critic_trainer.minimize(critic_loss)
         
    #    self._training_ops.append( self.critic_train_op )

        with tf.variable_scope('target'):
            vf_next_target_t = self._vf.get_output_for(self._next_observations_ph, keep_distrib=True)  # N
            self._vf_target_params = self._vf.get_params_internal()
        distrib_vf = self.project_TZ( vf_next_target_t)
        ys = tf.stop_gradient(
            self.scale_reward * self._rewards_ph +
            (1 - self._terminals_ph) * self._discount * distrib_vf#vf_next_target_t
        )  # N
        
        self._td_loss_t = 0.5 * tf.reduce_mean((ys - critic_loss)**2)
        
        qf_train_op = tf.train.AdamOptimizer(self._qf_lr).minimize(
            loss=self._td_loss_t,
            var_list=self._qf.get_params_internal()
        )

        self._training_ops.append(qf_train_op)
        


    def _init_actor_update(self):
        """Create minimization operations for policy and state value functions.

        Creates a `tf.optimizer.minimize` operations for updating
        policy and value functions with gradient descent, and appends them to
        `self._training_ops` attribute.

        In principle, there is no need for a separate state value function
        approximator, since it could be evaluated using the Q-function and
        policy. However, in practice, the separate function approximator
        stabilizes training.

        See Equations (8, 13) in [1], for further information
        of the value function and policy function update rules.
        """
        						                   
        actions, log_pi = self._policy.actions_for(observations=self._observations_ph,
                                                   with_log_pis=True)

        self._vf_t_distr = self._vf.get_output_for(self._observations_ph, reuse=True, keep_distrib=True)  # N
        self._vf_t = self._vf.get_output_for(self._observations_ph, reuse=True)  # N
        self._vf_params = self._vf.get_params_internal()

        self._vf_t_distr = self.project_TZ( self._vf_t_distr)
       # self._vf_t_distr = tf.negative(self._vf_t_distr)

        if self._action_prior == 'normal':
            D_s = actions.shape.as_list()[-1]
            policy_prior = tf.contrib.distributions.MultivariateNormalDiag(
                loc=tf.zeros(D_s), scale_diag=tf.ones(D_s))
            policy_prior_log_probs = policy_prior.log_prob(actions)
        elif self._action_prior == 'uniform':
            policy_prior_log_probs = 0.0

        log_target = self._qf.get_output_for(
            self._observations_ph, actions, reuse=True, keep_distrib=True)  # N
        log_target = self.project_TZ( log_target)
       # log_target= tf.negative(log_target)

        policy_kl_loss = tf.reduce_mean(log_pi * tf.stop_gradient(
            log_pi - log_target + self._vf_t_distr - policy_prior_log_probs))

        policy_regularization_losses = tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES,
            scope=self._policy.name)
        policy_regularization_loss = tf.reduce_sum(
            policy_regularization_losses)

        policy_loss = (policy_kl_loss
                       + policy_regularization_loss)

        self._vf_loss_t = 0.5 * tf.reduce_mean((
          self._vf_t_distr
          - tf.stop_gradient(log_target - log_pi + policy_prior_log_probs)
        )**2)

        policy_train_op = tf.train.AdamOptimizer(self._policy_lr).minimize(
            loss=policy_loss,
            var_list=self._policy.get_params_internal()
        )

        vf_train_op = tf.train.AdamOptimizer(self._vf_lr).minimize(
            loss=self._vf_loss_t,
            var_list=self._vf_params
        )

        self._training_ops.append(policy_train_op)
        self._training_ops.append(vf_train_op)


    def _init_target_ops(self):
        """Create tensorflow operations for updating target value function."""

        source_params = self._vf_params
        target_params = self._vf_target_params

        self._target_ops = [
            tf.assign(target, (1 - self._tau) * target + self._tau * source)
            for target, source in zip(target_params, source_params)
        ]

    @overrides
    def _init_training(self, env, policy, pool):
        super(SAC, self)._init_training(env, policy, pool)
        self._sess.run(self._target_ops)

    @overrides
    def _do_training(self, iteration, batch):
        """Runs the operations for updating training and target ops."""

        feed_dict = self._get_feed_dict(iteration, batch)
        self._sess.run(self._training_ops, feed_dict)

        if iteration % self._target_update_interval == 0:
            # Run target ops here.
            self._sess.run(self._target_ops)

    def _get_feed_dict(self, iteration, batch):
        """Construct TensorFlow feed_dict from sample batch."""

        feed_dict = {
            self._observations_ph: batch['observations'],
            self._actions_ph: batch['actions'],
            self._next_observations_ph: batch['next_observations'],
            self._rewards_ph: batch['rewards'],
            self._terminals_ph: batch['terminals'],
            self.not_done_ph : 1 - batch['terminals']
        }

        if iteration is not None:
            feed_dict[self._iteration_pl] = iteration

        return feed_dict

    @overrides
    def log_diagnostics(self, iteration, batch):
        """Record diagnostic information to the logger.

        Records mean and standard deviation of Q-function and state
        value function, and TD-loss (mean squared Bellman error)
        for the sample batch.

        Also calls the `draw` method of the plotter, if plotter defined.
        """

        feed_dict = self._get_feed_dict(iteration, batch)
        qf, vf, td_loss = self._sess.run(
            (self._qf_t, self._vf_t, self._td_loss_t), feed_dict)

        logger.record_tabular('qf-avg', np.mean(qf))
        logger.record_tabular('qf-std', np.std(qf))
        logger.record_tabular('vf-avg', np.mean(vf))
        logger.record_tabular('vf-std', np.std(vf))
        logger.record_tabular('mean-sq-bellman-error', td_loss)

        self._policy.log_diagnostics(iteration, batch)
        if self._plotter:
            self._plotter.draw()

    @overrides
    def get_snapshot(self, epoch):
        """Return loggable snapshot of the SAC algorithm.

        If `self._save_full_state == True`, returns snapshot of the complete
        SAC instance. If `self._save_full_state == False`, returns snapshot
        of policy, Q-function, state value function, and environment instances.
        """

        if self._save_full_state:
            snapshot = {
                'epoch': epoch,
                'algo': self
            }
        else:
            snapshot = {
                'epoch': epoch,
                'policy': self._policy,
                'qf': self._qf,
                'vf': self._vf,
                'env': self._env,
            }

        return snapshot

    def __getstate__(self):
        """Get Serializable state of the RLALgorithm instance."""

        d = Serializable.__getstate__(self)
        d.update({
            'qf-params': self._qf.get_param_values(),
            'vf-params': self._vf.get_param_values(),
            'policy-params': self._policy.get_param_values(),
            'pool': self._pool.__getstate__(),
            'env': self._env.__getstate__(),
        })
        return d

    def __setstate__(self, d):
        """Set Serializable state fo the RLAlgorithm instance."""

        Serializable.__setstate__(self, d)
        self._qf.set_param_values(d['qf-params'])
        self._vf.set_param_values(d['vf-params'])
        self._policy.set_param_values(d['policy-params'])
        self._pool.__setstate__(d['pool'])
        self._env.__setstate__(d['env'])
