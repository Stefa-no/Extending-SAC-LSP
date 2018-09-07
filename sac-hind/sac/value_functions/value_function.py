import tensorflow as tf

from rllab.core.serializable import Serializable

from sac.misc.mlp import MLPFunction
from sac.misc import tf_utils

class NNVFunction(MLPFunction):

    def __init__(self, env_spec, hidden_layer_sizes=(100, 100)):
        Serializable.quick_init(self, locals())

        self._Do = env_spec.spec.observation_space.flat_dim
        # we assume goals like obs
        self._Dg = env_spec._wrapped_env.model.nv
        self._obs_pl = tf.placeholder(
            tf.float32,
            shape=[None, self._Do],
            name='observation',
        )
        
        self._goal_pl = tf.placeholder(
            tf.float32,
            shape=[None, self._Dg],
            name='goal',
        )

        super(NNVFunction, self).__init__(
            'vf', (self._obs_pl, self._goal_pl,), hidden_layer_sizes)


class NNQFunction(MLPFunction):
    def __init__(self, env_spec, hidden_layer_sizes=(100, 100)):
        Serializable.quick_init(self, locals())

        self._Da = env_spec.spec.action_space.flat_dim
        self._Do = env_spec.spec.observation_space.flat_dim
        # we assume goals like obs
        self._Dg = env_spec._wrapped_env.model.nv

        self._obs_pl = tf.placeholder(
            tf.float32,
            shape=[None, self._Do],
            name='observation',
        )
        
        self._goal_pl = tf.placeholder(
            tf.float32,
            shape=[None, self._Dg],
            name='goal',
        )

        self._action_pl = tf.placeholder(
            tf.float32,
            shape=[None, self._Da],
            name='actions',
        )

        super(NNQFunction, self).__init__(
            'qf', (self._obs_pl, self._goal_pl, self._action_pl), hidden_layer_sizes)

class NNDiscriminatorFunction(MLPFunction):
    def __init__(self, env_spec, hidden_layer_sizes=(100, 100), num_skills=None):
        assert num_skills is not None
        Serializable.quick_init(self, locals())
        Parameterized.__init__(self)

        self._Da = env_spec.spec.action_space.flat_dim
        self._Do = env_spec.spec.observation_space.flat_dim

        self._obs_pl = tf.placeholder(
            tf.float32,
            shape=[None, self._Do],
            name='observation',
        )
        self._action_pl = tf.placeholder(
            tf.float32,
            shape=[None, self._Da],
            name='actions',
        )
        
        self._goal_pl = tf.placeholder(
            tf.float32,
            shape=[None, self._Dg],
            name='goal',
        )

        self._name = 'discriminator'
        self._input_pls = (self._obs_pl, self._goal_pl, self._action_pl)
        self._layer_sizes = list(hidden_layer_sizes) + [num_skills]
        self._output_t = self.get_output_for(*self._input_pls)

