import tensorflow as tf
from rllab import config


def get_default_session():
    return tf.get_default_session() or create_session()


def create_session(**kwargs):
    """ Create new tensorflow session with given configuration. """
    if "config" not in kwargs:
        kwargs["config"] = get_configuration()
        #print(kwargs)
    return tf.InteractiveSession(**kwargs)


def get_configuration():
    """ Returns personal tensorflow configuration. """
    if config.USE_GPU:
        print("supposedly not implemented")
        #raise NotImplementedError
        gpu_options = tf.GPUOptions(allow_growth=True)
        return tf.ConfigProto(
            gpu_options=gpu_options)

    config_args = dict()
    return tf.ConfigProto(**config_args)
