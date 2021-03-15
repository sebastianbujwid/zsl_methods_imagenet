import tensorflow as tf


def get_initializer(config):
    if config.type == 'GlorotUniform':
        gain = config.gain or 1.0
        return lambda shape, dtype=tf.float32: gain * tf.initializers.GlorotUniform()(shape, dtype=dtype)
    else:
        raise NotImplementedError()
