import tensorflow as tf


def log_metric(m, step):
    tf.summary.scalar(m.name, m.result(), step=step)


def lr_coefficient(epoch, config):
    if isinstance(config, float):
        coeff = config
        return coeff

    return calc_lr_coefficient(
        epoch,
        start_epoch=config.start_epoch,
        end_epoch=config.end_epoch,
        factor=config.factor
    )


def calc_lr_coefficient(epoch, start_epoch, end_epoch, factor):
    n_epochs = end_epoch - start_epoch
    coeff = tf.cast(tf.maximum(epoch - start_epoch, 0) / n_epochs, dtype=tf.float32)
    coeff = tf.minimum(coeff, 1.0)
    coeff = coeff * factor
    return coeff


if __name__ == '__main__':
    print([(i, calc_lr_coefficient(tf.convert_to_tensor(i, dtype=tf.int64), start_epoch=2, end_epoch=4, factor=10.).numpy()) for i in range(8)])
