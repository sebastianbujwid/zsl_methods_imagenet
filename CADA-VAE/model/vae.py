import tensorflow as tf
from tensorflow.keras.layers import Dense

from model.model_utils import get_initializer


def l1_reconstruction_loss(targets, reconstructed):
    l = tf.reduce_mean(
        tf.reduce_sum(
            tf.abs(targets - reconstructed), axis=-1
        )
    )
    return l


def l2_reconstruction_loss(targets, reconstructed):
    l = tf.reduce_mean(
        tf.reduce_sum(
            tf.square(targets - reconstructed), axis=-1
        )
    )
    return l


def reconstruction_loss(*, targets, reconstructed, loss):
    if loss == 'L1':
        return l1_reconstruction_loss(targets, reconstructed)
    elif loss == 'L2':
        return l2_reconstruction_loss(targets, reconstructed)
    else:
        raise NotImplementedError(f'Loss: {loss}')


def kl_divergance_normal(*, posterior_mean, posterior_log_var):
    # NOTE: prior assumed to be N(0, I)
    posterior_var = tf.exp(posterior_log_var)
    l = -0.5 * tf.reduce_sum(
        1. + posterior_log_var - tf.square(posterior_mean) - posterior_var,
        axis=-1
    )
    l = tf.reduce_mean(l, axis=0)
    return l


class VAE(tf.keras.layers.Layer):

    def __init__(self, encoder_hidden, decoder_hidden, latent, kernel_initializer_config=None, **kwargs):
        super(VAE, self).__init__(**kwargs)

        self.encoder_hidden = encoder_hidden
        self.decoder_hidden = decoder_hidden
        self.latent_size = latent

        self.kernel_initializer = None
        if kernel_initializer_config:
            self.kernel_initializer = get_initializer(kernel_initializer_config)

        self.encoder = [Dense(h, activation=tf.nn.relu, kernel_initializer=self.kernel_initializer)
                        for h in self.encoder_hidden]
        self.decoder = [Dense(h, activation=tf.nn.relu, kernel_initializer=self.kernel_initializer)
                        for h in self.decoder_hidden]

        self.latent_layer = Dense(self.latent_size * 2, activation=None, kernel_initializer=self.kernel_initializer)
        self.output_layer = None
        self.original_dim = None

    def build(self, input_shape):
        tf.print('Building a VAE:', self.name)
        self.original_dim = input_shape[-1]
        self.output_layer = Dense(self.original_dim, activation=None, kernel_initializer=self.kernel_initializer)

    def encode(self, _input, training):
        self._maybe_build(_input)

        x = _input
        for l in self.encoder:
            x = l(x, training=training)

        latent = self.latent_layer(x)
        mu, log_var = tf.split(latent, 2, axis=-1)
        z = VAE.reparametrize(mu=mu, log_var=log_var)
        return z, mu, log_var

    def decode(self, z, training):
        x = z
        for l in self.decoder:
            x = l(x, training=training)

        out = self.output_layer(x)
        return out

    @staticmethod
    def reparametrize(*, mu, log_var):
        std = tf.exp(0.5 * log_var)  # sigma = exp(0.5 * log(sigma^2))
        eps = tf.random.normal(tf.shape(mu), 0., 1.)
        return mu + eps * std


""""
if __name__ == '__main__':
    m = VAE([256, 128], [128, 256], 64)
    x = tf.random.normal([4, 512], 0, 1.)
    z, mu, log_var = m.encode(x, training=True)

    out = m.decode(mu, training=True)

    l = reconstruction_loss(x, out, loss='L1')
    print(l)
    a = None
"""