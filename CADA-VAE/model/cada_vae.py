import tensorflow as tf

from model import VAE
from model.vae import reconstruction_loss


def distribution_alignment_loss(mu_a, logvar_a, mu_b, logvar_b):
    mu_square_l2_norm = tf.reduce_sum(tf.square(mu_a - mu_b), axis=-1)

    sigma_a = tf.sqrt(tf.exp(logvar_a))
    sigma_b = tf.sqrt(tf.exp(logvar_b))

    sigma_square_fro_norm = tf.reduce_sum(
        tf.square(sigma_a - sigma_b),
        axis=-1
    )
    loss = tf.sqrt(mu_square_l2_norm + sigma_square_fro_norm)
    loss = tf.reduce_mean(loss)
    return loss


class CadaVAE(tf.keras.Model):

    def __init__(self, config, **kwargs):
        super(CadaVAE, self).__init__(**kwargs)
        self.config = config
        self.latent_size = config.VAE.latent_size

        self.img_feat_vae = VAE(encoder_hidden=config.ImgFeatVAE.encoder_hidden,
                                decoder_hidden=config.ImgFeatVAE.decoder_hidden,
                                latent=self.latent_size,
                                kernel_initializer_config=config.ImgFeatVAE.kernel_initializer,
                                name='img_vae')
        self.aux_data_vae = VAE(encoder_hidden=config.AuxDataVAE.encoder_hidden,
                                decoder_hidden=config.AuxDataVAE.decoder_hidden,
                                latent=self.latent_size,
                                kernel_initializer_config=config.AuxDataVAE.kernel_initializer,
                                name='aux_vae')

    def reconstruction_loss(self, *, targets, reconstructed):
        return reconstruction_loss(targets=targets, reconstructed=reconstructed,
                                   loss=self.config.VAE.reconstruction_loss)
