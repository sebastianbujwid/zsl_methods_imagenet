import tensorflow as tf
import tensorflow_addons as tfa
from typing import Union

from model import CadaVAE, LinearClassifier
from model.cada_vae import distribution_alignment_loss
from model.vae import kl_divergance_normal
from training_utils import lr_coefficient, log_metric


mean_loss = tf.keras.metrics.Mean(name='vae_train_loss')
mean_total_reconstruction_loss = tf.keras.metrics.Mean(name='vae_reconstruction_loss_total')
mean_img_reconstruction_loss = tf.keras.metrics.Mean(name='vae_reconstruction_loss_img')
mean_aux_reconstruction_loss = tf.keras.metrics.Mean(name='vae_reconstruction_loss_aux')
mean_total_kl_loss = tf.keras.metrics.Mean(name='vae_kl_loss_total')
mean_img_kl_loss = tf.keras.metrics.Mean(name='vae_kl_loss_img')
mean_aux_kl_loss = tf.keras.metrics.Mean(name='vae_kl_loss_aux')
mean_total_cross_reconstruction_loss = tf.keras.metrics.Mean(name='vae_cross_reconstruction_loss')
mean_img_2_aux_reconstruction_loss = tf.keras.metrics.Mean(name='vae_reconstruction_loss_img_2_aux')
mean_aux_2_img_reconstruction_loss = tf.keras.metrics.Mean(name='vae_reconstruction_loss_aux_2_img')
mean_distribution_alignment_loss = tf.keras.metrics.Mean(name='vae_distribution_alignment_loss')


@tf.function
def train_vaes_epoch(img_aux_label_data,
                     cada_vae: CadaVAE,
                     optimizer: tf.keras.optimizers.Optimizer,
                     config,
                     epoch,
                     tf_writer: tf.summary.SummaryWriter):

    training = tf.constant(True)

    reconstruction_coeff = 1.0
    beta_coeff = lr_coefficient(epoch, config.beta_coefficient)
    cross_reconstruction_coeff = lr_coefficient(epoch, config.cross_reconstruction_coefficient)
    distribution_alignment_coeff = lr_coefficient(epoch, config.distribution_alignment_coefficient)

    # Log metrics
    all_metrics = [
        mean_loss,
        mean_total_reconstruction_loss,
        mean_img_reconstruction_loss,
        mean_aux_reconstruction_loss,
        mean_total_kl_loss,
        mean_img_kl_loss,
        mean_aux_kl_loss,
        mean_total_cross_reconstruction_loss,
        mean_img_2_aux_reconstruction_loss,
        mean_aux_2_img_reconstruction_loss,
        mean_distribution_alignment_loss,
    ]

    for m in all_metrics:
        m.reset_states()

    pbar = tf.keras.utils.Progbar(None, unit_name='updates')
    for i, (img, aux, _) in enumerate(img_aux_label_data):

        with tf.GradientTape() as tape:
            z_img, mu_img, logvar_img = cada_vae.img_feat_vae.encode(img, training=training)
            img_2_img = cada_vae.img_feat_vae.decode(z_img, training=training)

            img_reconstruction_loss = cada_vae.reconstruction_loss(
                targets=img, reconstructed=img_2_img)
            img_kl_loss = kl_divergance_normal(posterior_mean=mu_img,
                                               posterior_log_var=logvar_img)

            z_aux, mu_aux, logvar_aux = cada_vae.aux_data_vae.encode(aux, training=training)
            aux_2_aux = cada_vae.aux_data_vae.decode(z_aux, training=training)

            aux_reconstruction_loss = cada_vae.reconstruction_loss(
                targets=aux, reconstructed=aux_2_aux)
            aux_kl_loss = kl_divergance_normal(posterior_mean=mu_aux,
                                               posterior_log_var=logvar_aux)

            if cross_reconstruction_coeff > 0.0:
                img_2_aux = cada_vae.aux_data_vae.decode(z_img, training=training)
                aux_2_img = cada_vae.img_feat_vae.decode(z_aux, training=training)
                img_2_aux_reconstruction_loss = cada_vae.reconstruction_loss(
                    targets=aux, reconstructed=img_2_aux)
                aux_2_img_reconstruction_loss = cada_vae.reconstruction_loss(
                    targets=img, reconstructed=aux_2_img)
            else:
                img_2_aux_reconstruction_loss = 0.0
                aux_2_img_reconstruction_loss = 0.0

            total_reconstruction_loss = reconstruction_coeff * (img_reconstruction_loss + aux_reconstruction_loss)
            total_kl_loss = beta_coeff * (img_kl_loss + aux_kl_loss)
            total_cross_reconstruction_loss = cross_reconstruction_coeff * \
                                              (img_2_aux_reconstruction_loss + aux_2_img_reconstruction_loss)
            distr_alignment_loss = distribution_alignment_coeff * \
                                   distribution_alignment_loss(mu_a=mu_img, logvar_a=logvar_img,
                                                               mu_b=mu_aux, logvar_b=logvar_aux)

            total_loss = total_reconstruction_loss + total_kl_loss + \
                         distr_alignment_loss + total_cross_reconstruction_loss

        variables = cada_vae.img_feat_vae.trainable_variables + cada_vae.aux_data_vae.trainable_variables
        gradients = tape.gradient(total_loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))

        # Update metrics
        mean_loss.update_state(total_loss)
        mean_total_reconstruction_loss.update_state(total_reconstruction_loss)
        mean_img_reconstruction_loss.update_state(img_reconstruction_loss)
        mean_aux_reconstruction_loss.update_state(aux_reconstruction_loss)
        mean_total_kl_loss.update_state(total_kl_loss)
        mean_img_kl_loss.update_state(img_kl_loss)
        mean_aux_kl_loss.update_state(aux_kl_loss)
        mean_total_cross_reconstruction_loss.update_state(total_cross_reconstruction_loss)
        mean_img_2_aux_reconstruction_loss.update_state(img_2_aux_reconstruction_loss)
        mean_aux_2_img_reconstruction_loss.update_state(aux_2_img_reconstruction_loss)
        mean_distribution_alignment_loss.update_state(distr_alignment_loss)

        if i % 200 == 0:
            tf.py_function(
                lambda t_loss, rec_loss, kl_loss, cross_rec_loss, dist_align_loss:
                pbar.add(200, values=[('total_loss', t_loss),
                                      ('rec_loss', rec_loss),
                                      ('kl_loss', kl_loss),
                                      ('cross_rec_loss', cross_rec_loss),
                                      ('dist_align_loss', dist_align_loss)
                                      ]),
                [total_loss, total_reconstruction_loss, total_kl_loss,
                 total_cross_reconstruction_loss, distr_alignment_loss],
                []
            )

    with tf_writer.as_default():
        for m in all_metrics:
            log_metric(m, step=epoch)
        tf.summary.scalar('vae_reconstruction_coeff', reconstruction_coeff, step=epoch)
        tf.summary.scalar('vae_beta_coeff', beta_coeff, step=epoch)
        tf.summary.scalar('vae_cross_reconstruction_coeff', cross_reconstruction_coeff, step=epoch)
        tf.summary.scalar('vae_distribution_alignment_coeff', distribution_alignment_coeff, step=epoch)

    tf_writer.flush()


def get_train_classifier_epoch_function():

    mean_clf_loss = tf.keras.metrics.Mean(name='clf_train_loss')

    @tf.function
    def train_classifier_epoch(z_latent_samples,
                               classifier: LinearClassifier,
                               optimizer: Union[tf.keras.optimizers.Optimizer, tfa.optimizers.AdamW],
                               tf_writer: tf.summary.SummaryWriter,
                               step,
                               use_weight_decay: bool):

        training = tf.constant(True)
        mean_clf_loss.reset_states()
        pbar = tf.keras.utils.Progbar(None, unit_name='updates')

        for i, (z_latent, labels) in enumerate(z_latent_samples):

            with tf.GradientTape() as tape:
                pred_logits = classifier(z_latent, training=training)

                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                                      logits=pred_logits)
                loss = tf.reduce_mean(loss)

            variables = classifier.trainable_variables
            gradients = tape.gradient(loss, variables)
            if use_weight_decay:
                optimizer.apply_gradients(zip(gradients, variables),
                                          decay_var_list=[v for v in variables if 'bias' not in v.name])
            else:
                optimizer.apply_gradients(zip(gradients, variables))

            mean_clf_loss.update_state(loss)

            if i % 200 == 0:
                tf.py_function(
                    lambda l: pbar.add(200, [('loss', l)]),
                    [loss],
                    []
                )

        with tf_writer.as_default():
            log_metric(mean_clf_loss, step=step)

    return train_classifier_epoch
