import tensorflow as tf


class Encoder(tf.keras.layers.Layer):

    def __init__(self, target_space_size, **kwargs):
        super(Encoder, self).__init__(**kwargs)

        self.target_space_size = target_space_size

        self.proj = tf.keras.layers.Dense(
            units=self.target_space_size,
            activation=None,
        )

    def call(self, inputs, **kwargs):
        return self.proj(inputs)


class SimpleZSL(tf.keras.Model):

    def __init__(self, config, **kwargs):
        super(SimpleZSL, self).__init__(**kwargs)
        self.config = config

        self.target_space_size = config.target_space_size

        self.img_feat_enc = Encoder(target_space_size=self.target_space_size)
        self.aux_feat_enc = Encoder(target_space_size=self.target_space_size)

    def encode(self, *, img_feat, aux_feat):
        img_feat_target = self.img_feat_enc(img_feat)
        aux_feat_target = self.aux_feat_enc(aux_feat)
        return img_feat_target, aux_feat_target

    def compute_similarity(self, *, img_feat, aux_feat):
        enc_img, enc_aux = self.encode(img_feat=img_feat, aux_feat=aux_feat)
        sim = SimpleZSL.similarity_func(enc_img, enc_aux)
        return sim

    def compute_target_similarity(self, img_feat, target_aux_feat):
        enc_img, enc_aux = self.encode(img_feat=img_feat, aux_feat=target_aux_feat)

        tf.assert_equal(tf.shape(enc_img), tf.shape(enc_aux))
        tsim = tf.reduce_sum(enc_img * enc_aux, axis=-1)
        return tsim

    @staticmethod
    def similarity_func(a, b):
        sim = a @ tf.transpose(b)
        return sim


def ranking_loss_with_targets(sim, target_sim, margin):
    tf.assert_greater(margin, 0.)
    # tf.assert_equal(tf.shape(target_sim), tf.TensorShape(tf.shape(sim)[:1]))

    loss = tf.reduce_sum(tf.maximum(0., margin - tf.expand_dims(target_sim, 1) + sim), axis=-1) - margin
    loss = tf.reduce_mean(loss)

    return loss


def ranking_loss(sim, labels, margin):
    tf.assert_greater(margin, 0.)

    batch_size = tf.shape(sim)[0]
    tf.assert_equal(tf.shape(labels)[0], batch_size)

    label_select_indices = tf.stack(
        [tf.range(tf.cast(batch_size, dtype=tf.int64)),
         labels],
        axis=-1
    )
    target_sim = tf.gather_nd(sim, indices=label_select_indices)
    tf.assert_equal(tf.shape(target_sim), tf.shape(labels))

    loss = tf.reduce_sum(tf.maximum(0., margin - tf.expand_dims(target_sim, 1) + sim), axis=-1) - margin
    loss = tf.reduce_mean(loss)

    return loss


def main():
    from omegaconf import OmegaConf

    tf.random.set_seed(42)

    config = OmegaConf.create({
        'target_space_size': 128
    })

    b = 2
    c = 3
    img_feat = tf.random.normal([b, 2048])
    aux_feat = tf.random.normal([c, 500])
    labels = tf.random.uniform([b], maxval=c, dtype=tf.int64)
    model = SimpleZSL(config)

    sim = model.compute_similarity(img_feat=img_feat, aux_feat=aux_feat)
    loss = ranking_loss(sim, labels, margin=1.)

    predicted = model.compute_similarity(img_feat=img_feat, aux_feat=aux_feat)

    pred_classes = tf.argmax(predicted, axis=-1)

    a = None


if __name__ == '__main__':
    main()
