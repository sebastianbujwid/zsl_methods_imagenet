import logging
import numpy as np
import tensorflow as tf

from data import LabelMapper
from simple_zsl import SimpleZSL
from simple_zsl import ranking_loss_with_targets
from metrics import TopKPerClassAccuracy
from training_utils import log_metric


def get_train_epoch_function():

    mean_loss = tf.keras.metrics.Mean(name='train_loss')

    @tf.function
    def train_epoch(model, img_aux_label_data,
                    all_aux_feat,  # NOTE: the order doesn't matter
                    optimizer, model_config, epoch, tf_writer):

        mean_loss.reset_states()

        margin = model_config.margin

        pbar = tf.keras.utils.Progbar(None, unit_name='updates')
        for i, (img_feat, aux_feat, wnid_labels) in enumerate(img_aux_label_data):

            with tf.GradientTape() as tape:

                sim = model.compute_similarity(img_feat=img_feat, aux_feat=all_aux_feat)
                tsim = model.compute_target_similarity(img_feat=img_feat, target_aux_feat=aux_feat)

                loss = ranking_loss_with_targets(sim, tsim, margin)

            variables = model.trainable_variables
            gradients = tape.gradient(loss, variables)

            optimizer.apply_gradients(zip(gradients, variables))

            mean_loss.update_state(loss)

            if i % 200 == 0:
                tf.py_function(
                    lambda l: pbar.add(200, values=[('loss', l)]),
                    [loss],
                    []
                )

        with tf_writer.as_default():
            log_metric(mean_loss, step=epoch)

        tf_writer.flush()

    return train_epoch


@tf.function
def compute_similarity(model: SimpleZSL,
                       *,
                       img_feat,
                       aux_feat):

    return model.compute_similarity(img_feat=img_feat, aux_feat=aux_feat)


def evaluate(img_aux_feat_label_data,
             aux_feat,  # NOTE: the order must correspond to the label_mapper
             model: SimpleZSL,
             label_mapper: LabelMapper,
             total_num_required_classes,
             tf_writer: tf.summary.SummaryWriter,
             step, name):

    num_classes = tf.shape(aux_feat)[0]

    top1_per_class_acc = TopKPerClassAccuracy(num_classes=num_classes, top_k=1,
                                              name=f'{name}_top1_per_class_acc')
    top5_per_class_acc = TopKPerClassAccuracy(num_classes=num_classes, top_k=5,
                                              name=f'{name}_top5_per_class_acc')

    pbar = tf.keras.utils.Progbar(None, unit_name='samples')
    for img_feat, _, label in img_aux_feat_label_data:

        sim = compute_similarity(model, img_feat=img_feat, aux_feat=aux_feat)
        sim = sim.numpy()
        y_true = np.array(label_mapper.names_to_ids(label))

        top1_per_class_acc.update_state(y_true=y_true, pred_logits=sim)
        top5_per_class_acc.update_state(y_true=y_true, pred_logits=sim)

        pbar.add(img_feat.shape[0])

    logging.info('\n')
    logging.info(f'Mean per class top-1 accuracy: {top1_per_class_acc.result()}')
    logging.info(f'Mean per class top-5 accuracy: {top5_per_class_acc.result()}')

    with tf_writer.as_default():
        log_metric(top1_per_class_acc, step=step)
        log_metric(top5_per_class_acc, step=step)

        tf.summary.scalar(f'{top1_per_class_acc.name}_all_classes',
                          top1_per_class_acc.result_expected_classes(total_num_required_classes),
                          step=step)
        tf.summary.scalar(f'{top5_per_class_acc.name}_all_classes',
                          top5_per_class_acc.result_expected_classes(total_num_required_classes),
                          step=step)

    tf_writer.flush()

    return top1_per_class_acc, top5_per_class_acc


def test_evaluate(img_feat_label_data,
                  aux_feat,  # NOTE: the order must correspond to the label_mapper
                  model: SimpleZSL,
                  label_mapper: LabelMapper,
                  total_num_required_classes,
                  tf_writer: tf.summary.SummaryWriter,
                  step, name):

    num_classes = tf.shape(aux_feat)[0]

    top1_per_class_acc = TopKPerClassAccuracy(num_classes=num_classes, top_k=1,
                                              name=f'{name}_top1_per_class_acc')
    top5_per_class_acc = TopKPerClassAccuracy(num_classes=num_classes, top_k=5,
                                              name=f'{name}_top5_per_class_acc')

    pbar = tf.keras.utils.Progbar(None, unit_name='samples')
    for img_feat, label in img_feat_label_data:

        sim = compute_similarity(model, img_feat=img_feat, aux_feat=aux_feat)
        sim = sim.numpy()
        y_true = np.array(label_mapper.names_to_ids(label))

        top1_per_class_acc.update_state(y_true=y_true, pred_logits=sim)
        top5_per_class_acc.update_state(y_true=y_true, pred_logits=sim)

        pbar.add(img_feat.shape[0])

    logging.info('\n')
    logging.info(f'Mean per class top-1 accuracy: {top1_per_class_acc.result()}')
    logging.info(f'Mean per class top-5 accuracy: {top5_per_class_acc.result()}')

    with tf_writer.as_default():
        log_metric(top1_per_class_acc, step=step)
        log_metric(top5_per_class_acc, step=step)

        tf.summary.scalar(f'{top1_per_class_acc.name}_all_classes',
                          top1_per_class_acc.result_expected_classes(total_num_required_classes),
                          step=step)
        tf.summary.scalar(f'{top5_per_class_acc.name}_all_classes',
                          top5_per_class_acc.result_expected_classes(total_num_required_classes),
                          step=step)

    tf_writer.flush()

    return top1_per_class_acc, top5_per_class_acc
