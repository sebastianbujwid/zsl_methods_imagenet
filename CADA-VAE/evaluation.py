import logging
import numpy as np
import tensorflow as tf

from model import LinearClassifier
from metrics import TopKPerClassAccuracy
from training_utils import log_metric
from data import LabelMapper


@tf.function
def predict(feat_data,
            classifier: LinearClassifier,
            encode_func):

    training = tf.constant(False)
    _, vae_mu, _ = encode_func(feat_data, training=training)
    pred_logits = classifier(vae_mu, training=training)
    return pred_logits


def evaluate(img_label_data,
             classifier: LinearClassifier,
             encode_func,
             label_mapper: LabelMapper,
             total_num_required_classes,
             tf_writer: tf.summary.SummaryWriter,
             step,
             name):

    top1_per_class_acc = TopKPerClassAccuracy(num_classes=classifier.num_classes, top_k=1,
                                              name=f'{name}_top1_per_class_acc')
    top5_per_class_acc = TopKPerClassAccuracy(num_classes=classifier.num_classes, top_k=5,
                                              name=f'{name}_top5_per_class_acc')

    pbar = tf.keras.utils.Progbar(None, unit_name='samples')
    for img, label in img_label_data:
        pred_logits = predict(img, classifier, encode_func)

        pred_logits = pred_logits.numpy()
        y_true = np.array(label_mapper.names_to_ids(label))

        top1_per_class_acc.update_state(y_true=y_true, pred_logits=pred_logits)
        top5_per_class_acc.update_state(y_true=y_true, pred_logits=pred_logits)

        pbar.add(img.shape[0])

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
