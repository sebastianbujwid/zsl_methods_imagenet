import pickle
import pymatreader
import tensorflow as tf
import numpy as np


def extract_imagenet_id_details(imagenet_w2v_extra_pkl):
    with open(imagenet_w2v_extra_pkl, 'rb') as f:
        dict_data = pickle.load(f)
    imagenet_details = {id + 1: (wnid, words) for id, (wnid, words) in enumerate(
        zip(dict_data['wnids'], dict_data['words'])
    )}
    return imagenet_details


def load_imagenet1k_data(imagenet1k, batch_size,
                         seen_classes,  # ImageNet IDs (starting from 1): e.g. [1, 1000]
                         unseen_classes):

    output_types = (tf.float32, tf.float32, tf.string)
    output_shapes = (tf.TensorShape([None, imagenet1k.img_feats_dim]),
                     tf.TensorShape([None, imagenet1k.aux_feats_dim]),
                     tf.TensorShape([None]))

    train_seen_img_aux_label_data = tf.data.Dataset.from_generator(
        lambda: imagenet1k.img_aux_label_generator(split='train',
                                                   batch_size=batch_size,
                                                   allow_only_classes=seen_classes),
        output_types=output_types,
        output_shapes=output_shapes,
    )
    val_seen_img_aux_label_data = tf.data.Dataset.from_generator(
        lambda: imagenet1k.img_aux_label_generator(split='val',
                                                   batch_size=batch_size,
                                                   allow_only_classes=seen_classes),
        output_types=output_types,
        output_shapes=output_shapes
    )

    val_unseen_img_aux_label_data = tf.data.Dataset.from_generator(
        lambda: imagenet1k.img_aux_label_generator(split='val',
                                                   batch_size=batch_size,
                                                   allow_only_classes=unseen_classes),
        output_types=output_types,
        output_shapes=output_shapes
    )

    prefetch_buffer = 10
    return train_seen_img_aux_label_data.prefetch(prefetch_buffer),\
           val_seen_img_aux_label_data.prefetch(prefetch_buffer),\
           val_unseen_img_aux_label_data.prefetch(prefetch_buffer)


def get_imagenet1k_train_val_class_splits(config):
    att_split = pymatreader.read_mat(config.env.imagenet_class_splits,
                                     variable_names=['train_classes', 'val_classes'])
    train_classes = set(np.array(att_split['train_classes'], dtype=np.int64))
    val_classes = set(np.array(att_split['val_classes'], dtype=np.int64))

    assert min(min(train_classes), min(val_classes)) == 1
    assert max(max(train_classes), max(val_classes)) == 1000

    return train_classes, val_classes


# TODO - should be renamed! The name is almost the same as the other function!
def get_imagenet1k_trainval_classes(config, test_split):
    att_split = pymatreader.read_mat(config.env.imagenet_class_splits)

    trainval_classes = set(np.array(att_split['trainval_classes'], dtype=np.int64))
    assert min(trainval_classes) == 1
    assert max(trainval_classes) == 1000

    test_classes = set(np.array(att_split[test_split], dtype=np.int64))

    return trainval_classes, test_classes
