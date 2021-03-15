import logging
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path

from definitions import ROOT_DIR
from utils import ExpEnv
from simple_zsl import SimpleZSL
from szsl_utils import get_train_epoch_function, evaluate

from data import LabelMapper
from data import ImageNet1K
from data import get_imagenet1k_train_val_class_splits, load_imagenet1k_data

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(module)20.20s.%(funcName)20.20s -:- %(message)s',
                    level=logging.DEBUG)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT_DIR / 'SimpleZSL' / 'configs' / 'simple_zsl_config.yml')
    parser.add_argument('--env_config', type=Path, default=ROOT_DIR / 'CADA-VAE' / 'configs' / 'env_config.yml')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--overwrite', nargs='+')
    return parser.parse_args()


def main(args):
    if args.seed:
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)

    exp_env = ExpEnv(args, configs=[args.config, args.env_config], overwrite_config=args.overwrite)
    config = exp_env.config
    model_config = config.SimpleZSL

    imagenet1k = ImageNet1K(hdf5_file=config.env.imagenet_1k,
                            aux_feats_file=config.env.aux_feats,
                            load_to_memory=config.env.load_img_feats_to_memory)

    # Select classes
    all_train_classes, val_classes = get_imagenet1k_train_val_class_splits(config)
    train_seen_classes = imagenet1k.get_imagnet_ids_with_aux().intersection(all_train_classes)
    val_classes_with_aux = imagenet1k.get_imagnet_ids_with_aux().intersection(val_classes)

    predict_classes = val_classes_with_aux
    num_required_predict_classes = len(val_classes)
    assert num_required_predict_classes >= len(predict_classes)
    seen_label_mapper = LabelMapper(all_class_names=imagenet1k.get_aux_wnid(imagenet_ids=train_seen_classes))
    seen_aux_feat = imagenet1k.get_aux_features(train_seen_classes)
    unseen_label_mapper = LabelMapper(all_class_names=imagenet1k.get_aux_wnid(imagenet_ids=predict_classes))
    unseen_aux_feat = imagenet1k.get_aux_features(predict_classes)
    logging.info(f'Preparing splits:\n')
    logging.info(f'train_seen_classes: {len(train_seen_classes)}/{len(all_train_classes)}')
    logging.info(f'val_classes_with_aux: {len(val_classes_with_aux)}/{len(val_classes)}')
    logging.info(f'Required classes to predict: {num_required_predict_classes}')

    # Load data
    imagenet_1k_data = load_imagenet1k_data(imagenet1k, batch_size=model_config.batch_size,
                                            seen_classes=train_seen_classes,
                                            unseen_classes=val_classes_with_aux)
    train_seen_img_aux_label_data, val_seen_img_aux_label_data, val_unseen_img_aux_label_data = imagenet_1k_data

    model = SimpleZSL(config=model_config)
    optimizer = tf.keras.optimizers.Adam(learning_rate=model_config.optim.lr,
                                         beta_1=model_config.optim.beta_1,
                                         beta_2=model_config.optim.beta_2,
                                         epsilon=model_config.optim.epsilon,
                                         amsgrad=True)

    train_epoch_function = get_train_epoch_function()

    for epoch in tf.range(model_config.epochs, dtype=tf.int64):

        logging.info(f'\n\nTraining epoch: {epoch}')

        train_epoch_function(model, train_seen_img_aux_label_data,
                             tf.convert_to_tensor(seen_aux_feat),
                             optimizer=optimizer,
                             model_config=model_config,
                             epoch=epoch, tf_writer=exp_env.tf_writer)

        logging.info(f'Evaluating on val_seen images...')
        evaluate(val_seen_img_aux_label_data,
                 tf.convert_to_tensor(seen_aux_feat),
                 model,
                 label_mapper=seen_label_mapper,
                 total_num_required_classes=len(all_train_classes),
                 tf_writer=exp_env.tf_writer,
                 step=epoch,
                 name='val_seen_img')

        logging.info(f'Evaluating on val_unseen images...')

        evaluate(val_unseen_img_aux_label_data,
                 tf.convert_to_tensor(unseen_aux_feat),
                 model,
                 label_mapper=unseen_label_mapper,
                 total_num_required_classes=num_required_predict_classes,
                 tf_writer=exp_env.tf_writer,
                 step=epoch,
                 name='val_unseen_img')


if __name__ == '__main__':
    main(parse_args())
