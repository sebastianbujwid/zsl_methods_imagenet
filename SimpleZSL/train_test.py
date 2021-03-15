import pickle
import logging
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path

from definitions import ROOT_DIR
from utils import ExpEnv
from simple_zsl import SimpleZSL
from szsl_utils import get_train_epoch_function, test_evaluate

from data import LabelMapper
from data import ImageNet1K, ImageNet20K
from data.imagenet_utils import extract_imagenet_id_details
from data import load_imagenet1k_data, get_imagenet1k_trainval_classes


logging.basicConfig(format='%(asctime)s %(levelname)-8s %(module)20.20s.%(funcName)20.20s -:- %(message)s',
                    level=logging.DEBUG)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT_DIR / 'SimpleZSL' / 'configs' / 'simple_zsl_config.yml')
    parser.add_argument('--env_config', type=Path, default=ROOT_DIR / 'CADA-VAE' / 'configs' / 'env_config.yml')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test_split', required=True, type=str)
    parser.add_argument('--eval_batch_size', type=int, default=128)
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

    all_train_classes, test_classes = get_imagenet1k_trainval_classes(config, test_split=args.test_split)
    train_seen_classes = imagenet1k.get_imagnet_ids_with_aux().intersection(all_train_classes)
    test_classes_with_aux = imagenet1k.get_imagnet_ids_with_aux().intersection(test_classes)

    predict_classes = test_classes_with_aux
    num_required_predict_classes = len(test_classes)
    assert num_required_predict_classes >= len(predict_classes)
    # seen_label_mapper = LabelMapper(all_class_names=imagenet1k.get_aux_wnid(imagenet_ids=train_seen_classes))
    seen_aux_feat = imagenet1k.get_aux_features(train_seen_classes)
    unseen_label_mapper = LabelMapper(all_class_names=imagenet1k.get_aux_wnid(imagenet_ids=predict_classes))
    unseen_aux_feat = imagenet1k.get_aux_features(predict_classes)
    logging.info(f'Preparing splits:\n')
    logging.info(f'train_seen_classes: {len(train_seen_classes)}/{len(all_train_classes)}')
    logging.info(f'test_classes_with_aux: {len(test_classes_with_aux)}/{len(test_classes)}')
    logging.info(f'Required classes to predict: {num_required_predict_classes}')

    # Load data
    imagenet_1k_data = load_imagenet1k_data(imagenet1k, batch_size=model_config.batch_size,
                                            seen_classes=train_seen_classes,
                                            unseen_classes=set())
    train_seen_img_aux_label_data, val_seen_img_aux_label_data, _ = imagenet_1k_data

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

    logging.info(f'Evaluating on test_unseen images...')

    imagenet20k = ImageNet20K(Path(config.env.imagenet_20k_dir),
                              imagenet_id_details=extract_imagenet_id_details(config.env.imagenet_w2v_extra_pkl))

    test_unseen_img_label_data = imagenet20k.get_img_feats(imagenet_ids=test_classes_with_aux,
                                                           max_batch_size=args.eval_batch_size)

    step = tf.constant(model_config.epochs - 1, dtype=tf.int64)

    logging.info('Saving model weights...')
    model.save_weights(str(exp_env.checkpoints_dir / f'simple_zsl_{args.test_split}'))

    pickle.dump(unseen_label_mapper, open(exp_env.checkpoints_dir / 'label_mapper.pkl', 'wb'))
    pickle.dump(unseen_label_mapper._id_to_name, open(exp_env.checkpoints_dir / 'label_mapping.pkl', 'wb'))

    test_unseen_top1_acc, test_unseen_top5_acc = test_evaluate(
        test_unseen_img_label_data,
        tf.convert_to_tensor(unseen_aux_feat),
        model,
        label_mapper=unseen_label_mapper,
        total_num_required_classes=num_required_predict_classes,
        tf_writer=exp_env.tf_writer,
        step=step,
        name='test_unseen_img')

    logging.info('-' * 10 + ' RESULTS ' + '-' * 10)
    results = {
        'test_unseen_top1_acc': test_unseen_top1_acc.per_class_results_mapped(unseen_label_mapper),
        'test_unseen_top1_acc_mean': test_unseen_top1_acc.result(),
        'test_unseen_top1_acc_mean_allclasses': test_unseen_top1_acc.result_expected_classes(len(test_classes)),
        'test_unseen_top5_acc': test_unseen_top5_acc.per_class_results_mapped(unseen_label_mapper),
        'test_unseen_top5_acc_mean': test_unseen_top5_acc.result(),
        'test_unseen_top5_acc_mean_allclasses': test_unseen_top5_acc.result_expected_classes(len(test_classes)),
    }
    logging.info(f"test_unseen_top1_acc_mean_allclasses: {results['test_unseen_top1_acc_mean_allclasses']}")
    logging.info(f"test_unseen_top5_acc_mean_allclasses: {results['test_unseen_top5_acc_mean_allclasses']}")

    with open(exp_env.run_dir / f'test_results_{args.test_split}.pkl', 'wb') as f:
        pickle.dump(results, f)



if __name__ == '__main__':
    main(parse_args())
