import pickle
import argparse
import logging
import scipy.stats
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from pathlib import Path

from definitions import ROOT_DIR
from utils import ExpEnv
from model import CadaVAE, LinearClassifier
from data import ImageNet1K, ImageNet20K
from data import load_imagenet1k_data, get_imagenet1k_trainval_classes
from data import LabelMapper
from data.imagenet_utils import extract_imagenet_id_details
from training import train_vaes_epoch, get_train_classifier_epoch_function
from evaluation import evaluate


logging.basicConfig(format='%(asctime)s %(levelname)-8s %(module)20.20s.%(funcName)20.20s -:- %(message)s',
                    level=logging.DEBUG)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--default_config', type=Path, default=ROOT_DIR / 'CADA-VAE' / 'configs' / 'default_config.yml')
    parser.add_argument('--config', type=Path, default=ROOT_DIR / 'CADA-VAE' / 'configs' / 'cadavae_config.yml')
    parser.add_argument('--env_config', type=Path, default=ROOT_DIR / 'CADA-VAE' / 'configs' / 'env_config.yml')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test_split', required=True, type=str)
    parser.add_argument('--vaes_checkpoint', type=Path)
    parser.add_argument('--eval_batch_size', type=int, default=128)
    parser.add_argument('--overwrite', nargs='+')
    return parser.parse_args()


def main(args):
    if args.seed:
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)

    exp_env = ExpEnv(args, configs=[args.default_config, args.config, args.env_config],
                     overwrite_config=args.overwrite)
    config = exp_env.config

    imagenet1k = ImageNet1K(hdf5_file=config.env.imagenet_1k,
                            aux_feats_file=config.env.aux_feats,
                            load_to_memory=config.env.load_img_feats_to_memory)

    all_train_classes, test_classes = get_imagenet1k_trainval_classes(config, test_split=args.test_split)
    train_seen_classes = imagenet1k.get_imagnet_ids_with_aux().intersection(all_train_classes)
    test_classes_with_aux = imagenet1k.get_imagnet_ids_with_aux().intersection(test_classes)

    if config.CadaVAE.generalized_zsl:
        predict_classes = train_seen_classes.union(test_classes_with_aux)
        num_required_predict_classes = len(all_train_classes) + len(test_classes)
    else:
        predict_classes = test_classes_with_aux
        num_required_predict_classes = len(test_classes)
    assert num_required_predict_classes >= len(predict_classes)
    label_mapper = LabelMapper(all_class_names=imagenet1k.get_aux_wnid(imagenet_ids=predict_classes))
    logging.info(f'Preparing splits:\n')
    logging.info(f'train_seen_classes: {len(train_seen_classes)}/{len(all_train_classes)}')
    logging.info(f'test_classes_with_aux: {len(test_classes_with_aux)}/{len(test_classes)}')
    logging.info(f'Required classes to predict: {num_required_predict_classes}')

    # Load data
    imagenet_1k_data = load_imagenet1k_data(imagenet1k, batch_size=config.CadaVAE.VAE.batch_size,
                                            seen_classes=train_seen_classes,
                                            unseen_classes=set())
    train_seen_img_aux_label_data, val_seen_img_aux_label_data, _ = imagenet_1k_data

    # Create models
    cada_vae = CadaVAE(config=config.CadaVAE)

    if args.vaes_checkpoint:
        raise NotImplementedError()

    else:
        optimizer_vae = tf.keras.optimizers.Adam(learning_rate=config.CadaVAE.VAE.learning_rate,
                                                 beta_1=0.9, beta_2=0.999,
                                                 epsilon=1e-08,
                                                 amsgrad=True)

        for vae_epoch in tf.range(config.CadaVAE.VAE.epochs, dtype=tf.int64):

            logging.info(f'Training VAEs: epoch {vae_epoch}')

            train_vaes_epoch(train_seen_img_aux_label_data, cada_vae, optimizer_vae, config.CadaVAE.VAE,
                             vae_epoch, exp_env.tf_writer)

    # Create and train a classifier
    classifier_config = config.CadaVAE.LinearClassifier
    classifier = LinearClassifier(classifier_config, num_classes=len(predict_classes))
    classifier_use_weight_decay = classifier_config.weight_decay is not None
    if classifier_use_weight_decay:
        optimizer_classifier = tfa.optimizers.AdamW(weight_decay=classifier_config.weight_decay,
                                                    learning_rate=classifier_config.learning_rate,
                                                    beta_1=0.5,
                                                    beta_2=0.999,
                                                    epsilon=1e-08)
    else:
        optimizer_classifier = tf.keras.optimizers.Adam(learning_rate=classifier_config.learning_rate,
                                                        beta_1=0.5,
                                                        beta_2=0.999,
                                                        epsilon=1e-08)
    z_latent_samples = imagenet1k.z_latent_labels_samples(
        generalized_zsl=config.CadaVAE.generalized_zsl,
        img_classes=train_seen_classes, aux_classes=test_classes_with_aux,
        label_mapper=label_mapper,
        img_encode_func=cada_vae.img_feat_vae.encode,
        aux_encode_func=cada_vae.aux_data_vae.encode,
        num_img_samples_per_class=classifier_config.num_img_samples_per_class,
        num_aux_samples_per_class=classifier_config.num_aux_samples_per_class,
        batch_size=classifier_config.batch_size)

    train_classifier_epoch_function = get_train_classifier_epoch_function()

    for clf_epoch in tf.range(config.CadaVAE.LinearClassifier.epochs, dtype=tf.int64):
        logging.info(f'Training LinearClassifier: epoch {clf_epoch}')

        train_classifier_epoch_function(z_latent_samples, classifier,
                                        optimizer=optimizer_classifier,
                                        tf_writer=exp_env.tf_writer,
                                        step=clf_epoch,
                                        use_weight_decay=classifier_use_weight_decay)

    step = tf.constant(config.CadaVAE.LinearClassifier.epochs - 1, dtype=tf.int64)

    logging.info('Saving model weights...')
    classifier.save_weights(str(exp_env.checkpoints_dir / f'linear_classifier_{args.test_split}'))
    if args.vaes_checkpoint is None:
        cada_vae.save_weights(str(exp_env.checkpoints_dir / f'cada_vae_model'))
    else:
        Path(exp_env.checkpoints_dir / 'cada_vae_model').symlink_to(args.vaes_checkpoint)

    pickle.dump(label_mapper, open(exp_env.checkpoints_dir / 'label_mapper.pkl', 'wb'))
    pickle.dump(label_mapper._id_to_name, open(exp_env.checkpoints_dir / 'label_mapping.pkl', 'wb'))

    # Evaluate the classifier
    if config.CadaVAE.generalized_zsl:
        val_seen_img_label_data = val_seen_img_aux_label_data.map(
            lambda img, aux, label: (img, label)
        )
        logging.info(f'Evaluating on val_seen images...')
        val_seen_top1_acc, val_seen_top5_acc = evaluate(
            val_seen_img_label_data, classifier, encode_func=cada_vae.img_feat_vae.encode,
            label_mapper=label_mapper,
            total_num_required_classes=len(all_train_classes),
            tf_writer=exp_env.tf_writer,
            step=step,
            name='val_seen_img'
        )

    imagenet20k = ImageNet20K(Path(config.env.imagenet_20k_dir),
                              imagenet_id_details=extract_imagenet_id_details(config.env.imagenet_w2v_extra_pkl))

    test_unseen_img_label_data = imagenet20k.get_img_feats(imagenet_ids=test_classes_with_aux,
                                                           max_batch_size=args.eval_batch_size)

    logging.info(f'Evaluating on test_unseen images...')
    test_unseen_top1_acc, test_unseen_top5_acc = evaluate(
        test_unseen_img_label_data, classifier, encode_func=cada_vae.img_feat_vae.encode,
        label_mapper=label_mapper,
        total_num_required_classes=len(test_classes),
        tf_writer=exp_env.tf_writer,
        step=step,
        name='test_unseen_img'
    )

    logging.info('-' * 10 + ' RESULTS ' + '-' * 10)
    results = {
        'test_unseen_top1_acc': test_unseen_top1_acc.per_class_results_mapped(label_mapper),
        'test_unseen_top1_acc_mean': test_unseen_top1_acc.result(),
        'test_unseen_top1_acc_mean_allclasses': test_unseen_top1_acc.result_expected_classes(len(test_classes)),
        'test_unseen_top5_acc': test_unseen_top5_acc.per_class_results_mapped(label_mapper),
        'test_unseen_top5_acc_mean': test_unseen_top5_acc.result(),
        'test_unseen_top5_acc_mean_allclasses': test_unseen_top5_acc.result_expected_classes(len(test_classes)),
    }
    logging.info(f"test_unseen_top1_acc_mean_allclasses: {results['test_unseen_top1_acc_mean_allclasses']}")
    logging.info(f"test_unseen_top5_acc_mean_allclasses: {results['test_unseen_top5_acc_mean_allclasses']}")

    if config.CadaVAE.generalized_zsl:
        results['val_seen_top1_acc'] = val_seen_top1_acc.per_class_results_mapped(label_mapper)
        results['val_seen_top1_acc_mean'] = val_seen_top1_acc.result()
        results['val_seen_top1_acc_mean_allclasses'] = val_seen_top1_acc.result_expected_classes(len(all_train_classes))
        results['val_seen_top5_acc'] = val_seen_top5_acc.per_class_results_mapped(label_mapper)
        results['val_seen_top5_acc_mean'] = val_seen_top5_acc.result()
        results['val_seen_top5_acc_mean_allclasses'] = val_seen_top5_acc.result_expected_classes(len(all_train_classes))

        # Calc harmonic mean (seen + unseen)
        assert len(all_train_classes) + len(test_classes) == num_required_predict_classes
        hmean_top1 = scipy.stats.hmean(
            [results['val_seen_top1_acc_mean_allclasses'],
             results['test_unseen_top1_acc_mean_allclasses']]
        )
        hmean_top5 = scipy.stats.hmean(
            [results['val_seen_top5_acc_mean_allclasses'],
             results['test_unseen_top5_acc_mean_allclasses']]
        )
        results['hmean_top1'] = hmean_top1
        results['hmean_top5'] = hmean_top5

        with exp_env.tf_writer.as_default():
            tf.summary.scalar('hmean_top1', hmean_top1, step=step)
            tf.summary.scalar('hmean_top5', hmean_top5, step=step)

        logging.info('')
        logging.info(f"val_seen_top1_acc_mean_allclasses: {results['val_seen_top1_acc_mean_allclasses']}")
        logging.info(f"val_seen_top5_acc_mean_allclasses: {results['val_seen_top5_acc_mean_allclasses']}")

        logging.info('')
        logging.info(f'hmean_top1: {hmean_top1}')
        logging.info(f'hmean_top5: {hmean_top5}')

    with open(exp_env.run_dir / f'test_results_{args.test_split}.pkl', 'wb') as f:
        pickle.dump(results, f)

    classifier.save_weights(str(exp_env.checkpoints_dir / f'linear_classifier_{args.test_split}'))
    if args.vaes_checkpoint is None:
        cada_vae.save_weights(str(exp_env.checkpoints_dir / f'cada_vae_model'))
    else:
        Path(exp_env.checkpoints_dir / 'cada_vae_model').symlink_to(args.vaes_checkpoint)


if __name__ == '__main__':
    main(parse_args())
