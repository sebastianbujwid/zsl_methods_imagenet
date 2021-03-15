import argparse
import logging
import pickle
import numpy as np
import scipy.stats
import tensorflow as tf
import tensorflow_addons as tfa
from pathlib import Path

from utils import ExpEnv
from definitions import ROOT_DIR
from model import CadaVAE, LinearClassifier
from data import ImageNet1K
from data import load_imagenet1k_data, get_imagenet1k_train_val_class_splits
from data import LabelMapper
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

    # Select classes
    all_train_classes, val_classes = get_imagenet1k_train_val_class_splits(config)
    train_seen_classes = imagenet1k.get_imagnet_ids_with_aux().intersection(all_train_classes)
    val_classes_with_aux = imagenet1k.get_imagnet_ids_with_aux().intersection(val_classes)

    if config.CadaVAE.generalized_zsl:
        predict_classes = train_seen_classes.union(val_classes_with_aux)
        num_required_predict_classes = len(all_train_classes) + len(val_classes)
    else:
        predict_classes = val_classes_with_aux
        num_required_predict_classes = len(val_classes)
    assert num_required_predict_classes >= len(predict_classes)
    label_mapper = LabelMapper(all_class_names=imagenet1k.get_aux_wnid(imagenet_ids=predict_classes))
    pickle.dump(label_mapper, open(exp_env.checkpoints_dir / 'label_mapper.pkl', 'wb'))
    pickle.dump(label_mapper._id_to_name, open(exp_env.checkpoints_dir / 'label_mapping.pkl', 'wb'))

    logging.info(f'Preparing splits:\n')
    logging.info(f'train_seen_classes: {len(train_seen_classes)}/{len(all_train_classes)}')
    logging.info(f'val_classes_with_aux: {len(val_classes_with_aux)}/{len(val_classes)}')
    logging.info(f'Required classes to predict: {num_required_predict_classes}')

    # Load data
    imagenet_1k_data = load_imagenet1k_data(imagenet1k, batch_size=config.CadaVAE.VAE.batch_size,
                                            seen_classes=train_seen_classes,
                                            unseen_classes=val_classes_with_aux)
    train_seen_img_aux_label_data, val_seen_img_aux_label_data, val_unseen_img_aux_label_data = imagenet_1k_data

    # Create models
    cada_vae = CadaVAE(config=config.CadaVAE)
    optimizer_vae = tf.keras.optimizers.Adam(learning_rate=config.CadaVAE.VAE.learning_rate,
                                             beta_1=0.9, beta_2=0.999,
                                             epsilon=1e-08,
                                             amsgrad=True)

    classifier_config = config.CadaVAE.LinearClassifier
    clf_aggregated_epochs = tf.Variable(0, trainable=False, dtype=tf.int64)
    best_score = -1.0

    for vae_epoch in tf.range(config.CadaVAE.VAE.epochs, dtype=tf.int64):

        logging.info(f'Training VAEs: epoch {vae_epoch}')

        train_vaes_epoch(train_seen_img_aux_label_data, cada_vae, optimizer_vae, config.CadaVAE.VAE,
                         vae_epoch, exp_env.tf_writer)

        should_evaluate = ((vae_epoch + 1) % config.CadaVAE.evaluate_every_n_epochs == 0) \
                          or (vae_epoch == config.CadaVAE.VAE.epochs - 1)
        if not should_evaluate:
            continue

        # Create and train a classifier
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
            img_classes=train_seen_classes, aux_classes=val_classes_with_aux,
            label_mapper=label_mapper,
            img_encode_func=cada_vae.img_feat_vae.encode,
            aux_encode_func=cada_vae.aux_data_vae.encode,
            num_img_samples_per_class=classifier_config.num_img_samples_per_class,
            num_aux_samples_per_class=classifier_config.num_aux_samples_per_class,
            batch_size=classifier_config.batch_size)

        train_classifier_epoch_function = get_train_classifier_epoch_function()

        for clf_epoch in tf.range(config.CadaVAE.LinearClassifier.epochs, dtype=tf.int64):

            logging.info(f'VAEs epoch: {vae_epoch}. Training LinearClassifier: epoch {clf_epoch}')

            # TODO - tune hparams separately?
            # TODO - choose a different classifier? SVM, RandomForest, etc?
            train_classifier_epoch_function(z_latent_samples, classifier,
                                            optimizer=optimizer_classifier,
                                            tf_writer=exp_env.tf_writer,
                                            step=clf_aggregated_epochs,
                                            use_weight_decay=classifier_use_weight_decay)

            if config.CadaVAE.generalized_zsl:
                val_seen_aux_label_data = val_seen_img_aux_label_data.map(
                    # aux feats are the same for a given label, but they will be encoded later with sampling
                    # yielding different inputs to the classifier
                    lambda img, aux, label: (aux, label)
                )
                logging.info(f'Evaluating on val_seen aux samples...')
                evaluate(
                    val_seen_aux_label_data, classifier, encode_func=cada_vae.aux_data_vae.encode,
                    label_mapper=label_mapper,
                    total_num_required_classes=len(all_train_classes),
                    tf_writer=exp_env.tf_writer,
                    step=clf_aggregated_epochs,
                    name='val_seen_aux'
                )

                val_seen_img_label_data = val_seen_img_aux_label_data.map(
                    lambda img, aux, label: (img, label)
                )
                logging.info(f'Evaluating on val_seen images...')
                val_seen_top1_acc, val_seen_top5_acc = evaluate(
                    val_seen_img_label_data, classifier, encode_func=cada_vae.img_feat_vae.encode,
                    label_mapper=label_mapper,
                    total_num_required_classes=len(all_train_classes),
                    tf_writer=exp_env.tf_writer,
                    step=clf_aggregated_epochs,
                    name='val_seen_img'
                )

            val_unseen_aux_label_data = val_unseen_img_aux_label_data.map(
                # aux feats are the same for a given label, but they will be encoded later with sampling
                # yielding different inputs to the classifier
                lambda img, aux, label: (aux, label)
            )
            logging.info(f'Evaluating on val_unseen aux samples...')
            evaluate(
                val_unseen_aux_label_data, classifier, encode_func=cada_vae.aux_data_vae.encode,
                label_mapper=label_mapper,
                total_num_required_classes=len(val_classes),
                tf_writer=exp_env.tf_writer,
                step=clf_aggregated_epochs,
                name='val_unseen_aux'
            )

            val_unseen_img_label_data = val_unseen_img_aux_label_data.map(
                lambda img, aux, label: (img, label)
            )
            logging.info(f'Evaluating on val_unseen images...')
            val_unseen_top1_acc, val_unseen_top5_acc = evaluate(
                val_unseen_img_label_data, classifier, encode_func=cada_vae.img_feat_vae.encode,
                label_mapper=label_mapper,
                total_num_required_classes=len(val_classes),
                tf_writer=exp_env.tf_writer,
                step=clf_aggregated_epochs,
                name='val_unseen_img'
            )

            with exp_env.tf_writer.as_default():
                tf.summary.scalar('clf_vae_epoch_vs_clf_step', vae_epoch, step=clf_aggregated_epochs)
            clf_aggregated_epochs.assign_add(1)

            results = {
                'vae_epoch': vae_epoch.numpy(),
                'clf_epoch': clf_epoch.numpy(),
                'clf_aggregated_epoch': clf_aggregated_epochs.value().numpy(),
                'val_unseen_top1_acc': val_unseen_top1_acc.per_class_results_mapped(label_mapper),
                'val_unseen_top1_acc_mean': val_unseen_top1_acc.result(),
                'val_unseen_top5_acc': val_unseen_top5_acc.per_class_results_mapped(label_mapper),
                'val_unseen_top5_acc_mean': val_unseen_top5_acc.result(),
            }
            if config.CadaVAE.generalized_zsl:
                results['val_seen_top1_acc'] = val_seen_top1_acc.per_class_results_mapped(label_mapper)
                results['val_seen_top1_acc_mean'] = val_seen_top1_acc.result()
                results['val_seen_top5_acc'] = val_seen_top5_acc.per_class_results_mapped(label_mapper)
                results['val_seen_top5_acc_mean'] = val_seen_top5_acc.result()

                # Calc harmonic mean (seen + unseen)
                assert len(all_train_classes) + len(val_classes) == num_required_predict_classes
                hmean_top1 = scipy.stats.hmean(
                    [val_seen_top1_acc.result_expected_classes(len(all_train_classes)),
                     val_unseen_top1_acc.result_expected_classes(len(val_classes))]
                )
                hmean_top5 = scipy.stats.hmean(
                    [val_seen_top5_acc.result_expected_classes(len(all_train_classes)),
                     val_unseen_top5_acc.result_expected_classes(len(val_classes))]
                )
                results['hmean_top1'] = hmean_top1
                results['hmean_top5'] = hmean_top5

                with exp_env.tf_writer.as_default():
                    tf.summary.scalar('hmean_top1', hmean_top1, step=clf_aggregated_epochs)
                    tf.summary.scalar('hmean_top5', hmean_top5, step=clf_aggregated_epochs)

            if config.CadaVAE.generalized_zsl:
                is_best_score = hmean_top1 > best_score
            else:
                is_best_score = val_unseen_top1_acc.result() > best_score
            if is_best_score:
                if config.CadaVAE.generalized_zsl:
                    logging.info(f'Found new best model! vae_epoch: {vae_epoch}, clf_epoch: {clf_epoch}.'
                                 f' hmean_top1: {hmean_top1}')
                    best_score = hmean_top1
                else:
                    logging.info(f'Found new best model! vae_epoch: {vae_epoch}, clf_epoch: {clf_epoch}.'
                                 f' val_unseen_top1_acc: {val_unseen_top1_acc.result()}')
                    best_score = val_unseen_top1_acc.result().numpy()

                with open(exp_env.run_dir / 'best_results.pkl', 'wb') as f:
                    pickle.dump(results, f)

                cada_vae.save_weights(str(exp_env.checkpoints_dir / 'cada_vae_best_model'), overwrite=True)
                classifier.save_weights(str(exp_env.checkpoints_dir / 'linear_classifier_best_model'), overwrite=True)

    logging.info('Finished!')


if __name__ == '__main__':
    main(parse_args())
