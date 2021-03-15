import argparse
import pickle
import logging
import numpy as np
import tensorflow as tf
from pathlib import Path
from collections import defaultdict

from definitions import ROOT_DIR
import evaluation
from utils import SimpleEnv
from model import CadaVAE, LinearClassifier
from data import LabelMapper
from data import ImageNet1K, ImageNet20K
from data import load_imagenet1k_data
from data import get_imagenet1k_train_val_class_splits, get_imagenet1k_trainval_classes
from data.imagenet_utils import extract_imagenet_id_details

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(module)20.20s.%(funcName)20.20s -:- %(message)s',
                    level=logging.DEBUG)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--env_config', type=Path, default=ROOT_DIR / 'CADA-VAE' / 'configs' / 'env_config.yml')
    parser.add_argument('--cada_vae_weights', required=True, type=Path)
    parser.add_argument('--eval_set', required=True,
                        choices=['val', 'mp500'])
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed should have no effect')
    parser.add_argument('--overwrite', nargs='+')
    return parser.parse_args()


def weights_exist(path: Path):
    if len(list(path.parent.glob(path.name + '.data-*'))) == 0:
        return False
    else:
        return True


def create_dummy_aux(imagenet_id_details):
    aux_feats = {
        i: {
            'feats': np.array([np.nan]),
            'wnid': imagenet_id_details[i][0]
        }
        for i in range(1, 1001)
    }
    return aux_feats


def eval_confusion_matrix(
        img_label_data,
        classifier: LinearClassifier,
        encode_func,
        label_mapper: LabelMapper):

    confusion_matrix = {wnid: defaultdict(int) for wnid in label_mapper.get_all_ordered_names()}

    pbar = tf.keras.utils.Progbar(None, unit_name='samples')
    for img, label in img_label_data:
        pred_logits = evaluation.predict(img, classifier, encode_func)

        pred_logits = pred_logits.numpy()
        pred_labels = pred_logits.argmax(axis=-1)
        pred_wnids = label_mapper.ids_to_names(pred_labels)

        true_wnids = label_mapper.ids_to_names(label_mapper.names_to_ids(label))
        for true_wnid, pred_wnid in zip(true_wnids, pred_wnids):
            confusion_matrix[true_wnid][pred_wnid] += 1

        pbar.add(img.shape[0])

    return confusion_matrix


def per_class_accuracy(confusion_matrix):
    class_acc = []
    for wnid, preds in confusion_matrix.items():
        acc = preds[wnid] / sum(preds.values())
        class_acc.append(acc)

    mean_acc = sum(class_acc) / len(class_acc)
    return mean_acc


def main(args):
    if args.seed:
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)

    cada_vae_weights = args.cada_vae_weights
    eval_set = args.eval_set

    weights_dir = cada_vae_weights.parent

    if not weights_exist(cada_vae_weights):
        raise ValueError(f'Cannot find {cada_vae_weights}')

    if eval_set != 'val':
        linear_classifier_weights = cada_vae_weights.with_name(
            cada_vae_weights.name.replace('cada_vae_model', f'linear_classifier_{eval_set}')
        )
    else:
        linear_classifier_weights = cada_vae_weights.with_name(
            cada_vae_weights.name.replace('cada_vae', 'linear_classifier')
        )
    if not weights_exist(linear_classifier_weights):
        raise ValueError(f'Cannot find {linear_classifier_weights}')

    output_confusion_matrix_file = cada_vae_weights.with_name('confusion_matrix.pkl')
    if output_confusion_matrix_file.exists():
        raise ValueError(f'The output file already exists: {output_confusion_matrix_file.absolute()}')

    label_mapper = pickle.load(open(cada_vae_weights.with_name('label_mapper.pkl'), 'rb'))

    exp_env = SimpleEnv(args, output_dir=weights_dir,
                        configs=[args.config, args.env_config])
    config = exp_env.config

    imagenet_id_details = extract_imagenet_id_details(config.env.imagenet_w2v_extra_pkl)
    imagenet1k = ImageNet1K(hdf5_file=config.env.imagenet_1k,
                            aux_feats_file=None,
                            aux_feats=create_dummy_aux(imagenet_id_details),    # aux_feats not used
                            load_to_memory=False)

    if eval_set == 'val':
        _, val_classes = get_imagenet1k_train_val_class_splits(config)
        eval_classes = val_classes
    else:
        _, test_classes = get_imagenet1k_trainval_classes(config, test_split=eval_set)
        eval_classes = test_classes

    eval_classes_with_aux = {i for i in eval_classes
                             if imagenet_id_details[i][0] in set(label_mapper.get_all_ordered_names())}
    assert len(eval_classes_with_aux) == label_mapper.num_classes

    if config.CadaVAE.generalized_zsl:
        raise NotImplementedError()

    cada_vae = CadaVAE(config=config.CadaVAE)
    classifier = LinearClassifier(config=config.CadaVAE.LinearClassifier, num_classes=label_mapper.num_classes)

    logging.info('Loading weights... CadaVAE')
    cada_vae.load_weights(str(cada_vae_weights))
    logging.info('Loading weights... LinearClassifier')
    classifier.load_weights(str(linear_classifier_weights))

    if eval_set == 'val':
        imagenet_1k_data = load_imagenet1k_data(imagenet1k, batch_size=config.CadaVAE.VAE.batch_size,
                                                seen_classes=set(),
                                                unseen_classes=eval_classes_with_aux)
        _, _, val_unseen_img_aux_label_data = imagenet_1k_data
        eval_unseen_img_label_data = val_unseen_img_aux_label_data.map(
            lambda img, aux, label: (img, label)
        )
    else:
        imagenet20k = ImageNet20K(Path(config.env.imagenet_20k_dir),
                                  imagenet_id_details=imagenet_id_details)
        test_unseen_img_label_data = imagenet20k.get_img_feats(imagenet_ids=eval_classes_with_aux,
                                                               max_batch_size=32)
        eval_unseen_img_label_data = test_unseen_img_label_data

    confusion_matrix = eval_confusion_matrix(
        eval_unseen_img_label_data,
        classifier,
        encode_func=cada_vae.img_feat_vae.encode,
        label_mapper=label_mapper,
    )

    acc = per_class_accuracy(confusion_matrix)
    logging.info(f'Mean top-1 per-class acc: {acc}')
    pickle.dump(confusion_matrix, open(output_confusion_matrix_file, 'wb'))


if __name__ == '__main__':
    main(parse_args())
