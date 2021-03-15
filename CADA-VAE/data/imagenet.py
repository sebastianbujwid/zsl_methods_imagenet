import h5py
import pickle
import logging
import numpy as np
import tensorflow as tf

from model import VAE
from data import LabelMapper


# TODO - add unit tests!
def select_indices(labels, classes):
    assert labels.ndim == 1
    if isinstance(classes, set):
        classes = np.array(list(classes))
    mask = np.isin(labels, test_elements=classes)
    indices = np.where(mask)[0].astype(np.int64)
    return indices


def oversample_array(a, num_samples):
    assert len(a) < num_samples
    num_times_whole_array = len(a) // num_samples
    out = np.concatenate([a for _ in range(num_times_whole_array)])
    num_remaining_samples = num_samples - num_times_whole_array * len(a)
    if num_remaining_samples > 0:
        out = np.concatenate([out, np.random.choice(a, num_remaining_samples, replace=False)])

    assert len(out) == num_samples
    return out


class ImageNet1K:

    def __init__(self, hdf5_file, aux_feats_file, load_to_memory, aux_feats=None):
        self.hdf5_file = hdf5_file
        if aux_feats_file:
            with open(aux_feats_file, 'rb') as f:
                self.aux_feats = pickle.load(f)  # TODO - what if it is really big? 20K? Check w2v size!
        else:
            if not aux_feats:
                raise ValueError('Either "aux_feats_file" or "aux_feats" needs to be passed')
            self.aux_feats = aux_feats

        self.aux_feats_dim = next(iter(self.aux_feats.values()))['feats'].shape[0]
        self.img_feats_dim = 2048

        self._features = None
        self._features_val = None
        self._labels = None
        self._labels_val = None
        if load_to_memory:
            self._cache_data()

    def _cache_data(self):
        logging.info(f'Caching img features...')
        with h5py.File(self.hdf5_file, 'r') as f:
            self._labels = np.array(f['labels'], dtype=np.int64).flatten()
            self._labels_val = np.array(f['labels_val'], dtype=np.int64).flatten()

            self._features = np.array(f['features']).T
            self._features_val = np.array(f['features_val'])

        logging.info(f'Finished caching img features')

    def get_labels(self, split, h5py_file):
        if split == 'train':
            if self._labels is None:
                l = h5py_file['labels']
            else:
                l = self._labels
        elif split == 'val':
            if self._labels_val is None:
                l = h5py_file['labels_val']
            else:
                l = self._labels_val
        elif split == 'trainval':
            raise NotImplementedError()  # Concat labels?
        else:
            raise ValueError()

        labels = np.array(l, dtype=np.int64).flatten()
        assert labels.min() == 1 and labels.max() == 1000
        return labels

    def get_features(self, indices, split, h5py_file):
        if split == 'train':
            if self._features is None:
                feats = h5py_file['features'][:, indices].T
            else:
                feats = self._features[indices, :]
        elif split == 'val':
            if self._features_val is None:
                feats = h5py_file['features_val'][indices, :]
            else:
                feats = self._features_val[indices, :]
        elif split == 'train_val':
            raise NotImplementedError()   # Assumed concat feats [train, val]
        else:
            raise ValueError()

        return feats

    def get_imagnet_ids_with_aux(self):
        return set(self.aux_feats.keys())

    def get_aux_features(self, imagnet_ids):
        return np.array([self.aux_feats[c]['feats'] for c in imagnet_ids])

    def get_aux_wnid(self, imagenet_ids):
        return np.array([self.aux_feats[c]['wnid'] for c in imagenet_ids])

    def get_aux_feats_with_labels(self, imagenet_ids,
                                  label_mapper: LabelMapper = None):
        if isinstance(imagenet_ids, set):
            imagenet_ids = list(imagenet_ids)

        labels = self.get_aux_wnid(imagenet_ids)
        if label_mapper:
            labels = label_mapper.names_to_ids(labels)
        att_feats = self.get_aux_features(imagenet_ids)
        return att_feats, labels

    def img_aux_label_generator(self,
                                split,
                                batch_size,
                                allow_only_classes=None,  # ImageNet ID: [1, 1000]
                                ):
        with h5py.File(self.hdf5_file, 'r') as f:
            labels = self.get_labels(split=split, h5py_file=f)

            indices = select_indices(labels, allow_only_classes)
            np.random.shuffle(indices)

            from_index = 0
            to_index = batch_size
            while from_index < len(indices):
                batch_indices = sorted(indices[from_index:to_index])

                batch_feats = self.get_features(batch_indices, split, h5py_file=f)
                batch_imagenet_id_labels = labels[batch_indices]
                batch_aux_feats = self.get_aux_features(imagnet_ids=batch_imagenet_id_labels)
                batch_wnid_labels = self.get_aux_wnid(imagenet_ids=batch_imagenet_id_labels)

                yield batch_feats, batch_aux_feats, batch_wnid_labels

                from_index = to_index
                to_index += batch_size
        return

    def sample_per_class_generator(self, split, classes,
                                   label_mapper: LabelMapper,
                                   num_samples_per_class, batch_size_iterable,
                                   oversample=True):

        with h5py.File(self.hdf5_file, 'r') as f:
            labels = self.get_labels(split, h5py_file=f)

            class_indices = []
            for _class in classes:
                c_indices = select_indices(labels, classes={_class})
                if len(c_indices) >= num_samples_per_class:
                    rand_indices = np.random.choice(c_indices, num_samples_per_class, replace=False)
                else:
                    if oversample:
                        rand_indices = oversample_array(c_indices, num_samples_per_class)
                    else:
                        raise NotImplementedError()
                class_indices.append(rand_indices)

            all_indices = np.concatenate(class_indices)
            np.random.shuffle(all_indices)

            from_index = 0
            for batch_size in batch_size_iterable:
                to_index = from_index + batch_size

                batch_indices = sorted(all_indices[from_index:to_index])
                batch_feats = self.get_features(batch_indices, split, h5py_file=f)
                batch_imagenet_id_labels = labels[batch_indices]
                batch_wnid_labels = self.get_aux_wnid(imagenet_ids=batch_imagenet_id_labels)
                batch_labels = label_mapper.names_to_ids(batch_wnid_labels)

                yield batch_feats, batch_labels

                from_index = to_index

        assert to_index == len(all_indices)  # All samples yielded
        return

    def z_latent_labels_samples(self, generalized_zsl,
                                img_classes, aux_classes,
                                label_mapper: LabelMapper,
                                img_encode_func, aux_encode_func,
                                num_img_samples_per_class, num_aux_samples_per_class,
                                batch_size):
        total_num_samples_img_feats = len(img_classes) * num_img_samples_per_class
        total_num_samples_aux_feats = len(aux_classes) * num_aux_samples_per_class

        if generalized_zsl:
            assert num_img_samples_per_class > 0

        if generalized_zsl:
            img_feats_batch_size_iterable, aux_feats_batch_size_iterable = sample_batch_size_iters(
                num_elements_a=total_num_samples_img_feats,
                num_elements_b=total_num_samples_aux_feats,
                zip_batch_size=batch_size
            )

            img_feats_samples = tf.data.Dataset.from_generator(
                lambda: self.sample_per_class_generator(
                    split='train',
                    classes=img_classes,
                    label_mapper=label_mapper,
                    num_samples_per_class=num_img_samples_per_class,
                    batch_size_iterable=img_feats_batch_size_iterable,
                    oversample=True),
                output_types=(tf.float32, tf.int64),
                output_shapes=(tf.TensorShape([None, None]), tf.TensorShape([None]))    # TODO - specify latent_size?
            )
            img_feats_samples = img_feats_samples.map(
                lambda img, label: (img_encode_func(img, training=tf.constant(False))[0], label)
            )
        else:
            aux_feats_batch_size_iterable, _ = sample_batch_size_iters(num_elements_a=total_num_samples_aux_feats,
                                                                       num_elements_b=0,
                                                                       zip_batch_size=batch_size)

        aux_feats, aux_labels = self.get_aux_feats_with_labels(aux_classes, label_mapper)
        _, aux_feats_mu, aux_feats_logvar = aux_encode_func(aux_feats, training=tf.constant(False))
        latent_dim = tf.shape(aux_feats_mu)[-1]
        aux_feats_samples = tf.data.Dataset.from_generator(
            lambda: oversample_z_from_feats(mu=aux_feats_mu,
                                            log_var=aux_feats_logvar,
                                            labels=aux_labels,
                                            num_samples_per_feat=num_aux_samples_per_class,
                                            batch_size_iterable=aux_feats_batch_size_iterable),
            output_types=(tf.float32, tf.int64),
            output_shapes=(tf.TensorShape([None, latent_dim]), tf.TensorShape([None]))
        )

        if generalized_zsl:
            z_latent_labels_samples = tf.data.Dataset.zip((img_feats_samples, aux_feats_samples))
            z_latent_labels_samples = z_latent_labels_samples.map(
                lambda img, aux: (
                    tf.concat([img[0], aux[0]], axis=0),    # z_latent feats
                    tf.concat([img[1], aux[1]], axis=0)     # labels
                )
            )
        else:
            z_latent_labels_samples = aux_feats_samples

        return z_latent_labels_samples.prefetch(10)


def sample_batch_size_iters(num_elements_a, num_elements_b, zip_batch_size):
    total_num_elements = num_elements_a + num_elements_b
    use_a_mask = np.concatenate([np.ones(num_elements_a, dtype=np.bool),
                                 np.zeros(num_elements_b, dtype=np.bool)])
    np.random.shuffle(use_a_mask)
    split_indices = np.cumsum(np.array([zip_batch_size] * (total_num_elements // zip_batch_size)))
    use_a_mask_batches = np.split(use_a_mask, split_indices)
    assert sum(map(lambda x: x.size, use_a_mask_batches)) == total_num_elements

    batches_a, batches_b = zip(*[(np.sum(x), x.size - np.sum(x)) for x in use_a_mask_batches])
    assert len(batches_a) == len(batches_b)
    assert sum(batches_a) == num_elements_a
    assert sum(batches_b) == num_elements_b

    return batches_a, batches_b


def oversample_z_from_feats(mu, log_var, labels, num_samples_per_feat, batch_size_iterable):
    tf.assert_equal(tf.shape(mu), tf.shape(log_var))
    tf.assert_equal(tf.shape(mu)[0], tf.shape(labels)[0])

    num_samples = tf.shape(mu)[0]
    indices = np.repeat(np.arange(num_samples), num_samples_per_feat)
    np.random.shuffle(indices)
    indices = tf.convert_to_tensor(indices)

    from_index = 0
    for batch_size in batch_size_iterable:
        to_index = from_index + batch_size
        batch_indices = indices[from_index:to_index]
        batch_mu = tf.gather(mu, batch_indices)
        batch_logvar = tf.gather(log_var, batch_indices)
        batch_labels = tf.gather(labels, batch_indices)
        batch_z_latent = VAE.reparametrize(mu=batch_mu, log_var=batch_logvar)

        yield batch_z_latent, batch_labels

        from_index = to_index

    assert to_index == len(indices)  # All samples yielded
    return
