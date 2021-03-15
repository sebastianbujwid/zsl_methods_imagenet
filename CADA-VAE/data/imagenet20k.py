import numpy as np
import tensorflow as tf
from array import array


class ImageNet20K:

    def __init__(self, resnet_feats_dir, imagenet_id_details):
        self.resnet_feats_dir = resnet_feats_dir
        self.imagenet_id_details = imagenet_id_details

    def get_img_feats(self, imagenet_ids, max_batch_size):

        img_feats_data = tf.data.Dataset.from_generator(
            lambda: self.img_feats_generator(imagenet_ids,
                                             max_batch_size),
            output_types=(tf.float32, tf.string),
            output_shapes=(tf.TensorShape([None, None]), tf.TensorShape([None]))
        )

        return img_feats_data.prefetch(10)

    def img_feats_generator(self, imagenet_ids, max_batch_size):

        for imagenet_id in imagenet_ids:
            feats = self._read_feats(imagenet_id)
            wnid, _ = self.imagenet_id_details[imagenet_id]

            from_index = 0
            to_index = max_batch_size
            while from_index < len(feats):
                batch_feats = feats[from_index:to_index]
                batch_size = batch_feats.shape[0]

                batch_wnids = [wnid] * batch_size

                yield batch_feats, batch_wnids

                from_index = to_index
                to_index += batch_size

        return

    def _read_feats(self, imagenet_id):
        with open(self.resnet_feats_dir / f'{imagenet_id}.bin', 'rb') as f:
            num_samples = int.from_bytes(f.read(4), byteorder='little')
            dim = int.from_bytes(f.read(4), byteorder='little')
            data = array('d')
            data.fromfile(f, num_samples * dim)
        x = np.asarray(data).reshape(num_samples, dim)
        return x.astype(np.float32)
