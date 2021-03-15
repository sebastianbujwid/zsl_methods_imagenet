from typing import Sequence
import tensorflow as tf


def process_name(name):
    if isinstance(name, str):
        return name
    elif isinstance(name, tf.Tensor):
        return name.numpy().decode('utf-8')
    else:
        raise NotImplementedError()


class LabelMapper:

    def __init__(self, all_class_names: Sequence):
        # NOTE: preserves the order of class_names!
        self.all_class_names = all_class_names
        self.num_classes = len(self.all_class_names)

        self._id_to_name = [name for name in self.all_class_names]
        self._name_to_id = {self._id_to_name[i]: i for i in range(len(self._id_to_name))}

    def ids_to_names(self, ids):
        return [self._id_to_name[i] for i in ids]

    def get_all_ordered_names(self):
        return self._id_to_name

    def names_to_ids(self, names):
        return [self._name_to_id[process_name(n)] for n in names]
