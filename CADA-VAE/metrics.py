import tensorflow as tf
import numpy as np


class TopKPerClassAccuracy:

    def __init__(self, num_classes, top_k=1, name='per_class_accuracy'):
        self.name = name
        self.num_classes = num_classes
        self.k = top_k

        self.topk_class_acc = [tf.keras.metrics.SparseTopKCategoricalAccuracy(k=self.k) for _ in range(self.num_classes)]

        self.cached_results = None

    def update_state(self, y_true, pred_logits):

        self.cached_results = None

        for c in np.unique(y_true):
            c_indices = tf.where(y_true == c)[:, 0]
            c_y_true = tf.gather(y_true, c_indices)
            c_pred_logits = tf.gather(pred_logits, c_indices)

            self.topk_class_acc[c].update_state(y_true=c_y_true, y_pred=c_pred_logits)

    def per_class_result(self):
        return [self.topk_class_acc[c].result() for c in range(self.num_classes) if self.topk_class_acc[c].count >= 1]

    def num_observed_classes(self):
        return len([x for x in self.topk_class_acc if x.count >= 1])

    def result_expected_classes(self, num_required_classes):
        # Account for classes that were required to be predicted but were not.
        # Assume acc=0.0 on all such classes to adjust per class accuracy.
        assert self.num_observed_classes() <= num_required_classes

        acc = self.result()
        return acc * (self.num_observed_classes() / num_required_classes)

    def per_class_results_mapped(self, label_mapper):
        c_results = self.per_class_result()
        class_names = label_mapper.ids_to_names(range(self.num_classes))
        return {c: r.numpy() for c, r in zip(class_names, c_results)}

    def result(self):
        if self.cached_results is None:
            self.cached_results = tf.reduce_mean(self.per_class_result())
        return self.cached_results

    def reset_states(self):
        for c_m in self.topk_class_acc:
            c_m.reset_states()
