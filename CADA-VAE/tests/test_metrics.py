import unittest
import tensorflow as tf

from metrics import TopKPerClassAccuracy


class TestTop2PerClassAccuracy(unittest.TestCase):

    def setUp(self) -> None:
        self.metric = TopKPerClassAccuracy(num_classes=4, top_k=2)
        self.y_true = tf.constant([0, 1, 3, 0, 0, 3, 3])

    def test_acc(self):
        pred_logits = tf.constant([
            [0.0, 0.1, 0.3, 0.5],
            [0.0, 0.9, 0.3, 0.5],
            [0.0, 0.49, 0.52, 0.5],
            [0.0, 0.1, 0.3, 0.5],
            [0.0, 0.1, 0.3, 0.5],
            [0.0, 0.51, 0.52, 0.5],
            [0.0, 0.51, 0.52, 0.5],
        ])
        self.metric.update_state(y_true=self.y_true, pred_logits=pred_logits)
        per_class_results = self.metric.per_class_result()
        self.assertEqual(per_class_results,
                         [0., 1., 1/3])
        res = self.metric.result()
        self.assertEqual(res, (4/3) / 3)


if __name__ == '__main__':
    unittest.main()
