import tensorflow as tf
from sklearn.metrics import confusion_matrix

class BalancedAccuracy(tf.keras.metrics.Metric):
    def __init__(self, num_classes=2, name='balanced_accuracy', debug=False, **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.debug = debug
        self.true_positives = self.add_weight(
            shape=(num_classes,), initializer='zeros', name='tp'
        )
        self.false_negatives = self.add_weight(
            shape=(num_classes,), initializer='zeros', name='fn'
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert one-hot -> indices
        if y_true.shape.ndims > 1 and y_true.shape[-1] == self.num_classes:
            y_true = tf.argmax(y_true, axis=-1)
        if y_pred.shape.ndims > 1 and y_pred.shape[-1] == self.num_classes:
            y_pred = tf.argmax(y_pred, axis=-1)

        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)

        # Mask: True Positives
        tp_mask = tf.equal(y_true, y_pred)
        tp_counts = tf.math.bincount(
            tf.boolean_mask(y_true, tp_mask),
            minlength=self.num_classes,
            maxlength=self.num_classes,
            dtype=self.dtype
        )

        # Mask: False Negatives (true class i, predicted != i)
        fn_mask = tf.not_equal(y_true, y_pred)
        fn_counts = tf.math.bincount(
            tf.boolean_mask(y_true, fn_mask),
            minlength=self.num_classes,
            maxlength=self.num_classes,
            dtype=self.dtype
        )

        # Update counters
        self.true_positives.assign_add(tp_counts)
        self.false_negatives.assign_add(fn_counts)

    def result(self):
        recalls = self.true_positives / (self.true_positives + self.false_negatives + 1e-7)
        if self.debug:
            tf.print("\nTP:", self.true_positives, summarize=-1)
            tf.print("FN:", self.false_negatives, summarize=-1)
            tf.print("Recalls:", recalls, summarize=-1)
        return tf.reduce_mean(recalls)

    def reset_states(self):
        self.true_positives.assign(tf.zeros_like(self.true_positives))
        self.false_negatives.assign(tf.zeros_like(self.false_negatives))
