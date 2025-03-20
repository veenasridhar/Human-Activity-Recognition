import tensorflow as tf

import tensorflow as tf


# Custom metric to compute the confusion matrix
class ConfusionMatrix(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name="confusion_matrix", **kwargs):
        super(ConfusionMatrix, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        # Initialize confusion matrix as a zero matrix of shape (num_classes, num_classes)
        self.conf_matrix = self.add_weight(
            name='confusion_matrix',
            shape=(num_classes, num_classes),
            initializer='zeros',
            dtype=tf.int32
        )

    def update_state(self, labels, predictions, sample_weight=None):
        # Flatten and cast labels and predictions to integer type
        labels = tf.reshape(tf.cast(labels, dtype=tf.int32), [-1])
        predictions = tf.reshape(tf.cast(predictions, dtype=tf.int32), [-1])

        # Compute the confusion matrix for the current batch
        new_conf_matrix = tf.math.confusion_matrix(
            labels,
            predictions,
            num_classes=self.num_classes,
            dtype=tf.int32
        )
        # Update the overall confusion matrix
        self.conf_matrix.assign_add(new_conf_matrix)

    def result(self):
        # Return the confusion matrix
        return self.conf_matrix

    def reset_state(self):
        # Reset the confusion matrix to all zeros
        self.conf_matrix.assign(tf.zeros((self.num_classes, self.num_classes), dtype=tf.int32))


# Custom accuracy metric
class Accuracy(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name="accuracy", **kwargs):
        super(Accuracy, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        # Initialize counters for correct predictions and total predictions
        self.correct_predictions = self.add_weight(name="correct", initializer="zeros", dtype=tf.float32)
        self.total_predictions = self.add_weight(name="total", initializer="zeros", dtype=tf.float32)

    def update_state(self, labels, predictions, sample_weight=None):
        # Flatten and cast labels and predictions to integer type
        labels = tf.reshape(tf.cast(labels, dtype=tf.int32), [-1])
        predictions = tf.reshape(tf.cast(predictions, dtype=tf.int32), [-1])

        # Compute the number of correct predictions
        correct = tf.reduce_sum(tf.cast(tf.math.equal(labels, predictions), dtype=tf.float32))
        total = tf.cast(tf.size(labels), dtype=tf.float32)

        # Update the counters for correct and total predictions
        self.correct_predictions.assign_add(correct)
        self.total_predictions.assign_add(total)

    def result(self):
        # Compute and return the accuracy as a percentage
        return tf.math.divide_no_nan(self.correct_predictions, self.total_predictions) * 100

    def reset_state(self):
        # Reset the counters to zero
        self.correct_predictions.assign(0)
        self.total_predictions.assign(0)
