import gin
import tensorflow as tf
from sklearn.utils import resample
import numpy as np

@gin.configurable
def preprocess(signal, label):
    """Dataset preprocessing: Normalizing and resizing"""

    # Normalize image: `uint8` -> `float32`.
    signal= tf.cast(signal, tf.float32)
    label = tf.cast(label, tf.int32)

    return signal, label

def augment(signal, label):
    """Data augmentation"""

    return signal, label