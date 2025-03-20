import gin
import tensorflow as tf


@gin.configurable
def dense_layer(inputs, dense_units):
    """A single dense layer

    Parameters:
        inputs (Tensor): input of the dense layer
        dense_units (int): number of filters used for the dense layer

    Returns:
        (Tensor): output of the single dense layer
    """

    outputs = tf.keras.layers.Dense(dense_units, activation='linear')(inputs)
    outputs = tf.keras.layers.BatchNormalization()(outputs)
    outputs = tf.keras.layers.Activation('tanh')(outputs)
    return outputs


@gin.configurable
def gru_layer(inputs, dense_units, return_sequences=True):
    """
    A GRU block consisting of a GRU layer followed by Batch Normalization and tanh activation.

    Parameters:
        inputs (Tensor): Input tensor.
        dense_units (int): Number of GRU units.
        return_sequences (bool): Whether to return the full sequence or only the last output.

    Returns:
        Tensor: Output tensor after applying GRU, BatchNorm, tanh activation.
    """
    outputs = tf.keras.layers.GRU(dense_units, return_sequences=return_sequences)(inputs)
    outputs = tf.keras.layers.BatchNormalization()(outputs)
    outputs = tf.keras.layers.Activation('tanh')(outputs)
    return outputs

def LSTM_layer(inputs,filters,return_seq):
    """
    A LSTM block consisting of a LSTM layer

    Parameters:
        inputs (Tensor): Input tensor.
        filters (int): Number of LSTM units.
        return_sequences (bool): Whether to return the full sequence or only the last output.

    Returns:
        Tensor: Output tensor after applying GRU, BatchNorm, tanh activation.
    """
    output = tf.keras.layers.LSTM(filters,return_sequences = return_seq, activation='tanh')(inputs)
    return output



