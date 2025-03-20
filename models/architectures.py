import gin
import tensorflow as tf
from models.layers import *

@gin.configurable
def gru_model(input_shape, n_classes, dropout_rate, dense_unit):
    """
    A GRU-based model for sequence-to-label tasks.

    Parameters:
        input_shape (tuple): Shape of the input data.
        n_classes (int): Number of output classes.
        dropout_rate (float): Dropout rate for regularization.
        dense_unit (int): Number of units in dense layers. Dense_unit is used by dividing numbers for ease during hyperparameter tuning

    Returns:
        tf.keras.Model: Compiled GRU model.
    """
    inputs = tf.keras.Input(input_shape)
    outputs = gru_layer(inputs, dense_unit)
    outputs = dense_layer(outputs, dense_unit//2)
    outputs = tf.keras.layers.Dropout(dropout_rate)(outputs)
    outputs = gru_layer(outputs, dense_unit//2)
    outputs = dense_layer(outputs, dense_unit//4)
    outputs = tf.keras.layers.Dropout(dropout_rate)(outputs)
    outputs = gru_layer(outputs, dense_unit//4, return_sequences=False)
    outputs = dense_layer(outputs, dense_unit//8)
    outputs = tf.keras.layers.Dropout(dropout_rate)(outputs)
    outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(outputs)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='sequence2label_GRU_model')

@gin.configurable
def ConvLSTMseq2seq(input_shape,n_classes):
    """
    A ConvLSTM-based sequence-to-sequence model.

    Parameters:
        input_shape (tuple): Shape of the input data.
        n_classes (int): Number of output classes.

    Returns:
        tf.keras.Model: Compiled ConvLSTM sequence-to-sequence model.
    """
    inputs = tf.keras.layers.Input(input_shape)
    conv1 = tf.keras.layers.Conv1D(64,5,1,activation='relu',padding="same")(inputs)
    maxpool1 = tf.keras.layers.MaxPool1D()(conv1)
    output1 = tf.keras.layers.Dropout(0.2)(maxpool1)
    conv2 = tf.keras.layers.Conv1D(128,5,1,activation='relu',padding="same")(output1)
    maxpool2 = tf.keras.layers.MaxPool1D()(conv2)
    output2 = tf.keras.layers.Dropout(0.2)(maxpool2)
    lstm = tf.keras.layers.LSTM(64,return_sequences=True,return_state=True)
    enc_out,enc_h,enc_c = lstm(output2)
    decoder_inputs = tf.keras.layers.RepeatVector(128)(enc_h)                           # Repeat encoded context vector
    decoder_lstm = tf.keras.layers.LSTM(64, return_sequences=True)(decoder_inputs)
    outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_classes,activation='softmax'))(decoder_lstm)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='seq2seq_model')


@gin.configurable
def lstm_model(input_shape, n_classes):
    """
    Creates an LSTM-based sequence-to-label model.

    Parameters:
        input_shape (tuple): Shape of the input data.
        n_classes (int): Number of output classes.

    Returns:
        tf.keras.Model: Compiled LSTM model.
    """
    inputs =tf.keras.layers.Input(input_shape)
    lstm1 = LSTM_layer(inputs,64, return_seq=True)
    lstm2 = LSTM_layer(lstm1, 64,return_seq=False)
    dense  = tf.keras.layers.Dense(64, activation=tf.nn.relu)(lstm2)
    dropout = tf.keras.layers.Dropout(0.2)(dense)
    outputs = tf.keras.layers.Dense(n_classes, activation=tf.nn.softmax)(dropout)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='seq2labelLSTM')
   


