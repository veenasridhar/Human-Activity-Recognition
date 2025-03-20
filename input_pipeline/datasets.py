from random import shuffle

import gin
import logging
import tensorflow as tf
import os
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler


@gin.configurable
def create_sliding_windows(activity_data,activity_label, window_size, overlap):
    X, y = [], []
    step = int(window_size * (1 - overlap))  # step size calculation
    for i in range(0, len(activity_data) - window_size + 1, step):
        window = activity_data[i:i + window_size]
        if len(np.unique(activity_label[i:i + window_size])) == 1:
            curr_label = activity_label[i]
            X.append(window)
            y.append(curr_label)
    return np.array(X), np.array(y)

def create_datasets(data,labels):
  X_train,y_train = [],[]
  for exp_data, exp_labels in zip(data, labels):
    X_exp,y_exp = [],[]
    for _, row in exp_labels.iterrows():
        activity_data = exp_data[row.starting:row.ending]
        activity_label = row.activity_id
        X_exp.append(activity_data)
        y_exp.append([activity_label]*len(activity_data))
    X_train.append(np.concatenate(X_exp))
    y_train.append(np.concatenate(y_exp))
  return np.concatenate(X_train),np.concatenate(y_train)

@gin.configurable
def create_seq2seq_windows(activity_data,activity_label,window_size,overlap):
     X_features, y_activities = [], []
     step = int(window_size * (1 - overlap))  # step size calculation
     for i in range(0, len(activity_data) - window_size + 1, step):
        window = activity_data[i:i + window_size]
        X_features.append(window)
        label_window = activity_label[i:i + window_size]
        y_activities.append(label_window)
     return np.array(X_features), np.array(y_activities)

@gin.configurable
def load(model_type,name, data_dir,window_size,overlap):
    if name == "hapt":
        logging.info(f"Preparing dataset {name}...")
        folder_name = 'TFRecords'+model_type
        if not os.path.exists(os.path.join(os.getcwd(),folder_name)):
            accelerometer_files = [files for files in os.listdir(data_dir) if files.startswith("acc") == True]
            gyroscope_files = [files for files in os.listdir(data_dir) if files.startswith("gyro") == True]
            accelerometer_files.sort()
            gyroscope_files.sort()
            total_experiments = range(1, 62)

            scaler = StandardScaler()
            X_train = []
            X_val = []
            X_test = []


            for (accfile, gyrofile, expid) in zip(accelerometer_files, gyroscope_files, total_experiments):
                acc_data = pd.read_csv(data_dir + accfile, sep=' ', header=None)
                gyro_data = pd.read_csv(data_dir + gyrofile, sep=' ', header=None)
                X = np.concatenate([acc_data, gyro_data],axis=1)
                if expid <= 43:
                    X_train.append(X)
                elif expid >= 56 and expid < 62:
                    X_val.append(X)
                else:
                    X_test.append(X)

            labels_data = pd.read_csv(data_dir + "labels.txt", sep=" ", header=None)
            labels_data.columns = ["experiment_id", "user_id", "activity_id", "starting", "ending"]
            train_indices = range(1, 44)
            val_indices = range(56, 62)
            test_indices = range(44, 56)

            y_train = labels_data[labels_data.experiment_id.isin(train_indices)]
            y_val = labels_data[labels_data.experiment_id.isin(val_indices)]
            y_test = labels_data[labels_data.experiment_id.isin(test_indices)]

            y_train = [group for _, group in y_train.groupby("experiment_id")]
            y_val = [group for _, group in y_val.groupby("experiment_id")]
            y_test = [group for _, group in y_test.groupby("experiment_id")]

            X_train_features,y_train_labels = create_datasets(X_train,y_train)
            X_val_features,y_val_labels = create_datasets(X_val,y_val)
            X_test_features,y_test_labels = create_datasets(X_test,y_test)

            #Normalizing the Data
            scaler.fit(X_train_features)
            X_train_normalized = scaler.transform(X_train_features)
            X_val_normalized = scaler.transform(X_val_features)
            X_test_normalized = scaler.transform(X_test_features)


            if model_type == 'seq2label':
                X_train, y_train = create_sliding_windows(X_train_normalized, y_train_labels, window_size, overlap)
                X_val, y_val = create_sliding_windows(X_val_normalized, y_val_labels, window_size, overlap)
                X_test, y_test = create_sliding_windows(X_test_normalized, y_test_labels, window_size, overlap)

        
            elif model_type == 'seq2seq':
                X_train, y_train = create_seq2seq_windows(X_train_normalized, y_train_labels, window_size, overlap)
                X_val, y_val = create_seq2seq_windows(X_val_normalized, y_val_labels, window_size, overlap)
                X_test, y_test = create_seq2seq_windows(X_test_normalized, y_test_labels, window_size, overlap)


            writeTFRecords('train.tfrecord',X_train,y_train,model_type)
            writeTFRecords('val.tfrecord',X_val,y_val,model_type)
            writeTFRecords('test.tfrecord',X_test,y_test,model_type)


        ds_info = {
                "num_classes": 12,  # number of classes in HAPT, for instance
                "class_names": ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING",
                                "STAND_TO_SIT", "SIT_TO_STAND", "SIT_TO_LIE", "LIE_TO_SIT", "STAND_TO_LIE", "LIE_TO_STAND"],
                "description": "HAPT dataset",
                "input_shape": (window_size, 6)  # (timesteps, features)
            }

        print(os.path.join(folder_name,'train.tfrecord'))
        train_dataset = tf.data.TFRecordDataset(os.path.join(folder_name,'train.tfrecord'))
        ds_train = train_dataset.map(_parse_function)
        val_dataset = tf.data.TFRecordDataset(os.path.join(folder_name,'val.tfrecord'))
        ds_val = val_dataset.map(_parse_function)
        test_dataset = tf.data.TFRecordDataset(os.path.join(folder_name,'test.tfrecord'))
        ds_test = test_dataset.map(_parse_function)


        def convert_to_zero_based(x, y):
            return x, y - 1

        ds_train = ds_train.map(convert_to_zero_based)
        ds_val = ds_val.map(convert_to_zero_based)
        ds_test = ds_test.map(convert_to_zero_based)   
        
        return prepare(ds_train,ds_val,ds_test,ds_info,window_size)


@gin.configurable
def prepare(ds_train, ds_val, ds_test, ds_info, batch_size, caching, window_size):
    # Apply the preprocessing function
    ds_train = ds_train.shuffle(buffer_size=1000)
    ds_train = ds_train.batch(batch_size, drop_remainder=False)
    ds_train = ds_train.repeat()
    ds_train = ds_train.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    ds_val = ds_val.batch(batch_size, drop_remainder=False)   
    ds_val = ds_val.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    
    ds_test = ds_test.batch(batch_size, drop_remainder=False)
    ds_test = ds_test.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return ds_train, ds_val, ds_test, ds_info, window_size

def serialize_data(X_features,y_labels):
    ser_x_data = tf.io.serialize_tensor(X_features)         #to handle numpy arrays
    ser_y_data = tf.io.serialize_tensor(y_labels)
    features = {
        'x': tf.train.Feature(bytes_list= tf.train.BytesList(value = [ser_x_data.numpy()] )),
        'label': tf.train.Feature(bytes_list = tf.train.BytesList(value = [ser_y_data.numpy()]))
    }
    example_proto = tf.train.Example(features = tf.train.Features(feature=features))
    return example_proto.SerializeToString()

def writeTFRecords(filename,X_features,y_labels,model_type):    
    folder_name = 'TFRecords'+model_type
    if not os.path.exists(os.path.join(os.getcwd(),folder_name)):
        os.mkdir(folder_name)
    filename = os.path.join(os.getcwd(),folder_name,filename)
    with tf.io.TFRecordWriter(filename) as writer:
        for i in range(len(X_features)):
            serialized_example = serialize_data(X_features[i],y_labels[i])
            writer.write(serialized_example)

def _parse_function(example_proto):
    features = {
        'x': tf.io.FixedLenFeature([],tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),       }
    temp = tf.io.parse_single_example(example_proto,features)
    x = tf.io.parse_tensor(temp['x'],tf.float64)
    y = tf.io.parse_tensor(temp['label'],tf.int64)
    return x,y




