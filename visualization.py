import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from models.architectures import *
from scipy.ndimage import median_filter
import gin

def labeling(label_seq, num_categories):
        """Define the color map corresponding to each label."""
        label_color = [
            '#FFFFFF',  # Pastel White
            '#87CEFA',  # Pastel Sky Blue
            '#FFB347',  # Pastel Orange
            '#77DD77',  # Pastel Green
            '#AEC6CF',  # Pastel Blue
            '#FF6961',  # Pastel Red
            '#FDFD96',  # Pastel Yellow
            '#B0E0E6',  # Pastel Cyan
            '#C3B1E1',  # Pastel Purple
            '#C4A484',  # Pastel Brown
            '#BFFF00',  # Pastel Lime
            '#FFB6C1'  # Pastel Pink
        ]

        # Ensure `label_seq` is a 1D integer array
        label_seq = np.array(label_seq)

        start = 0
        for i in range(1, label_seq.size):
            if int(label_seq[i].item()) != int(label_seq[i - 1].item()):  # Ensure scalar conversion
                end = i - 1
                plt.axvspan(start, end, facecolor=label_color[int(label_seq[i - 1].item())], alpha=0.5)
                start = i

        plt.axvspan(start, label_seq.size - 1, facecolor=label_color[int(label_seq[-1].item())], alpha=0.5)


@gin.configurable
def visual(model, checkpoint, data, model_type, num_class, run_paths):
    manager = tf.train.CheckpointManager(checkpoint, run_paths['path_ckpts_train'], max_to_keep=5)
    checkpoint.restore(manager.latest_checkpoint)

    if manager.latest_checkpoint:
        print(f"Restored from {manager.latest_checkpoint}")
    else:
        print("No checkpoint found")

    pred_list = []
    label_list = []
    acc_x, acc_y, acc_z = [], [], []
    gyro_x, gyro_y, gyro_z = [], [], []

    for idx, (signal, labels) in enumerate(data):
        if model_type == 'seq2seq':
            predictions = model(signal, training=False)

        predictions = np.concatenate(predictions.numpy())
        predictions = tf.argmax(predictions,axis=-1)
        labels = np.concatenate(labels.numpy())

        pred_list.append(predictions)
        label_list.append(labels)

        signal = np.concatenate(signal.numpy())
        acc_x.append(signal[:, 0])
        acc_y.append(signal[:, 1])
        acc_z.append(signal[:, 2])
        gyro_x.append(signal[:, 3])
        gyro_y.append(signal[:, 4])
        gyro_z.append(signal[:, 5])

        if idx >= 0:  # Only process the first batch
            break

    # Convert lists to NumPy arrays
    acc_x, acc_y, acc_z = np.concatenate(acc_x), np.concatenate(acc_y), np.concatenate(acc_z)
    gyro_x, gyro_y, gyro_z = np.concatenate(gyro_x), np.concatenate(gyro_y), np.concatenate(gyro_z)

    preds = np.concatenate(pred_list,dtype=np.int32)
    labels = np.concatenate(label_list,dtype=np.int32)

    # Select a portion of the data for visualization
    view = [0, len(labels)]
    acc_x, acc_y, acc_z = acc_x[view[0]:view[1]], acc_y[view[0]:view[1]], acc_z[view[0]:view[1]]
    gyro_x, gyro_y, gyro_z = gyro_x[view[0]:view[1]], gyro_y[view[0]:view[1]], gyro_z[view[0]:view[1]]

    # Plot predicted labels
    plt.figure(figsize=(12, 6), dpi=150)

    ax11 = plt.subplot(2, 1, 1)
    plt.title('Prediction')
    plt.tick_params(labelsize='x-small')
    ax11.set_xlabel("Time series")
    ax11.set_ylabel("Accelerometer")
    ax12 = ax11.twinx()
    ax12.set_ylabel("Gyroscope")

    plt.plot(acc_x, label='acc_x', linewidth=1)
    plt.plot(acc_y, label='acc_y', linewidth=1)
    plt.plot(acc_z, label='acc_z', linewidth=1)
    plt.plot(gyro_x, label='gyro_x', linewidth=1)
    plt.plot(gyro_y, label='gyro_y', linewidth=1)
    plt.plot(gyro_z, label='gyro_z', linewidth=1)

    labeling(preds, num_class)

    # Plot ground truth labels
    ax21 = plt.subplot(2, 1, 2)
    plt.title('True labels')
    plt.tick_params(labelsize='x-small')
    ax21.set_xlabel("Time series")
    ax21.set_ylabel("Accelerometer")
    ax22 = ax21.twinx()
    ax22.set_ylabel("Gyroscope")

    plt.plot(acc_x, label='acc_x', linewidth=1)
    plt.plot(acc_y, label='acc_y', linewidth=1)
    plt.plot(acc_z, label='acc_z', linewidth=1)
    plt.plot(gyro_x, label='gyro_x', linewidth=1)
    plt.plot(gyro_y, label='gyro_y', linewidth=1)
    plt.plot(gyro_z, label='gyro_z', linewidth=1)

    labeling(labels, num_class)

    plt.tight_layout()
    plt.savefig(run_paths['path_model_id'] + '/Visualisation.png', dpi=150)
    plt.show()