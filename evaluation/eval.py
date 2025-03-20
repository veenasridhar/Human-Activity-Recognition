import tensorflow as tf
from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from evaluation.metrics import *

def evaluate(model, checkpoint, ds_test, ds_info, run_paths,model_type):
    """
    Evaluate a trained model on a test set.

    Parameters:
        model: Trained model to evaluate
        checkpoint: Checkpoint for restoring model state
        ds_test: Test dataset
        ds_info: Test dataset info
        run_paths: Paths for storing checkpoints
        model_type: Type of model ('seq2label' or 'seq2seq')
    """
    manager = tf.train.CheckpointManager(checkpoint,run_paths['path_ckpts_train'],max_to_keep=5)
    checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
      print("Restored from {}".format(manager.latest_checkpoint))                  
    else:
      print("No checkpoint found")

    num_classes = ds_info["num_classes"]
    confusion_matrix = ConfusionMatrix(num_classes)
    accuracy = Accuracy(num_classes)

    y_preds_list = []
    labels_list = []

    for idx, (images, labels) in enumerate(ds_test):
        if model_type == 'seq2label':
          predictions = model.predict(images)
          y_pred = tf.cast(tf.reshape(tf.math.argmax(predictions, axis=1), shape=(-1, 1)), dtype=tf.int32)
          y_preds_list.append(y_pred)
          labels_list.append(labels.numpy().tolist())
          accuracy.update_state(labels, y_pred)
          confusion_matrix.update_state(labels, y_pred)

        elif model_type == 'seq2seq':
           predictions = model.predict(images)
           y_pred = tf.argmax(predictions,axis=-1)
           labels = tf.reshape(labels,[-1])
           y_pred = tf.reshape(y_pred,[-1])
           y_preds_list.append(y_pred)
           labels_list.append(labels.numpy().tolist())
           accuracy.update_state(labels, y_pred)
           confusion_matrix.update_state(labels, y_pred)       

    y_preds_array = np.concatenate(y_preds_list, axis=0)
    labels_array = np.concatenate(labels_list, axis=0)

    # Print evaluation results
    print('------------------Results-----------------------------')
    print("Test Accuracy {}".format(accuracy.result()))
    confusion_mat = confusion_matrix.result().numpy()
    noramalised_confusion_mat =  confusion_mat.astype('float') /confusion_mat.sum(axis=1)[:,np.newaxis]

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(18,18))
    sns.heatmap(noramalised_confusion_mat, annot=True,cmap="coolwarm", fmt='.2f', xticklabels=ds_info['class_names'], yticklabels=ds_info['class_names'], square=True)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    # Print the raw confusion matrix and F1 score
    print('Confusion Matrix: \n',confusion_matrix.result().numpy())
    print('F1 Score: ',f1_score(labels_array, y_preds_array,average="weighted"))

    # Reset metrics after evaluation
    accuracy.reset_state()
    confusion_matrix.reset_state()