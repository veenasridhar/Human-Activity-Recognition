import gin
import tensorflow as tf
import logging
import os
import numpy as np

@gin.configurable
class Trainer(object):
    def __init__(self, model, ds_train, ds_val, ds_info, run_paths, model_type, total_steps, log_interval, ckpt_interval, learning_rate, patience, min_delta):
        # Initialize model, datasets, and training parameters
        self.model = model
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.ds_info = ds_info
        self.run_paths = run_paths
        self.total_steps = total_steps
        self.log_interval = log_interval
        train_log_dir = os.path.join(run_paths['path_model_id'], 'logs', 'train')
        val_log_dir = os.path.join(run_paths['path_model_id'], 'logs', 'val')
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.val_summary_writer = tf.summary.create_file_writer(val_log_dir)
        self.ckpt_interval = ckpt_interval
        self.model_type = model_type
        
        # Loss function and optimizer
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

        # Metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        self.val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

        # Checkpoint Manager
        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, net=self.model)
        self.manager = tf.train.CheckpointManager(self.checkpoint, run_paths['path_ckpts_train'], max_to_keep=5)
        if self.manager.latest_checkpoint:
            print("Restored from {}".format(self.manager.latest_checkpoint))
        else:
            print("Initializing from scratch...")

        # Early stopping parameters
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_metric = float('-inf')  # Initialize best validation metric
        self.epochs_without_improvement = 0  # Track epochs with no improvement

    @tf.function
    def train_step(self, signal, labels):
        with tf.GradientTape() as tape:
            if self.model_type == 'seq2label':
                predictions = self.model(signal, training=True)
            elif self.model_type == 'seq2seq':
                predictions = self.model(signal, training=True)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(labels, predictions)
        return labels, predictions
    
    @tf.function
    def val_step(self, signal, labels):
        if self.model_type == 'seq2label':
            predictions = self.model(signal, training=False)
        elif self.model_type == 'seq2seq':
            predictions = self.model(signal, training=False)
        t_loss = self.loss_object(labels, predictions)

        self.val_loss(t_loss)
        self.val_accuracy(labels, predictions)
        return labels, predictions
    
    def train(self):
        max_accuracy = 0.0
        min_loss = float("inf")
        for idx, (signal, labels) in enumerate(self.ds_train):

            step = idx + 1
            self.train_step(signal, labels)

            if step % self.log_interval == 0:

                # Reset test metrics
                self.val_loss.reset_state()
                self.val_accuracy.reset_state()
    
                for val_signal, val_labels in self.ds_val:
                    self.val_step(val_signal, val_labels)


                template = 'Step {}, Loss: {}, Accuracy: {}, Validation Loss: {}, Validation Accuracy: {}'
                logging.info(template.format(step,
                                             self.train_loss.result(),
                                             self.train_accuracy.result() * 100,
                                             self.val_loss.result(),
                                             self.val_accuracy.result() * 100
                                             ))

                curr_accuracy = self.val_accuracy.result().numpy()
                curr_loss = self.val_loss.result().numpy()

                # Early stopping logic
                val_metric = self.val_accuracy.result().numpy()
                if val_metric > self.best_val_metric + self.min_delta:
                    self.best_val_metric = val_metric
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1

                if self.epochs_without_improvement >= self.patience:
                    logging.info(
                        f'Early stopping triggered at step {step}. Best val_accuracy: {self.best_val_metric:.4f}')
                    logging.info("Saved checkpoint for step {}: {}".format(int(self.checkpoint.step), save_path))
                    return self.best_val_metric

                # Write summary to tensorboard
                with self.train_summary_writer.as_default():
                    tf.summary.scalar('loss', self.train_loss.result(), step=step)
                    tf.summary.scalar('accuracy', self.train_accuracy.result() * 100, step=step)


                with self.val_summary_writer.as_default():
                    tf.summary.scalar('loss', self.val_loss.result(), step=step)
                    tf.summary.scalar('accuracy', self.val_accuracy.result() * 100, step=step)


                # Reset train metrics
                self.train_loss.reset_state()
                self.train_accuracy.reset_state()


                yield self.val_accuracy.result().numpy()


            if step % self.ckpt_interval == 0:
                if self.val_accuracy.result() >= max_accuracy and self.val_loss.result() <= min_loss:
                    min_loss = self.val_loss.result().numpy()
                    max_accuracy = self.val_accuracy.result().numpy()
                    # Save checkpoint
                    self.checkpoint.step.assign_add(1)
                    save_path = self.manager.save()
                    logging.info("Saved checkpoint for step {}: {}".format(int(self.checkpoint.step), save_path ))
                elif self.val_loss.result() <= min_loss:
                    min_loss = self.val_loss.result().numpy()
                    max_accuracy = self.val_accuracy.result().numpy()
                    self.checkpoint.step.assign_add(1)
                    save_path = self.manager.save()
                    logging.info("Saved checkpoint for step {}: {}".format(int(self.checkpoint.step), save_path ))

            if step % self.total_steps == 0:
                logging.info(f'Finished training after {step} steps.')
                # Save final checkpoint
                if self.val_accuracy.result() >= max_accuracy and self.val_loss.result() <= min_loss:
                    self.checkpoint.step.assign_add(1)
                    save_path = self.manager.save()
                logging.info("Saved checkpoint for step {}: {}".format(int(self.checkpoint.step), save_path))

                return self.val_accuracy.result().numpy()
