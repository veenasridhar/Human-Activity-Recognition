import logging
import gin

import ray
from ray import tune
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

from input_pipeline.datasets import load
from models.architectures import gru_model
from train import Trainer
from utils import utils_params, utils_misc
import os


#@gin.configurable
def train_func(config):
    # Hyperparameters
    bindings = [f"{key}={value}" for key, value in config.items()]

    # Generate folder structures
    run_paths = utils_params.gen_run_folder(','.join(bindings))

    # Set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    #gin-config
    gin.parse_config_files_and_bindings(['configs/config.gin'],bindings) # change path to absolute path of config file
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # TensorBoard log directory
    tensorboard_log_dir = os.path.join("C:/ray_results", "tensorboard_logs")
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    writer = tf.summary.create_file_writer(tensorboard_log_dir)
    print(f"TensorBoard logs are being saved to: {tensorboard_log_dir}")

    # Hyperparameters and metrics for TensorBoard
    HP_TOTAL_STEPS = hp.HParam('Trainer.total_steps', hp.Discrete([5e3]))
    HP_DENSE_UNITS = hp.HParam('gru_model.dense_units', hp.Discrete([128, 256, 512]))
    HP_DROPOUT_RATE = hp.HParam('gru_model.dropout_rate', hp.RealInterval(0.0, 0.9))
    METRIC_ACCURACY = hp.Metric('val_accuracy', display_name='Validation Accuracy')

    # Hyperparameter and metric configuration logging to TensorBoard
    with writer.as_default():
        hp.hparams_config(
            hparams=[HP_TOTAL_STEPS,HP_DENSE_UNITS, HP_DROPOUT_RATE],
            metrics=[METRIC_ACCURACY],
        )
    # Hyperparameters logging for this specific trial
    with writer.as_default():
        hp.hparams(config)

    # Dataset loading
    ds_train, ds_val, ds_test, ds_info, window_size  = load()

    # Model
    model = gru_model(input_shape=(window_size,6), n_classes=ds_info['num_classes'])
    model.summary()

    trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths)

    best_val_accuracy = 0  # Track the best accuracy so far

    # Training loop
    for step, val_accuracy in enumerate(trainer.train()):
        best_val_accuracy = max(best_val_accuracy, val_accuracy)  # Keep the max accuracy
        print(f"Step {step}: Validation Accuracy = {val_accuracy}, Best Accuracy = {best_val_accuracy}")

        tune.report({"val_accuracy": best_val_accuracy})  # Report the best validation accuracy so far to Ray Tune

    writer.close()

# Initialize Ray with error suppression for reinitialization
ray.init(ignore_reinit_error=True)

# Hyperparameter tuning with Ray Tune
analysis = tune.run(
    train_func,
    config={
        "Trainer.total_steps": tune.grid_search([5e3]),                 # Grid search over total steps
        "gru_model.dropout_rate": tune.uniform(0, 0.9),    # Uniform distribution for dropout rate
        "gru_model.dense_unit": tune.choice([128, 256, 512]),           # Choice of dense unit sizes
    },
    resources_per_trial={"cpu": 6, "gpu": 0},      # Allocate CPU and GPU resources
    num_samples=30,                                # Number of trials to run
    max_concurrent_trials=1,                       # Number of concurrent trials
    stop={"training_iteration": 12}                # Stop condition
)

# The best configuration found during tuning
print("Best config: ", analysis.get_best_config(metric="val_accuracy", mode="max"))

# Get a dataframe for analyzing trial results.
df = analysis.dataframe()
