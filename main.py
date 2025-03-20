import gin
import logging
from absl import app, flags
import numpy as np
import os
from train import Trainer
from evaluation.eval import *
from input_pipeline import datasets
from utils import utils_params, utils_misc
from visualization import visual
from models.architectures import *

FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', True, 'Specify whether to train or evaluate a model.')

model_name = 'GRU_model'            #Possible options: GRU_model, LSTM_model, ConvLSTM_seq2seq_model
model_type = 'seq2label'             #seq2seq, seq2label

@gin.configurable
def main(argv):

    model_name = 'GRU_model'
    model_type = 'seq2label'
    folder_path = ''
    if len(argv) >1:
        model_type = argv[1] 
        model_name = argv[2]            #Model Name
        if len(argv) == 4:
            folder_path = argv[3]           #CheckPoint Path
        else:
            folder_path = ''
    else:
        print("Insufficient Inputs For Training: python main.py --train=True model_type model_name. \n For  Evaluation: python main.py --train=False model_type model_name checkpoint_folderpath")

    # generate folder structures1
    run_paths = utils_params.gen_run_folder('')

    if folder_path != '':
        run_paths = utils_params.gen_run_folder(folder_path)

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # gin-config
    gin.parse_config_files_and_bindings(['configs/config.gin'], [])
    utils_params.save_config(run_paths['path_gin'], gin.config_str())
    

    # setup pipeline
    ds_train, ds_val, ds_test, ds_info, window_size = datasets.load(model_type)

    #model
    if model_name == 'GRU_model':
        model = gru_model(input_shape=(window_size,6), n_classes=ds_info['num_classes'])
    elif model_name == 'LSTM_model':
        model = lstm_model(input_shape=(window_size,6), n_classes=ds_info['num_classes'])
    elif model_name == 'ConvLSTM':
        model = ConvLSTMseq2seq(input_shape=(window_size,6), n_classes=ds_info['num_classes'])
    model.summary()

    if FLAGS.train:
        trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths, model_type)
        for _ in trainer.train():
            continue

    else:
        print(model_type)
        checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=tf.keras.optimizers.Adam(), net=model)
        evaluate(model,checkpoint,ds_test,ds_info,run_paths,model_type)
        if model_type == 'seq2seq':
            visual(model,checkpoint, ds_test,model_type,ds_info['num_classes'],run_paths= run_paths)

if __name__ == "__main__":
    app.run(main)