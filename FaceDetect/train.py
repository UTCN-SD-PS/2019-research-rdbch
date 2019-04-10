import sys
sys.path.append('.')

from NeuralNetworks.SqueezeDet.model.SqueezeDet       import  SqueezeDet
from NeuralNetworks.SqueezeDet.config.make_config     import  load
from NeuralNetworks.SqueezeDet.data_generator.DataGenerator import data_generator_path

import os
import gc
import pickle
import argparse
import tensorflow as tf
import keras.backend as K

from keras import optimizers
from keras.callbacks import ModelCheckpoint

dataset = 'DummyObjectDataset'
# dataset = 'TrafficSigns'
img_file              =   ".\\Assets\\" + dataset + "\\images.txt"
gt_file               =   ".\\Assets\\" + dataset + "\\labels.txt"
log_dir_name          =   '.\\log'
EPOCHS                =   100
STEPS                 =   None
VERBOSE               =   False
PRINT_TIME            =   0                                                                                                                                                
REDUCELRONPLATEAU     =   True

CONFIG = ".\\NeuralNetworks\\SqueezeDet\\config\\SQD.config"


def train():
   
    #open files with images and ground truths files with full path names
    with open(img_file) as imgs:
        img_names = imgs.read().splitlines()

   
    with open(gt_file) as gts:
        gt_names = gts.read().splitlines()

    # img_names = img_names[:1000]
    # gt_names = gt_names[:1000]
    # create config object
    cfg = load(CONFIG)

    #compute number of batches per epoch
    nbatches_train, mod = divmod(len(img_names), cfg.BATCH_SIZE)

    cfg.STEPS = nbatches_train

    #print some run info
    print("Number of images: {}".format(len(img_names)))
    print("Number of epochs: {}".format(EPOCHS))
    print("Number of batches: {}".format(nbatches_train))
    print("Batch size: {}".format(cfg.BATCH_SIZE))

    #tf config and session
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    K.set_session(sess)


    #instantiate model
    squeeze = SqueezeDet(cfg)

    #callbacks
    cb = []

    #set optimizer 0.0001
    opt = optimizers.Adam(lr=0.0001, clipnorm=cfg.MAX_GRAD_NORM)
        

    #print keras model summary
    if VERBOSE:
        print(squeeze.model.summary())

    #create train generator
    train_generator = data_generator_path(img_names, gt_names, cfg)

    # add a checkpoint saver
    ckp_saver = ModelCheckpoint(".\\Utils\\model.{epoch:02d}-{loss:.2f}.hdf5", 
                                monitor ='loss', 
                                # verbose=0,
                                save_best_only=False,
                                save_weights_only=False)
    cb.append(ckp_saver)


    print("Using single GPU")
    #compile model from squeeze object, loss is not a function of model directly
    squeeze.model.compile(optimizer =   opt,
                            loss    =   [squeeze.loss],
                            metrics =   [squeeze.bbox_loss, squeeze.class_loss, squeeze.conf_loss]
                            )

    #actually do the training
    squeeze.model.fit_generator(train_generator, 
                                    epochs          =   EPOCHS,
                                    steps_per_epoch =   nbatches_train, 
                                    callbacks       =   cb)


    gc.collect()


if __name__ == "__main__":

    train()
