import os
import glob
import json
import numpy as np
import string
from easydict import EasyDict as ED

#=================================== SQD CONFIG =========================================
def getSQDconfig():
    cfg = ED()

    # detection classes
    # cfg.CLASS_NAMES     =   [str(i) for i in range(43-5)]
    # cfg.CLASS_NAMES     =   ['A', 'B', 'C', 'D', 'E' ]
    # cfg.CLASS_NAMES       =   [letter for letter in string.ascii_uppercase[:5]]
    # cfg.CLASS_NAMES     =   [str(i) for i in range(43)]
    cfg.CLASS_NAMES     =   ['park', 'stop']
    
    cfg.CLASS_NO        =   len(cfg.CLASS_NAMES)
    cfg.CLASS_TO_IDX    =   dict(zip(cfg.CLASS_NAMES, range(cfg.CLASS_NO)))

    # dropout during training
    cfg.DROPOUT         =   0.15

    # image info
    cfg.IMAGE_WIDTH     =   640
    cfg.IMAGE_HEIGHT    =   480
    cfg.NO_CHANNELS     =   3
    cfg.INPUT_SHAPE     =   ( cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, cfg.NO_CHANNELS)

    # batch info
    cfg.BATCH_SIZE              =   4
    
    # optimizer coeffs
    cfg.WEIGHT_DECAY    =   0.0001
    cfg.LEARNING_RATE   =   0.01
    cfg.MAX_GRAD_NORM   =   1.0
    cfg.MOMENTUM        =   0.9
    cfg.EPSILON         =   1e-16
    cfg.EXP_THRES       =   1.0

    # eval thresholds
    cfg.NMS_THRESH      =   0.4                 # thresholds used for detecting
    cfg.PROB_THRESH     =   0.005               # and keeping the boxes
    cfg.TOP_N_DETECTION =   4
    cfg.IOU_THRESH      =   0.5
    cfg.FINAL_THRESHOLD =   0.0
    
    #coefficients of loss function
    cfg.LOSS_COEF_BBOX        = 5.0
    cfg.LOSS_COEF_CONF_POS    = 75.0
    cfg.LOSS_COEF_CONF_NEG    = 100.0
    cfg.LOSS_COEF_CLASS       = 1.0

    # threshold for safe exponential operation
    cfg.EXP_THRESH = 1.0
    
    # particular anchors

    # each cell of the grid will have a bounding box with the below W and H

    # cfg.ANCHOR_SEED     =  np.array([[  36.,  37.], [ 175., 174.], [ 60.,  59.],
    #                                  [ 162.,  87.], [  38.,  90.], [ 258., 173.],
    #                                  [ 224., 108.], [  78., 170.], [  72.,  43.]])
                                    

    cfg.ANCHOR_SEED     =  np.array([[  31.,  30.], [ 40., 41.], [ 50.,  51.],
                                    [  20.,  21.], [ 10., 11.],
                                    [ 60., 61.], [  100., 110.], [  120.,  110.]])

    cfg.ANCHOR_PER_GRID     =   len(cfg.ANCHOR_SEED)

    # how many bounding boxes are spaced from one another
    # taken from last layer shape of the model 
    # 85 50 / 63 50 / 
    cfg.ANCHOR_WIDTH      =   40
    cfg.ANCHOR_HEIGHT     =   30

    # all anchors

    cfg.ANCHOR_BOX          =   set_anchors(cfg)        # this will contain a dictionary 
                                                        # with all the generated bounding 
                                                        # in the format (Cx, Cy, W, H)

    cfg.ANCHORS_NO          =   len(cfg.ANCHOR_BOX)     # the total number of the bounding 
                                                        # boxed
    print(  cfg.ANCHORS_NO) 
    return cfg

#=================================== SQD SAVE ============================================
def save(configDict, path = '.\\NeuralNetworks\\SqueezeDet\\config\\SQDts.config'):
    
    # make save-able
    for key, val in configDict.items():
        if type(val) is np.ndarray:
            configDict[key] = val.tolist()
    # save
    with open(path, 'w') as f:
        json.dump(configDict, f, indent = 0)

#=================================== SQD LOAD ============================================
def load(path):

    with open(path, 'r') as f:
        cfg = json.load(f)
        
    for key, val in cfg.items():
        if type(val) is list:
            cfg[key] = np.array(val)

    cfg = ED(cfg)
    return cfg

#=================================== SET ANCHORS =========================================
def set_anchors(cfg):
    '''Generate all anchors in the form of (CenterX, CenterY, Width, Height)

    
    Arguments:
        cfg {dict} -- a configuration dictionary where the anchors width, height, 
        number/box and seeds are deffinied
    
    Returns:
        array -- an array that contains all bounding boxed in the form specified above
    '''

    H, W, B = cfg.ANCHOR_HEIGHT, cfg.ANCHOR_WIDTH, cfg.ANCHOR_PER_GRID

    anchor_shapes = np.reshape(
        [cfg.ANCHOR_SEED] * H * W,
        (H, W, B, 2)
    )

    center_x = np.reshape(
        np.transpose(
            np.reshape(
                np.array([np.arange(1, W+1) * float(cfg.IMAGE_WIDTH)/(W + 1)] * H * B),
                (B, H, W)
            ),
            (1, 2, 0)
        ),
        (H, W, B, 1)
    )

    center_y = np.reshape(
        np.transpose(
            np.reshape(
                np.array([np.arange(1, H+1) * float(cfg.IMAGE_HEIGHT)/( H + 1 )] * W * B),
                (B, W, H)
            ),
            (2, 1, 0)
        ),
        (H, W, B, 1)
    )

    anchors = np.reshape(
        np.concatenate((center_x, center_y, anchor_shapes), axis=3),
        (-1, 4)
    )

    return anchors

if __name__ =='__main__':
    save(getSQDconfig())