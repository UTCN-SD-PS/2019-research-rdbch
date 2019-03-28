import os
import glob
import json
import numpy as np

from easydict import EasyDict as ED

#=================================== SQD CONFIG =========================================
def getSQDconfig():
    cfg = ED()

    # detection classes
    cfg.CLASS_NAMES     =   ['A', 'B', 'C', 'D', 'E']
    cfg.CLASS_NO        =   len(cfg.CLASS_NAMES)
    cfg.CLASS_TO_IDX    =   dict(zip(cfg.CLASS_NAMES, range(cfg.CLASS_NO)))

    # dropout during training
    cfg.DROPOUT         =   0.5

    
    # image info
    cfg.IMAGE_WIDTH     =   1000
    cfg.IMAGE_HEIGHT    =   800
    cfg.NO_CHANNELS     =   3
    cfg.INPUT_SHAPE     =   ( cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, cfg.NO_CHANNELS)

    # batch info
    cfg.BATCH_SIZE                  =   100
    cfg.BATCH_SIZE_VISUALIZATION    =   1

    # loss function params
    cfg.LOSS_PARAM_BBOX         =   5.0         # params taken from the keras 
    cfg.LOSS_PARAM_CONF_POS     =   75.0        #implementation of sqd found on github
    cfg.LOSS_PARAM_CONF_NRG     =   100.0
    cfg.LOSS_PARAM_CLASS        =   1.0

    # optimizer coeffs
    cfg.WEIGHT_DECAY    =   0.001
    cfg.LEARNING_RATE   =   0.01
    cfg.MAX_GRAD_NORM   =   1.0
    cfg.MOMENTUM        =   0.9
    cfg.EPSILON         =   1e-16
    cfg.EXP_THRES       =   1.0

    # eval thresholds
    cfg.NMS_THRESH      =   0.4                 # thresholds used for detecting
    cfg.PROB_THRESH     =   0.005               # and keeping the boxes
    cfg.TOP_N_DETECT    =   64
    cfg.IOU_THRESH      =   0.5
    cfg.FINAL_THRES     =   0.0

    # particular anchors

    # by anchors seed we denote the (W,H) for the bounding box template
    # each cell of the grid will have a bounding box with the below W and H

    cfg.ANCHOR_SEED     =  np.array([[  36.,  37.], [ 366., 174.], [ 115.,  59.],
                                    [ 162.,  87.], [  38.,  90.], [ 258., 173.],
                                    [ 224., 108.], [  78., 170.], [  72.,  43.]])
    cfg.ANCHOR_PER_GRID     =   len(cfg.ANCHOR_SEED)

    # how far the center of the bounding boxes are spaced from one another
    cfg.ANCHOR_WIDTH        =   50
    cfg.ANCHOR_HEIGHT       =   25

    # all anchors

    cfg.ANCHOR_BOX          =   set_anchors(cfg)        # this will contain a dictionary 
                                                        # with all the generated bounding 
                                                        # in the format (Cx, Cy, W, H)

    cfg.ANCHORS_NO          =   len(cfg.ANCHOR_BOX)     # the total number of the bounding 
                                                        # boxed

    return cfg

#=================================== SQD SAVE ============================================
def save(configDict, path = 'Assets\\config\\SQD.config'):
    
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