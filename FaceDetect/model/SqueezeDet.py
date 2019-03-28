import keras

import numpy        as  np
import tensorflow   as  tf

from keras               import     backend as K
from keras.layers        import     *
from keras.models        import     Model
from keras.regularizers  import     l2
from keras.initializers  import     TruncatedNormal


K.set_image_data_format('channels_last')

#================================ CLASS DEFFINITION ======================================
class SqueezeDet:

    #================================ CONSTRUCTOR ========================================
    def __init__(self, config): 
        '''constructor
        
        Arguments:
            config {easyDict} -- easydict object for configuratio. Generate with make 
            config
        '''

        self.config     =   config
        self.model      =   self._create_model()

    #================================ CREATE MODEL =======================================
    def _create_model(self):
        '''Create the actual body of the network
        
        Returns:
            model -- the keras model
        '''

        net = {}
        cfg = self.config

        net['input'] = Input(shape = cfg.INPUT_SHAPE )
        net['1'] = Conv2D(64, (3,3), 
                                strides     =   (2,2), 
                                padding     =   'same', 
                                activation  =   'relu', 
                                kernel_initializer  =   TruncatedNormal(stddev=0.001), 
                                kernel_regularizer  =   l2(cfg.WEIGHT_DECAY)
                            )(net['input'])
                            
        net['2'] = MaxPool2D((3,3), strides=(2,2), padding='same')(net['1'])

        # fire module 1
        net['3'] = self._fire_module(net, net['2'], modId = 1, sqNo = 16, expNo = 64)
        net['4'] = self._fire_module(net, net['3'], modId = 2, sqNo = 16, expNo = 64)
        net['5'] = MaxPool2D((3,3), strides = (2,2), padding = 'same')(net['4'])
        
        # fire module 2
        net['6'] = self._fire_module(net, net['5'], modId = 3, sqNo = 32, expNo = 128)
        net['7'] = self._fire_module(net, net['6'], modId = 4, sqNo = 32, expNo = 128)
        net['8'] = MaxPool2D((3,3), strides = (2,2), padding = 'same')(net['7'])

        # fire module 3
        net['9']  = self._fire_module(net, net['8'],  modId = 5, sqNo = 48, expNo = 192)
        net['10'] = self._fire_module(net, net['9'],  modId = 6, sqNo = 48, expNo = 192)
        net['11'] = self._fire_module(net, net['10'], modId = 7, sqNo = 64, expNo = 256)
        net['12'] = self._fire_module(net, net['11'], modId = 8, sqNo = 64, expNo = 256)
        
        # fire module 4
        net['13'] = self._fire_module(net, net['12'], modId = 9, sqNo = 96, expNo = 384 )
        net['14'] = self._fire_module(net, net['13'], modId = 10, sqNo = 96, expNo = 384)
        

        net['15'] = Dropout(0.15)(net['14'])

        outputNo  = cfg.ANCHOR_PER_GRID * (cfg.CLASS_NO + 1 + 4)

        net['16'] = Conv2D(outputNo, (3,3), 
                        strides     =   (1,1),
                        padding     =   'same',
                        use_bias    =   True,
                        kernel_initializer  =   TruncatedNormal(stddev=0.001),
                        kernel_regularizer  =   l2(cfg.WEIGHT_DECAY)
                        )(net['15'])

       
        net['17'] = Reshape((cfg.ANCHORS_NO , -1))(net['16'])
        
        model = Model(inputs=net['input'], outputs=net['17'])

        return model

    # ===================================== FIRE MODULE ==================================
    def _fire_module(self, mDict, lastLayer, modId, sqNo = 16, expNo = 64):
        '''The fire module explained in the SqueezeNet Keras
        
        Arguments:
            mDict {dict} -- model dictionary
            lastLayer {layer} -- the last layer in the model
            modId {int} -- the id of the curent module used in name
        
        Keyword Arguments:
            sqNo {int} -- the number of squeeze filters (default: {16})
            expNo {int} -- the number of expansion filters (default: {64})
        
        Returns:
            layer -- the last layer for making further connections
        '''

        modSuf = 'fire' + str(modId) + '/'

        name1 = modSuf + "squeeze1x1" 
        mDict[name1] = Conv2D(sqNo, (1,1), 
                                strides = (1, 1), 
                                padding = 'same', 
                                activation = 'relu', 
                                name = name1 
                            )(lastLayer)

        name2 = modSuf + "expand1x1"
        mDict[name2] = Conv2D(expNo, (1,1), 
                                strides     = (1, 1), 
                                padding     = 'same', 
                                activation  = 'relu', 
                                name        = name2
                            )(mDict[name1])

        name3 = modSuf + "expand3x3"
        mDict[name3] = Conv2D(expNo, (3,3), 
                                strides     = (1, 1), 
                                padding     = 'same',
                                activation  = 'relu', 
                                name        = name3 
                            )(mDict[name1])

        name4 = modSuf + 'concat'

        return concatenate([ mDict[name2], mDict[name3] ], axis = 3, name = name4)



#================================ TEST ===================================================
# import json
# from easydict import EasyDict as ED

# def load(path):

#     with open(path, 'r') as f:
#         cfg = json.load(f)

#     for key, val in cfg.items():
#         if type(val) is list:
#             cfg[key] = np.array(val)

#     cfg = ED(cfg)
#     return cfg

# cfgPath = ".\\NeuralNetworks\\SqueezeDet\\config\\SQDc.config"
# myCfg = load(cfgPath)

# a = SqueezeDet(myCfg)
# a.model.summary()