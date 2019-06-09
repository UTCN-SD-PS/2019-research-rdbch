import keras

import numpy        as  np
import tensorflow   as  tf

from keras               import     backend as K
from keras.layers        import     *
from keras.models        import     Model
from keras.regularizers  import     l2
from keras.initializers  import     TruncatedNormal

import FaceDetect.utils.utils as ut
from FaceDetect.data_generator.DataGenerator import data_generator_path

K.set_image_data_format('channels_last')

#================================ CLASS DEFFINITION ======================================
class SqueezeDet:

    #================================ CONSTRUCTOR ========================================
    def __init__(self, config): 
        self.config     =   config
        self.model      =   self._create_model()

    #================================ CREATE MODEL =======================================
    def _create_model(self):
        net = {}
        cfg = self.config

        net['input'] = Input(shape = cfg.INPUT_SHAPE )
        net['1'] = Conv2D(96, (3,3), 
                                strides     =   (2,2), 
                                padding     =   'same', 
                                activation  =   'relu', 
                                kernel_initializer  =   TruncatedNormal(stddev=0.001), 
                                kernel_regularizer  =   l2(cfg.WEIGHT_DECAY)
                            )(net['input'])
                            
        net['2'] = MaxPool2D((3,3), strides=(2,2), padding='same')(net['1'])

        # fire module 1
        net['3'] = self._fire_module(net, net['2'], modId = 1, sqNo = 96, expNo = 64)
        net['4'] = self._fire_module(net, net['3'], modId = 2, sqNo = 96, expNo = 64)
        net['5'] = self._fire_module(net, net['4'], modId = 3, sqNo = 192, expNo = 128)
        net['6'] = MaxPool2D((3,3), strides = (2,2), padding = 'same')(net['5'])
        
        # fire module 2
        net['7'] = self._fire_module(net, net['6'], modId = 4, sqNo = 192, expNo = 128)
        net['8'] = self._fire_module(net, net['7'], modId = 5, sqNo = 288, expNo = 192)
        net['9'] = self._fire_module(net, net['8'], modId = 6, sqNo = 288, expNo = 192)
        net['10'] = self._fire_module(net, net['9'], modId = 7, sqNo = 384, expNo = 256)
        net['11'] = MaxPool2D((3,3), strides = (2,2), padding = 'same')(net['10'])

        # fire module 3
        net['12']  = self._fire_module(net, net['11'], modId = 8, sqNo = 384, expNo = 256)        
        net['13'] = self._fire_module(net, net['12'], modId = 9,  sqNo = 384, expNo = 256 )
        net['14'] = self._fire_module(net, net['13'], modId = 10, sqNo = 384, expNo = 256)
        

        net['15'] = Dropout(0.15)(net['14'])

        outputNo  = cfg.ANCHOR_PER_GRID * (cfg.CLASS_NO + 1 + 4)

        net['16'] = Conv2D(outputNo, (3,3), 
                        strides     =   (1,1),
                        padding     =   'same',
                        use_bias    =   True,
                        kernel_initializer  =   TruncatedNormal(stddev=0.001),
                        kernel_regularizer  =   l2(cfg.WEIGHT_DECAY)
                        )(net['15'])

        #reshape for getting the info for each box
        net['17'] = Reshape((cfg.ANCHORS_NO , -1))(net['16'])

        # model won't compile otherwise
        net['18'] = Lambda(self._pad)( net['17'])

        model = Model(inputs=net['input'], outputs=net['18'])
        model.summary()
        model.compile('adam', loss=[self.loss])
        
        return model

    #================================ PAD ================================================
    def _pad(self, input):

        padding = np.zeros((3,2))
        padding[2,1] = 4
        return tf.pad(input, padding ,"CONSTANT")

    # ===================================== FIRE MODULE ==================================
    def _fire_module(self, mDict, lastLayer, modId, sqNo = 16, expNo = 64):
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

   
    #================================ LOSS ===============================================
    def loss(self, y_true, y_pred):

        #handle for config
        mc = self.config

        #slice y_true
        input_mask = y_true[:, :, 0]
        input_mask = K.expand_dims(input_mask, axis=-1)
        box_input = y_true[:, :, 1:5]
        box_delta_input = y_true[:, :, 5:9]
        labels = y_true[:, :, 9:]

        #number of objects. Used to normalize bbox and classification loss
        num_objects = K.sum(input_mask)


        #before computing the losses we need to slice the network outputs
        pred_class_probs, pred_conf, pred_box_delta = ut.slice_predictions(y_pred, mc)

        tf.print( pred_class_probs ) 
        tf.print( pred_conf ) 

        #compute boxes
        det_boxes = ut.boxes_from_deltas(pred_box_delta, mc)

        #again unstack is not avaible in pure keras backend
        unstacked_boxes_pred = []
        unstacked_boxes_input = []

        for i in range(4):
            unstacked_boxes_pred.append(det_boxes[:, :, i])
            unstacked_boxes_input.append(box_input[:, :, i])



        #compute the ious
        ious = ut.tensor_iou(ut.bbox_transform(unstacked_boxes_pred),
                                ut.bbox_transform(unstacked_boxes_input),
                                input_mask,
                                mc
                                )


        
        #compute class loss,add a small value into log to prevent blowing up
        class_loss = K.sum(labels * (-K.log(pred_class_probs + mc.EPSILON))
                 + (1 - labels) * (-K.log(1 - pred_class_probs + mc.EPSILON))
                * input_mask * mc.LOSS_COEF_CLASS) / num_objects

        #bounding box loss
        bbox_loss = (K.sum(mc.LOSS_COEF_BBOX * K.square(input_mask * (pred_box_delta - box_delta_input))) / num_objects)

        #reshape input for correct broadcasting
        input_mask = K.reshape(input_mask, [mc.BATCH_SIZE, mc.ANCHORS_NO])

        #confidence score loss
        conf_loss = K.mean(
            K.sum(
                K.square((ious - pred_conf))
                * (input_mask * mc.LOSS_COEF_CONF_POS / num_objects
                   + (1 - input_mask) * mc.LOSS_COEF_CONF_NEG / (mc.ANCHORS_NO - num_objects)),
                axis=[1]
            ),
        )

        # add above losses
        total_loss = class_loss + conf_loss + bbox_loss

        return total_loss


    #the sublosses, to be used as metrics during training

    #================================ BBOX LOSS ==========================================
    def bbox_loss(self, y_true, y_pred):

        #handle for config
        mc = self.config

        #calculate non padded entries
        n_outputs = mc.CLASS_NO + 1 + 4

        #slice and reshape network output
        y_pred = y_pred[:, :, :n_outputs]
        y_pred = K.reshape(y_pred, (mc.BATCH_SIZE, mc.ANCHOR_HEIGHT, mc.ANCHOR_WIDTH, -1))


        #slice y_true
        input_mask = y_true[:, :, 0]
        input_mask = K.expand_dims(input_mask, axis=-1)
        box_delta_input = y_true[:, :, 5:9]

        #number of objects. Used to normalize bbox and classification loss
        num_objects = K.sum(input_mask)

        #number of class probabilities, n classes for each anchor
        num_class_probs = mc.ANCHOR_PER_GRID * mc.CLASS_NO

        #number of confidence scores, one for each anchor + class probs
        num_confidence_scores = mc.ANCHOR_PER_GRID+num_class_probs

        #slice the confidence scores and put them trough a sigmoid for probabilities
        pred_conf = K.sigmoid(
            K.reshape(
                  y_pred[:, :, :, num_class_probs:num_confidence_scores],
                  [mc.BATCH_SIZE, mc.ANCHORS_NO]
              )
          )

        #slice remaining bounding box_deltas
        pred_box_delta = K.reshape(
              y_pred[:, :, :, num_confidence_scores:],
              [mc.BATCH_SIZE, mc.ANCHORS_NO, 4]
          )

        #bounding box loss
        bbox_loss = (K.sum(mc.LOSS_COEF_BBOX * K.square(input_mask * (pred_box_delta - box_delta_input))) / num_objects)

        return bbox_loss


    #================================ CONF LOSS ==========================================
    def conf_loss(self, y_true, y_pred):

        #handle for config
        mc = self.config

        #calculate non padded entries
        n_outputs = mc.CLASS_NO + 1 + 4

        #slice and reshape network output
        y_pred = y_pred[:, :, 0:n_outputs]
        y_pred = K.reshape(y_pred, (mc.BATCH_SIZE, mc.ANCHOR_HEIGHT, mc.ANCHOR_WIDTH, -1))


        #slice y_true
        input_mask = y_true[:, :, 0]
        input_mask = K.expand_dims(input_mask, axis=-1)
        box_input = y_true[:, :, 1:5]

        #number of objects. Used to normalize bbox and classification loss
        num_objects = K.sum(input_mask)

        #number of class probabilities, n classes for each anchor
        num_class_probs = mc.ANCHOR_PER_GRID * mc.CLASS_NO



        #number of confidence scores, one for each anchor + class probs
        num_confidence_scores = mc.ANCHOR_PER_GRID+num_class_probs

        #slice the confidence scores and put them trough a sigmoid for probabilities
        pred_conf = K.sigmoid(
            K.reshape(
                  y_pred[:, :, :, num_class_probs:num_confidence_scores],
                  [mc.BATCH_SIZE, mc.ANCHORS_NO]
              )
          )

        #slice remaining bounding box_deltas
        pred_box_delta = K.reshape(
              y_pred[:, :, :, num_confidence_scores:],
              [mc.BATCH_SIZE, mc.ANCHORS_NO, 4]
          )

        #compute boxes
        det_boxes = ut.boxes_from_deltas(pred_box_delta, mc)


        #again unstack is not avaible in pure keras backend
        unstacked_boxes_pred = []
        unstacked_boxes_input = []

        for i in range(4):
            unstacked_boxes_pred.append(det_boxes[:, :, i])
            unstacked_boxes_input.append(box_input[:, :, i])



        #compute the ious
        ious = ut.tensor_iou(ut.bbox_transform(unstacked_boxes_pred),
                                ut.bbox_transform(unstacked_boxes_input),
                                input_mask,
                                mc
                                )



        #reshape input for correct broadcasting
        input_mask = K.reshape(input_mask, [mc.BATCH_SIZE, mc.ANCHORS_NO])

        #confidence score loss
        conf_loss = K.mean(
            K.sum(
                K.square((ious - pred_conf))
                * (input_mask * mc.LOSS_COEF_CONF_POS / num_objects
                   + (1 - input_mask) * mc.LOSS_COEF_CONF_NEG / (mc.ANCHORS_NO - num_objects)),
                axis=[1]
            ),
        )

        return conf_loss


    #================================ CLASS LOSS =========================================
    def class_loss(self, y_true, y_pred):

        #handle for config
        mc = self.config

        #calculate non padded entries
        n_outputs = mc.CLASS_NO + 1 + 4

        #slice and reshape network output
        y_pred = y_pred[:, :, 0:n_outputs]
        y_pred = K.reshape(y_pred, (mc.BATCH_SIZE, mc.ANCHOR_HEIGHT, mc.ANCHOR_WIDTH, -1))


        #slice y_true
        input_mask = y_true[:, :, 0]
        input_mask = K.expand_dims(input_mask, axis=-1)
        labels = y_true[:, :, 9:]

        #number of objects. Used to normalize bbox and classification loss
        num_objects = K.sum(input_mask)


        #before computing the losses we need to slice the network outputs

        #number of class probabilities, n classes for each anchor
        num_class_probs = mc.ANCHOR_PER_GRID * mc.CLASS_NO

        #slice pred tensor to extract class pred scores and then normalize them
        pred_class_probs = K.reshape(
            K.softmax(
                K.reshape(
                    y_pred[:, :, :, :num_class_probs],
                    [-1, mc.CLASS_NO]
                )
            ),
            [mc.BATCH_SIZE, mc.ANCHORS_NO, mc.CLASS_NO],
        )

        #compute class loss
        class_loss = K.sum((labels * (-K.log(pred_class_probs + mc.EPSILON))
                 + (1 - labels) * (-K.log(1 - pred_class_probs + mc.EPSILON)))
                * input_mask * mc.LOSS_COEF_CLASS) / num_objects




        return class_loss


    #================================ LOSS ===============================================
    def loss_without_regularization(self, y_true, y_pred):

        #handle for config
        mc = self.config

        #slice y_true
        input_mask = y_true[:, :, 0]
        input_mask = K.expand_dims(input_mask, axis=-1)
        box_input = y_true[:, :, 1:5]
        box_delta_input = y_true[:, :, 5:9]
        labels = y_true[:, :, 9:]

        #number of objects. Used to normalize bbox and classification loss
        num_objects = K.sum(input_mask)

        #before computing the losses we need to slice the network outputs
        pred_class_probs, pred_conf, pred_box_delta = ut.slice_predictions(y_pred, mc)

        #compute boxes
        det_boxes = ut.boxes_from_deltas(pred_box_delta, mc)

        #again unstack is not avaible in pure keras backend
        unstacked_boxes_pred = []
        unstacked_boxes_input = []

        for i in range(4):
            unstacked_boxes_pred.append(det_boxes[:, :, i])
            unstacked_boxes_input.append(box_input[:, :, i])



        #compute the ious
        ious = ut.tensor_iou(ut.bbox_transform(unstacked_boxes_pred),
                                ut.bbox_transform(unstacked_boxes_input),
                                input_mask,
                                mc)


        #compute class loss
        class_loss = K.sum(labels * (-K.log(pred_class_probs + mc.EPSILON))
                 + (1 - labels) * (-K.log(1 - pred_class_probs + mc.EPSILON))
                * input_mask * mc.LOSS_COEF_CLASS) / num_objects



        #bounding box loss
        bbox_loss = (K.sum(mc.LOSS_COEF_BBOX * K.square(input_mask * (pred_box_delta - box_delta_input))) / num_objects)

        #reshape input for correct broadcasting
        input_mask = K.reshape(input_mask, [mc.BATCH_SIZE, mc.ANCHORS_NO])

        #confidence score loss
        conf_loss = K.mean(
            K.sum(
                K.square((ious - pred_conf))
                * (input_mask * mc.LOSS_COEF_CONF_POS / num_objects
                   + (1 - input_mask) * mc.LOSS_COEF_CONF_NEG / (mc.ANCHORS_NO - num_objects)),
                axis=[1]
            ),
        )

        # add above losses 
        total_loss = class_loss + conf_loss + bbox_loss

        return total_loss