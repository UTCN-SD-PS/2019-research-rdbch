import time
import numpy as np
import tensorflow as tf
import keras.backend as K

#================================ IOU ====================================================
def iou(box1, box2):

    lr = min(box1[0]+0.5*box1[2], box2[0]+0.5*box2[2]) - \
        max(box1[0]-0.5*box1[2], box2[0]-0.5*box2[2])
    if lr > 0:
        tb = min(box1[1]+0.5*box1[3], box2[1]+0.5*box2[3]) - \
            max(box1[1]-0.5*box1[3], box2[1]-0.5*box2[3])
        if tb > 0:
            intersection = tb*lr
            union = box1[2]*box1[3]+box2[2]*box2[3]-intersection

        return intersection/union

    return 0

#================================ BATCH IOU ==============================================
def batch_iou(boxes, box):
  lr = np.maximum(
      np.minimum(boxes[:,0]+0.5*boxes[:,2], box[0]+0.5*box[2]) - \
      np.maximum(boxes[:,0]-0.5*boxes[:,2], box[0]-0.5*box[2]),
      0
  )
  tb = np.maximum(
      np.minimum(boxes[:,1]+0.5*boxes[:,3], box[1]+0.5*box[3]) - \
      np.maximum(boxes[:,1]-0.5*boxes[:,3], box[1]-0.5*box[3]),
      0
  )
  inter = lr*tb
  union = boxes[:,2]*boxes[:,3] + box[2]*box[3] - inter
  return inter/union


#================================ NMS ====================================================
def nms(boxes, probs, threshold):

  order = probs.argsort()[::-1]
  keep = [True]*len(order)

  for i in range(len(order)-1):
    ovps = batch_iou(boxes[order[i+1:]], boxes[order[i]])
    for j, ov in enumerate(ovps):
      if ov > threshold:
        keep[order[j+i+1]] = False
  return keep

#================================ SPARSE TO DENSE ========================================
def sparse_to_dense(sp_indices, output_shape, values, default_value=0):
  
    assert len(sp_indices) == len(values), \
        'Length of sp_indices is not equal to length of values'

    array = np.ones(output_shape) * default_value
    for idx, value in zip(sp_indices, values):
        array[tuple(idx)] = value
    return array
#================================ CONVERT CHANNELS =======================================
def bgr_to_rgb(ims):
    """Convert a list of images from BGR format to RGB format."""
    out = []
    for im in ims:
        out.append(im[:,:,::-1])
    return out

#================================ BBOX TRANSFORM==========================================
def bbox_transform(bbox):
    """convert a bbox of form [cx, cy, w, h] to [xmin, ymin, xmax, ymax]. Works
    for numpy array or list of tensors.
    """
    cx, cy, w, h = bbox
    out_box = [[]]*4
    out_box[0] = cx-w/2
    out_box[1] = cy-h/2
    out_box[2] = cx+w/2
    out_box[3] = cy+h/2

    return out_box

#================================ BBOX TRANSFORM =========================================
def bbox_transform_inv(bbox):
    """convert a bbox of form [xmin, ymin, xmax, ymax] to [cx, cy, w, h]. Works
    for numpy array or list of tensors.
    """
    xmin, ymin, xmax, ymax = bbox
    out_box = [[]]*4

    width       = xmax - xmin + 1.0
    height      = ymax - ymin + 1.0
    out_box[0]  = xmin + 0.5*width
    out_box[1]  = ymin + 0.5*height
    out_box[2]  = width
    out_box[3]  = height

    return out_box

#================================ EXP ====================================================
def safe_exp(w, thresh):

    slope = np.exp(thresh)
    lin_bool = w > thresh

    lin_region = K.cast(lin_bool, dtype='float32')

    lin_out = slope*(w - thresh + 1.)

    exp_out = K.exp(K.switch(lin_bool, K.zeros_like(w), w))

    out = lin_region*lin_out + (1.-lin_region)*exp_out

    return out

#================================ EXP NUMPY ==============================================
def safe_exp_np(w, thresh):

    slope = np.exp(thresh)
    lin_bool = w > thresh

    lin_region = lin_bool.astype(float)

    lin_out = slope*(w - thresh + 1.)

    exp_out = np.exp(np.where(lin_bool, np.zeros_like(w), w))

    out = lin_region*lin_out + (1.-lin_region)*exp_out

    return out



#================================ BOXFROMDELTA ===========================================
def boxes_from_deltas(pred_box_delta, config):

    # Keras backend allows no unstacking

    delta_x = pred_box_delta[:, :, 0]
    delta_y = pred_box_delta[:, :, 1]
    delta_w = pred_box_delta[:, :, 2]
    delta_h = pred_box_delta[:, :, 3]

    # get the coordinates and sizes of the anchor boxes from config

    anchor_x = config.ANCHOR_BOX[:, 0]
    anchor_y = config.ANCHOR_BOX[:, 1]
    anchor_w = config.ANCHOR_BOX[:, 2]
    anchor_h = config.ANCHOR_BOX[:, 3]

    # as we only predict the deltas, we need to transform the anchor box values before computing the loss

    box_center_x = K.identity(
        anchor_x + delta_x * anchor_w)
    box_center_y = K.identity(
        anchor_y + delta_y * anchor_h)
    box_width = K.identity(
        anchor_w * safe_exp(delta_w, config.EXP_THRESH))
    box_height = K.identity(
        anchor_h * safe_exp(delta_h, config.EXP_THRESH))

    # tranform into a real box with four coordinates

    xmins, ymins, xmaxs, ymaxs = bbox_transform([box_center_x, box_center_y, box_width, box_height])

    # trim boxes if predicted outside

    xmins = K.minimum(
        K.maximum(0.0, xmins), config.IMAGE_WIDTH - 1.0)
    ymins = K.minimum(
        K.maximum(0.0, ymins), config.IMAGE_HEIGHT - 1.0)
    xmaxs = K.maximum(
        K.minimum(config.IMAGE_WIDTH - 1.0, xmaxs), 0.0)
    ymaxs = K.maximum(
        K.minimum(config.IMAGE_HEIGHT - 1.0, ymaxs), 0.0)

    det_boxes = K.permute_dimensions(
        K.stack(bbox_transform_inv([xmins, ymins, xmaxs, ymaxs])),
        (1, 2, 0)
    )
    
    return (det_boxes)


#================================ BOX2DELTA ==============================================
def boxes_from_deltas_np(pred_box_delta, config):


    # Keras backend allows no unstacking

    delta_x = pred_box_delta[:, :, 0]
    delta_y = pred_box_delta[:, :, 1]
    delta_w = pred_box_delta[:, :, 2]
    delta_h = pred_box_delta[:, :, 3]

    # get the coordinates and sizes of the anchor boxes from config

    anchor_x = config.ANCHOR_BOX[:, 0]
    anchor_y = config.ANCHOR_BOX[:, 1]
    anchor_w = config.ANCHOR_BOX[:, 2]
    anchor_h = config.ANCHOR_BOX[:, 3]

    # as we only predict the deltas, we need to transform the anchor box values before computing the loss

    box_center_x = anchor_x + delta_x * anchor_w
    box_center_y = anchor_y + delta_y * anchor_h
    box_width = anchor_w * safe_exp_np(delta_w, config.EXP_THRESH)
    box_height = anchor_h * safe_exp_np(delta_h, config.EXP_THRESH)

    # tranform into a real box with four coordinates

    xmins, ymins, xmaxs, ymaxs = bbox_transform([box_center_x, box_center_y, box_width, box_height])

    # trim boxes if predicted outside

    xmins = np.minimum(
        np.maximum(0.0, xmins), config.IMAGE_WIDTH - 1.0)
    ymins = np.minimum(
        np.maximum(0.0, ymins), config.IMAGE_HEIGHT - 1.0)
    xmaxs = np.maximum(
        np.minimum(config.IMAGE_WIDTH - 1.0, xmaxs), 0.0)
    ymaxs = np.maximum(
        np.minimum(config.IMAGE_HEIGHT - 1.0, ymaxs), 0.0)

    det_boxes = np.transpose(
        np.stack(bbox_transform_inv([xmins, ymins, xmaxs, ymaxs])),
        (1, 2, 0)
    )

    return (det_boxes)

#================================ SLICE PRED =============================================
def slice_predictions(y_pred, config):
    
    # calculate non padded entries
    n_outputs = config.CLASS_NO + 1 + 4

    # slice and reshape network output
    y_pred = y_pred[:, :, 0:n_outputs]
    y_pred = K.reshape(y_pred, (config.BATCH_SIZE, config.ANCHOR_HEIGHT, config.ANCHOR_WIDTH, -1))
    
    # number of class probabilities, n classes for each anchor
    
    num_class_probs = config.ANCHOR_PER_GRID * config.CLASS_NO

    # slice pred tensor to extract class pred scores and then normalize them
    pred_class_probs = K.reshape(
        K.softmax(
            K.reshape(
                y_pred[:, :, :, :num_class_probs],
                [-1, config.CLASS_NO]
            )
        ),
        [config.BATCH_SIZE, config.ANCHORS_NO, config.CLASS_NO],
    )

    # number of confidence scores, one for each anchor + class probs
    num_confidence_scores = config.ANCHOR_PER_GRID + num_class_probs

    # slice the confidence scores and put them trough a sigmoid for probabilities
    pred_conf = K.sigmoid(
        K.reshape(
            y_pred[:, :, :, num_class_probs:num_confidence_scores],
            [config.BATCH_SIZE, config.ANCHORS_NO]
        )
    )

    # slice remaining bounding box_deltas
    pred_box_delta = K.reshape(
        y_pred[:, :, :, num_confidence_scores:],
        [config.BATCH_SIZE, config.ANCHORS_NO, 4]
    )
    
    return [pred_class_probs, pred_conf, pred_box_delta]


#================================ SLICE PRED NUMPY =======================================
def slice_predictions_np(y_pred, config):

    # calculate non padded entries
    n_outputs = config.CLASS_NO + 1 + 4
    # slice and reshape network output
    y_pred = y_pred[:, :, 0:n_outputs]
    y_pred = np.reshape(y_pred, (config.BATCH_SIZE, config.ANCHOR_HEIGHT, config.ANCHOR_WIDTH, -1))

    # number of class probabilities, n classes for each anchor

    num_class_probs = config.ANCHOR_PER_GRID * config.CLASS_NO

    # slice pred tensor to extract class pred scores and then normalize them
    pred_class_probs = np.reshape(
        softmax(
            np.reshape(
                y_pred[:, :, :, :num_class_probs],
                [-1, config.CLASS_NO]
            )
        ),
        [config.BATCH_SIZE, config.ANCHORS_NO, config.CLASS_NO],
    )

    # number of confidence scores, one for each anchor + class probs
    num_confidence_scores = config.ANCHOR_PER_GRID + num_class_probs

    # slice the confidence scores and put them trough a sigmoid for probabilities
    pred_conf = sigmoid(
        np.reshape(
            y_pred[:, :, :, num_class_probs:num_confidence_scores],
            [config.BATCH_SIZE, config.ANCHORS_NO]
        )
    )

    # slice remaining bounding box_deltas
    pred_box_delta = np.reshape(
        y_pred[:, :, :, num_confidence_scores:],
        [config.BATCH_SIZE, config.ANCHORS_NO, 4]
    )

    return [pred_class_probs, pred_conf, pred_box_delta]

#================================ TENSOR IOU =============================================
def tensor_iou(box1, box2, input_mask, config):
    
    xmin = K.maximum(box1[0], box2[0])
    ymin = K.maximum(box1[1], box2[1])
    xmax = K.minimum(box1[2], box2[2])
    ymax = K.minimum(box1[3], box2[3])

    w = K.maximum(0.0, xmax - xmin)
    h = K.maximum(0.0, ymax - ymin)

    intersection = w * h

    w1 = box1[2] - box1[0]
    h1 = box1[3] - box1[1]
    w2 = box2[2] - box2[0]
    h2 = box2[3] - box2[1]

    union = w1 * h1 + w2 * h2 - intersection

    return intersection / (union + config.EPSILON) * K.reshape(input_mask, [config.BATCH_SIZE, config.ANCHORS_NO])

#================================ SOFTMAX ================================================
def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x))
    return e_x / np.expand_dims(np.sum(e_x,axis=axis), axis=axis)

#================================ SIGMOID ================================================
def sigmoid(x):
    return 1/(1+np.exp(-x))