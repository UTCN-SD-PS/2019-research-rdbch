import cv2
import glob
import keras
import numpy as np
import NeuralNetworks.SqueezeDet.utils.utils        as utils
import NeuralNetworks.SqueezeDet.config.make_config as make_config

from NeuralNetworks.SqueezeDet.model.ModelLoader               import  load_only_possible_weights
from NeuralNetworks.SqueezeDet.model.SqueezeDet                import  SqueezeDet
from NeuralNetworks.SqueezeDet.data_generator.LoadAnnotations  import  load_annotation


keras.backend.set_image_data_format('channels_last')

def viz_image(image, name='Window'):
  cv2.imshow(name, image)
  key = cv2.waitKey(0) & 0XFF
  if key == ord('q'):
    exit()

#================================ FUNTIONS ================================================

def visualize_dt_and_gt(images, y_true, y_pred, config):
    
    img_with_boxes = []

    #filter batch with nms
    all_filtered_boxes, all_filtered_classes, all_filtered_scores = filter_batch(y_pred, config)

    font = cv2.FONT_HERSHEY_SIMPLEX

    #iterate images
    for i, img in enumerate(images):

        #iterate predicted boxes
        for j, det_box in enumerate(all_filtered_boxes[i]):
            #transform into xmin, ymin, xmax, ymax
            det_box = bbox_transform_single_box(det_box)

            #add rectangle and text
            cv2.rectangle(img, (det_box[0], det_box[1]), (det_box[2], det_box[3]), (255,0,0), 3)
            cv2.putText(img, config.CLASS_NAMES[all_filtered_classes[i][j]] + " " + str(all_filtered_scores[i][j])[:5] , (det_box[0], det_box[1]), font, 0.5, (0,0,255), 1, cv2.LINE_AA)
        
        viz_image(img)

        #chagne to rgb
        img_with_boxes.append(img[:,:, [2,1,0]])

    return img_with_boxes

def bbox_transform_single_box(bbox):

    cx, cy, w, h = bbox
    out_box = [[]]*4
    out_box[0] = int(np.floor(cx-w/2))
    out_box[1] = int(np.floor(cy-h/2))
    out_box[2] = int(np.floor(cx+w/2))
    out_box[3] = int(np.floor(cy+h/2))

    return out_box

def filter_batch( y_pred,config):

    #slice predictions vector
    pred_class_probs, pred_conf, pred_box_delta = utils.slice_predictions_np(y_pred, config)
    det_boxes = utils.boxes_from_deltas_np(pred_box_delta, config)

    #compute class probabilities
    probs = pred_class_probs * np.reshape(pred_conf, [config.BATCH_SIZE, config.ANCHORS_NO, 1])
    det_probs = np.max(probs, 2)
    det_class = np.argmax(probs, 2)

    #count number of detections
    num_detections = 0

    all_filtered_boxes = []
    all_filtered_scores = []
    all_filtered_classes = []

    #iterate batch
    for j in range(config.BATCH_SIZE):

        #filter predictions with non maximum suppression
        filtered_bbox, filtered_score, filtered_class = filter_prediction(det_boxes[j], det_probs[j],
                                                                          det_class[j], config)

        #you can use this to use as a final filter for the confidence score
        keep_idx = [idx for idx in range(len(filtered_score)) if filtered_score[idx] > float(config.FINAL_THRESHOLD)]

        final_boxes = [filtered_bbox[idx] for idx in keep_idx]
        final_probs = [filtered_score[idx] for idx in keep_idx]
        final_class = [filtered_class[idx] for idx in keep_idx]


        all_filtered_boxes.append(final_boxes)
        all_filtered_classes.append(final_class)
        all_filtered_scores.append(final_probs)


        num_detections += len(filtered_bbox)


    return all_filtered_boxes, all_filtered_classes, all_filtered_scores

def filter_prediction(boxes, probs, cls_idx, config):
    #check for top n detection flags
    if config.TOP_N_DETECTION < len(probs) and config.TOP_N_DETECTION > 0:
      order = probs.argsort()[:-config.TOP_N_DETECTION-1:-1]
      probs = probs[order]
      boxes = boxes[order]
      cls_idx = cls_idx[order]
      
    else:

      filtered_idx = np.nonzero(probs>config.PROB_THRESH)[0]
      probs = probs[filtered_idx]
      boxes = boxes[filtered_idx]
      cls_idx = cls_idx[filtered_idx]
    
    final_boxes = []
    final_probs = []
    final_cls_idx = []

    #go trough classes
    for c in range(config.CLASS_NO):
      idx_per_class = [i for i in range(len(probs)) if cls_idx[i] == c]            

      #do non maximum suppresion
      keep = utils.nms(boxes[idx_per_class], probs[idx_per_class], config.NMS_THRESH)
      for i in range(len(keep)):
        if keep[i]:
          final_boxes.append(boxes[idx_per_class[i]])
          final_probs.append(probs[idx_per_class[i]])
          final_cls_idx.append(c)

    return final_boxes, final_probs, final_cls_idx
    
#================================ LOAD FILES =============================================
configPath  = 'NeuralNetworks\\SqueezeDet\\config\\SQD.config'
weightsPath = 'Utils\\model.02-0.44.hdf5'
dataset = 'DummyObjectDataset'
# dataset = 'TrafficSignsBfmc'
imgPath     = 'Assets\\' + dataset + '\\images\\*.png'
lblPath     = 'Assets\\' + dataset + '\\labels\\*.txt'

images = glob.glob(imgPath)[0:]
labels = glob.glob(lblPath)[0:]

config  = make_config.load(configPath)
 
squeeze = SqueezeDet(config)
load_only_possible_weights(squeeze.model, weightsPath, verbose=False)

#================================ LOOP ===================================================
for i in range(0,len(images),4):
  imgs = []
  normImgs  = []
  anns = []
  
  for j in range(4):
    img = cv2.imread(images[j+i])
    # img = cv2.resize(img, (1000,800) )
    imgs.append(img)
    ann = load_annotation(labels[j+i], config)
    anns.append(ann)
    normImgs.append(img/255)

  imgPred  = squeeze.model.predict([normImgs])  

  a = visualize_dt_and_gt(imgs, anns , imgPred, config)


