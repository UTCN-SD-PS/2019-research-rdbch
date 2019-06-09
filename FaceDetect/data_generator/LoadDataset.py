import cv2
import numpy as np
from FaceDetect.data_generator.LoadAnnotations import load_annotation

#================================ SPARSE TO DENSE ========================================
def sparse_to_dense(spIdxs, outputShape, values, defaultValue=0):


    assert len(spIdxs) == len(values), 'Indexes and values have different length'

    array = np.ones(outputShape) * defaultValue
    
    for idx, value in zip(spIdxs, values):
        array[tuple(idx)] = value
    
    return array

#================================ BATCH IOU ==============================================
def batch_iou(boxes, box):

    lr = np.maximum(
        np.minimum(boxes[:,0] + 0.5 * boxes[:,2], box[0] + 0.5 * box[2]) - \
        np.maximum(boxes[:,0] - 0.5 * boxes[:,2], box[0] - 0.5 * box[2]),
        0
    )
    tb = np.maximum(
            np.minimum(boxes[:,1] + 0.5 * boxes[:,3], box[1] + 0.5 * box[3]) - \
            np.maximum(boxes[:,1] - 0.5 * boxes[:,3], box[1] - 0.5 * box[3]),
            0
    )
    inter = lr * tb
    union = boxes[:,2] * boxes[:,3] + box[2] * box[3] - inter
    
    return inter/union

#================================ GET ANCHOR INDEX =======================================
def get_anchor_index(config, aIdxSet, bboxLabel):
    
    #compute overlaps of bounding boxes and anchor boxes
    overlaps = batch_iou(config.ANCHOR_BOX, bboxLabel)

    #achor box index
    ancIndex = config.ANCHORS_NO
        
    #sort for biggest overlaps
    for ovIndex in np.argsort(overlaps)[::-1]:    
        #when overlap is zero break
        if overlaps[ovIndex] <= 0:
            break

        #if one is found add and break
        if ovIndex not in aIdxSet:
            aIdxSet.add(ovIndex)
            ancIndex = ovIndex
            break

    # if the largest available overlap is 0, choose the anchor box with the 
    # one that has the smallest square distance
    if ancIndex == config.ANCHORS_NO:
        dist = np.sum(np.square(bboxLabel - config.ANCHOR_BOX), axis=1)
        for dist in np.argsort(dist):
            if dist not in aIdxSet:
                aIdxSet.add(dist)
                ancIndex = dist
                break

    return ancIndex

#================================ GET OUTPUTS ============================================
def get_dense_outputs(config, labels, ancIndexes, deltas, bboxes):
    #we need to transform this batch annotations into a form we can feed into the model
    labelIdxs, bboxIdxs, deltaValues, maskIdxs, boxValues  = [], [], [], [], []

    ancIdxSet = set()


    for i in range(len(labels)):
        for j in range(len(labels[i])):

            if (i, ancIndexes[i][j]) not in ancIdxSet:
                ancIdxSet.add((i, ancIndexes[i][j]))

                labelIdxs.append([i, ancIndexes[i][j], labels[i][j]])
                
                maskIdxs.append([i, ancIndexes[i][j]])
                
                bboxIdxs.extend([[i, ancIndexes[i][j], k] for k in range(4)])
                
                deltaValues.extend(deltas[i][j])
                
                boxValues.extend(bboxes[i][j])

    #the boxes where an object is detected have value 1
    inputMask =  np.reshape(
            sparse_to_dense(
                        spIdxs = maskIdxs,
                        outputShape = [config.BATCH_SIZE, config.ANCHORS_NO],
                        values = [1.0] * len(maskIdxs)
                        ),
            [config.BATCH_SIZE, config.ANCHORS_NO, 1]
            )


    # put into a 3d tensor, for each image, when a bounding box is detected, 
    # put the bounding boxes deltax [cx, cy, w, h] (the label bbox)
    inputDelta =  sparse_to_dense(
                    spIdxs = bboxIdxs, 
                    outputShape = [config.BATCH_SIZE, config.ANCHORS_NO, 4],
                    values = deltaValues
                )

    # put the mathing bounding box anchor from the config file (the config bbox)
    inputBox =  sparse_to_dense(
                    spIdxs = bboxIdxs, 
                    outputShape = [config.BATCH_SIZE, config.ANCHORS_NO, 4],
                    values = boxValues
                )

    # where a bounding box is found, put the class index/type (classification style)
    inputLabels = sparse_to_dense(
                    spIdxs = labelIdxs,
                    outputShape = [config.BATCH_SIZE, config.ANCHORS_NO, config.CLASS_NO],
                    values = [1.0] * len(labelIdxs)
                )

    # #concatenate ouputs
    output = np.concatenate(
                        (inputMask, inputBox,  inputDelta, inputLabels), 
                        axis=-1
                    ).astype(np.float32)

    return output

#================================ READ IMAGE AND LABELS ==================================
def read_image_and_label(imgPaths, annPaths, config):

    labels, bboxes, deltas, ancIndexes = [], [], [] ,[]

    #init tensor of images
    imgs = np.zeros((
                    config.BATCH_SIZE,
                    config.IMAGE_HEIGHT, 
                    config.IMAGE_WIDTH, 
                    config.NO_CHANNELS)
                    )

    imgIndex = 0

    #iterate files
    for imgName, annName in zip(imgPaths, annPaths):

        #preprocess img
        img = cv2.imread(imgName).astype(np.float32)

        imgHeight, imgWidth = img.shape[0], img.shape[1]
        
        img = cv2.resize( img, (config.IMAGE_WIDTH, config.IMAGE_HEIGHT))
 
        # replace standardization with normalization
        img = img/255

        imgs[imgIndex] = np.asarray(img)
        imgIndex += 1

        # scale annotation
        xScale = config.IMAGE_WIDTH / imgWidth
        yScale = config.IMAGE_HEIGHT / imgHeight
        # print( yScale ) 
        # print( xScale ) 

        
        # load annotations
        anns = load_annotation(annName, config, xScale, yScale)
    

        #split in classes and boxes
        classLabels = [ann[0] for ann in anns]
        
        bboxLabels  = np.array([a[1:]for a in anns])

        bboxes.append(bboxLabels)

        aIdxImage, deltaImage = [], []
        aIdxSet = set()

        for i in range(len(bboxLabels)):

            ancIndex = get_anchor_index(config, aIdxSet, bboxLabels[i])

            #compute deltas for regression
            cx, cy, boxW, boxH = bboxLabels[i]
            delta = [0] * 4
            delta[0] = (cx - config.ANCHOR_BOX[ancIndex][0]) / config.ANCHOR_BOX[ancIndex][2]
            delta[1] = (cy - config.ANCHOR_BOX[ancIndex][1]) / config.ANCHOR_BOX[ancIndex][3]
            delta[2] = np.log(boxW / config.ANCHOR_BOX[ancIndex][2] )
            delta[3] = np.log(boxH / config.ANCHOR_BOX[ancIndex][3] )

            aIdxImage.append(ancIndex)
            deltaImage.append(delta)

        deltas.append(deltaImage)
        ancIndexes.append(aIdxImage)
        labels.append(classLabels)
    

    #convert indexes 
    output = get_dense_outputs(config, labels, ancIndexes, deltas, bboxes)

    return imgs, output

