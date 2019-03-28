import numpy as np
from LoadAnnotations import load_annotation

#================================ BATCH IOU ==============================================
def batchIou(boxes, box):
    '''Compute the Intersection-Over-Union of a batch of boxes with another
    box.

    Args:
        box1: 2D array of [cx, cy, width, height].
        box2: a single array of [cx, cy, width, height]
    Returns:
        ious: array of a float number in range [0, 1].

    '''

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

#================================ READ IMAGE AND LABELS ==================================
def read_image_and_gt(imgPaths, annPaths, config):
    labels  =   []
    bboxes  =   []
    deltas  =   []
    aidxs   =   []

    #init tensor of images
    imgs = np.zeros((
                    config.BATCH_SIZE,
                    config.IMAGE_HEIGHT, 
                    config.IMAGE_WIDTH, 
                    config.NO_CHANNELS)
                    )

    img_idx = 0

    #iterate files
    for imgName, annName in zip(imgPaths, annPaths):

        print( imgName, annName ) 

        #preprocess img
        img = cv2.imread(imgName).astype(np.float32, copy=False)
        
        imgHeight, imgWidth = img.shape[0], img.shape[1]
        
        img = cv2.resize( img, (config.IMAGE_WIDTH, config.IMAGE_HEIGHT))
        
        img = (img - np.mean(img))/ np.std(img)
        imgs[img_idx] = np.asarray(img)
        img_idx += 1

        
        # scale annotation
        xScale = config.IMAGE_WIDTH / imgWidth
        yScale = config.IMAGE_HEIGHT / imgHeight

        # load annotations
        anns = load_annotation(annName, config)

        #split in classes and boxes
        classLabels = [ann[0] for ann in anns]
        bboxLabels  = np.array([a[1:]for a in anns])

        #scale boxes
        bboxLabels[:, 0::2] = bboxLabels[:, 0::2] * xScale
        bboxLabels[:, 1::2] = bboxLabels[:, 1::2] * yScale

        bboxes.append(bboxLabels)

        aidxImage, deltaImage = [], []
        aidxSet = set()


        #iterate all bounding boxes for a file
        for i in range(len(bboxLabels)):

            #compute overlaps of bounding boxes and anchor boxes
            overlaps = batch_iou(config.ANCHOR_BOX, bboxLabels[i])

            #achor box index
            ancIndex = len(config.ANCHOR_BOX)
           
            maxOvIdx = np.argsort(overlaps)[-1]
            if overlaps[maxOvIdx] > 0:
                aidxSet.add(maxOvIdx)
                ancIndex = maxOvIdx
                    
            #TODO :  finish the data generator 

    return imgs, Y

#================================ OPEN PATHS =============================================
def openPaths(path):
    '''Open the paths from a file containing them.
    
    Arguments:
        path {str} -- file to path containing paths
    
    Returns:
        list -- parsed paths
    '''

    with open(path, 'r') as f:
        paths = f.readlines()
        for i in range(len(paths)):
            paths[i] = paths[i].strip()

    return paths

#================================ TEST ===================================================
# if __name__ == '__main__':
#     import cv2
#     import config.make_config
    
#     annPath = 'C:\\Users\\beche\\Documents\\GitHub\\DokProject-master\\Assets\\DummyObjectDataset\\labels.txt'
#     imgPath = 'C:\\Users\\beche\\Documents\\GitHub\\DokProject-master\\Assets\\DummyObjectDataset\\images.txt'

#     annPaths = openPaths(annPath)
#     imgPaths = openPaths(imgPath)
#     config = make_config.load('C:\\Users\\beche\\Documents\GitHub\\DokProject-master\\NeuralNetworks\\SqueezeDet\\config\\SQD.config')

#     read_image_and_gt(imgPaths[:config.BATCH_SIZE], annPaths[:config.BATCH_SIZE], config)
