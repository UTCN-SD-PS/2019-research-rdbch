#================================ BBOX TRANSFORM INVERSE =================================
def bbox_transform_inv(bbox):
    '''Transform a bounding box from shape [xmin, ymin, xmax, ymax] to [Cx, Cy, w, h]
    
    Arguments:
        bbox {tuple} -- box
    
    Returns:
        tuple -- the transformed box
    '''

    xmin, ymin, xmax, ymax = bbox
    out_box = [[]]*4

    width       = xmax - xmin + 1.0
    height      = ymax - ymin + 1.0
    out_box[0]  = xmin + 0.5*width
    out_box[1]  = ymin + 0.5*height
    out_box[2]  = width
    out_box[3]  = height

    return out_box

# ===================================== PROCESS DUMMY OBJ DATASET ========================
def process_dummy_obj_dataset(line, cfg):
    '''Take a line of the label from the generated dataset and returns a tuple with 
    required atributes [class, [xmin, ymin, xmax, ymax]]
    
    Arguments:
        line {str} -- the line from the label file
        cfg {dict} -- easyDict file for configuration
    
    Returns:
        tuple -- [class, [xmin, ymin, xmax, ymax]]
    '''

    obj = line.strip().split(' ')
    objClass = cfg.CLASS_TO_IDX[obj[0]]
    
    #get coordinates
    xmin = float(obj[1])
    ymin = float(obj[4])
    xmax = float(obj[3])
    ymax = float(obj[2])

    if xmax <= xmin:
        return None
    if ymax <= ymin:
        return None

    if xmin < 0:
        xmin = 0
    if ymin < 0:
        ymin = 0

    if xmax >= cfg.IMAGE_WIDTH:
        xmin = cfg.IMAGE_WIDTH
    if ymax >= cfg.IMAGE_HEIGHT:
        ymax = cfg.IMAGE_HEIGHT
    
    return [objClass, xmin, ymin, xmax, ymax]

#================================ BBOX TRANSFORM INVERSE =================================
def load_annotation(gt_file, config):
    '''Load every annotaion for a file
    
    Arguments:
        gt_file {str} -- path to the label file
        config {dict} -- easyDict file for configuration
    
    Returns:
        tuple -- a list of list containing every annotation 
    '''

    with open(gt_file, 'r') as f:
        lines = f.readlines()
    f.close()

    annotations = []

    #each line is an annotation bounding box
    for line in lines:
        
        procLine = process_dummy_obj_dataset(line, config)
        
        if procLine is not None:
            [objClass, xmin, ymin, xmax, ymax] = procLine

            #transform to  point + width and height representation
            x, y, w, h = bbox_transform_inv([xmin, ymin, xmax, ymax])

            annotations.append([objClass, x, y, w, h])
        else:
            print("Line ignored due to invalid coordinates.", line)

    return annotations


# ===================================== TEST =============================================
# import make_config
# import cv2


# annPath = 'C:\\Users\\beche\\Documents\\GitHub\\DokProject-master\\Assets\\DummyObjectDataset\\Labels\\label_0009999.txt'
# imgPath = 'C:\\Users\\beche\\Documents\\GitHub\\DokProject-master\\Assets\\DummyObjectDataset\\Images\\image_0009999.png'
# config = make_config.load('NeuralNetworks\\SqueezeDet\\config\\SQD.config')

# img = cv2.imread(imgPath)
# p = load_annotation(annPath, config)

# cv2.imshow('window',img)
# cv2.waitKey(0)
# print(p) 