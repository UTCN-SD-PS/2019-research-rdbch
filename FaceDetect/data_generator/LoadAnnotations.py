#================================ BBOX TRANSFORM INVERSE =================================
def bbox_transform_inv(bbox):
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
def processDummyObjDataset(line, cfg, xScale, yScale):
    obj = line.strip().split(' ')

    objClass = cfg.CLASS_TO_IDX[obj[0]]

    #get coordinates
    xmin = float(obj[1]) * xScale
    ymin = float(obj[2]) * yScale
    xmax = float(obj[3]) * xScale
    ymax = float(obj[4]) * yScale
    if xmin < 0:
        xmin = 0
    if ymin < 0:
        ymin = 0

    if xmax <= xmin:
        return None
    if ymax <= ymin:
        return None

    if xmax >= cfg.IMAGE_WIDTH:
        xmin = cfg.IMAGE_WIDTH
    if ymax >= cfg.IMAGE_HEIGHT:
        ymax = cfg.IMAGE_HEIGHT
    
    return [objClass, xmin, ymin, xmax, ymax]

#================================ BBOX TRANSFORM INVERSE =================================
def load_annotation(gt_file, config, xScale, yScale):
   
    with open(gt_file, 'r') as f:
        lines = f.readlines()

    annotations = []

    #each line is an annotation bounding box
    for line in lines:
        
        procLine = processDummyObjDataset(line, config, xScale, yScale)

        if procLine is not None:
            [objClass, xmin, ymin, xmax, ymax] = procLine

            #transform to  point + width and height representation
            x, y, w, h = bbox_transform_inv([xmin, ymin, xmax, ymax])
            annotations.append([objClass, x, y, w, h])
        else:
            print( gt_file ) 
            print("Line ignored", line)

    return annotations

