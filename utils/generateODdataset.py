#=========================================================================================
# SCRIPT USED FOR GENERATING A DATASET WITH LETTERS SPREAD ACROSS AN IMAGE
#=========================================================================================

import cv2
import numpy as np
import random
import string

#================================ DRAW LETTER ============================================
def draw_letter(letter, img, pos, color, scale, thickness):
    cv2.putText(
        img       = img,
        text      = letter, 
        org       = pos, 
        fontFace  = cv2.FONT_HERSHEY_SIMPLEX, 
        fontScale = scale,
        thickness = thickness, 
        color     = color, 
        lineType  = cv2.LINE_AA
        )


#================================ RAND POS ===============================================
def get_random_pos(limitx, limity):
    x = random.randint(limitx[0], limitx[1])
    y = random.randint(limity[0], limity[1])

    return (x,y)

#================================ RAND COLOR =============================================
def get_random_color():
    R = random.randint(0,255)
    G = random.randint(0,255)
    B = random.randint(0,255)

    return [R,G,B]

#================================ NOT ALIKKE COLOR =======================================
def get_not_alike_color(color):
    propColor = get_random_color()

    while color_too_alike(color, propColor):
        propColor = get_random_color()

    return propColor

#================================ NOT ALIKE COLORS =======================================
def get_not_alike_colors(length):
    colors = []
    
    while(len(colors) < length):
        clr = get_random_color()

        isOk = True
        for color in colors:
            if not color_too_alike(clr, color):
                isOk = False
                break

        if isOk:
            colors.append(clr)

    return colors

#================================ RANDOM LETTER ==========================================
def get_random_letter(length):
    letters = string.ascii_uppercase[:length]
    letter = random.choice(letters)

    return letter

#================================ VIZ IMAGES =============================================
def viz_image(image, name='Window'):
    cv2.imshow(name, image)
    key = cv2.waitKey(0) & 0XFF
    if key == ord('q'):
        exit()

#================================ COLOR TOO ALIKE ========================================
def color_too_alike(color1, color2):
    R = abs(color1[0] - color2[0])
    G = abs(color1[1] - color2[1])
    B = abs(color1[2] - color2[2])
    
    totalDiff = R + G + B

    if totalDiff < 30:
        return False

    return True

#================================ THICKNESS ==============================================
def get_uniform_thickness(length, maxNo, minNo = 0):
    return np.random.uniform(minNo, maxNo, length).astype(np.int)

#================================ SCALE ==================================================
def get_uniform_scale(length, maxNo, minNo = 1.0):
    return np.random.uniform(minNo, maxNo, length)
    
#================================ POS ====================================================
def get_uniform_pos(length, sizeX, sizeY, minDist):
    poss = []
    
    while(len(poss) < length):
        p = get_random_pos(sizeX, sizeY)

        isOk = True
        for pos in poss:
            if abs(pos[0]-p[0]) < minDist and abs(pos[1]-p[1]) < minDist:
                isOk = False
                break

        if isOk:
            poss.append(p)

    return poss 


def get_bbox_boundings(letter, pos, scale, thickness):
    
    size = cv2.getTextSize(letter, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]
    pt2 = (pos[0] + size [0], pos[1] + 3)
    pt1 = (pos[0],pos[1] - size[1] )
    
    return (pt1, pt2)

#================================ CREATE IMAGE ===========================================
def create_image():
    imgSize = (800,1000,3)
    noLettersPic = random.randint(20, 40)

    thicks  =  get_uniform_thickness(noLettersPic, 8, 3)
    sizes   =  get_uniform_scale(noLettersPic, 3, 2)
    posses  =  get_uniform_pos(noLettersPic, [0 , 950], [55,760], 50)

    colors  =  get_not_alike_colors(noLettersPic + 1)
    img     =  np.full(imgSize, dtype=np.uint8, fill_value = colors[-1])
    labels  = []

    for i in range(noLettersPic):
        letter  =   get_random_letter(5)
        th      =   thicks[i]
        sz      =   sizes[i]
        pos     =   posses[i]
        clr     =   colors[i]

        draw_letter(letter, img, pos, clr, sz, th)

        bbox    =  get_bbox_boundings(letter, pos, sz, th)
        label   =  letter, bbox
        labels.append(label)
    return img, labels

#================================ WRITE FILES ============================================
def write_files():
    folderPath = '.\\Assets\\DummyObjectDataset\\'
    for i in range(10000):
        img, labels = create_image()

        imgPath = folderPath + 'images\\image_{0:07d}.png'.format(i)
        lblPath = folderPath + 'labels\\label_{0:07d}.txt'.format(i)

        with open(lblPath, 'w') as f:
            for bbox in labels:
                label = bbox[0] + ' ' + str(bbox[1][0][0]) + ' ' + str(bbox[1][0][1]) \
                                + ' ' + str(bbox[1][1][0]) + ' ' + str(bbox[1][1][1]) + '\n'
                f.write(label)
        
        viz_image(img)
        cv2.imwrite(imgPath, img)

        if i % 500 == 0:
            print( 'Processed', i ) 

if __name__ == "__main__":
    write_files()

