#=========================================================================================
# SCRIPT USED AT DEMO (VERY UGLY IMPLEMENTATION)
#=========================================================================================
import os
import cv2
import time
import numpy as np
from openvino.inference_engine import IENetwork, IEPlugin
from utils.OpenVinoInferencer import OpenVinoInferencer

#================================ MAKE INFERENCE =========================================
def make_infer(frame, thresh):

    # prepare for inference
    frame     =  cv2.resize(frame, dsize=None, fx=0.85, fy=0.85)
    frame     =  frame[:320,:,: ]
    infFrame  =  np.expand_dims(frame, axis = 0)
    infFrame  =  np.transpose(infFrame, [0, 3, 1, 2])
    
    res       =  inf.predict_sync(infFrame, 0, [0, 1])
    
    resBox    =  res[0][0,0,:,:]
    persBox   =  res[1][0,0,:,:]
    noPers      = 0
    faces = []

    for box in resBox:
        if box[2] > thresh and box[1] == 2:
            
            xMin = int(box[3] *  544)
            yMin = int(box[4] *  320)
            xMax = int(box[5] *  544)
            yMax = int(box[6] *  320)
            if xMax-xMin < 5 or yMax-yMin <5:
                continue 
            fce = frame[yMin:yMax, xMin:xMax, :]

            if 0 in fce.shape:
                continue

            noPers += 1
            fce = cv2.resize(fce, (62,62))
            faces.append(fce)
            
            frame = cv2.rectangle(frame, (xMin, yMin), (xMax, yMax), (255,213,10), 2)
    
    for box in persBox:
        if box[2] > threshPers:
            xMin = int(box[3] *  544)
            yMin = int(box[4] *  320)
            xMax = int(box[5] *  544)
            yMax = int(box[6] *  320)
            if xMax-xMin < 5 or yMax-yMin <5:
                continue 
   
            frame = cv2.rectangle(frame, (xMin, yMin), (xMax, yMax), (0,213,10), 2)
    return frame, noPers, faces

#================================ MAKE FACE IMAGE ========================================
def make_face_image(faces, inf):
    facesImg = np.zeros((1000,1000,3)).astype(np.uint8)
    noX = 0
    noY = 0
    size = 250


    if len(faces) >0:
        faces = np.array(faces)
        genders = []

        for i in range(len(faces)):
            infFaces = np.transpose(faces[i:i+1, :, :, :], [0,3,1,2])
            res = inf.predict_sync(infFaces, 1, [2,3])
            age = res[0]
            age = age.ravel([0])
            aux = faces[i]
            aux = cv2.resize(aux, (size,size))
            gender = np.argmax(res[1].ravel())
            gender = 'F' if gender == 0 else 'M'
            cv2.putText(aux,gender + ' '+ str(int(age*100)), (5,size-10),  font, fontScale,fontColor,lineType)
          

            try:
                facesImg[noY:noY+size, noX:noX+size, :] = aux
            except Exception:
                pass

            noX += size

            if noX >=1000:
                noY += size
                noX = 0

    return facesImg
#================================ MAIN ===================================================
if __name__ == '__main__':
    #================================ PARAMS =============================================
    paths = ['path/to/person/detector',
            'path/to/age/gender/classification']
  # fonts
    thresh      = 0.9
    threshPers  = 0.6
    font        = cv2.FONT_HERSHEY_SIMPLEX
    fontScale   = 0.7
    fontColor   = (255,255,0)
    lineType    = 2
    
    # objects
    inf  = OpenVinoInferencer(1, paths, mode= 'GPU')
    cap = cv2.VideoCapture(0)

    tPrev = time.time()

    #================================ MAIN ===============================================
    while True:
        t1 = time.time()
        ret, frame = cap.read()
    
        if not ret:
            break
        
        frame, noPers, faces = make_infer(frame, thresh)
        faceImg = make_face_image(faces, inf)

        if t1 - tPrev >= 2:
            tPrev = t1
            print( 'No pers: ', noPers )
            
        cv2.imshow("detection", frame)
        cv2.imshow("Faces", faceImg)

        if cv2.waitKey(1) & 0xff == 27:
            break