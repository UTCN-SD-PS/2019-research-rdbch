import sys
sys.path.append('.')

import cv2

import threading
from threading import Thread

import multiprocessing
from multiprocessing import Process

import numpy as np
from Demo.utils.OpenVinoInferencer import OpenVinoInferencer

class Visualizer(Process):
    #================================ CAMERA PROCESS =====================================
    def __init__(self, inPs, outPs):
        Process.__init__(self)
        self.daemon = True
        
        self.inPs   = inPs
        self.outPs  = outPs

        self.threads = []

        self.xmlModels  = ['C:\\Users\\beche\\Documents\\GitHub\\open_model_zoo\\Retail\\object_detection\\face_pedestrian\\rmnet-ssssd-2heads\\0002\\dldt\\face-person-detection-retail-0002-fp16.xml',
                           'C:\\Users\\beche\\Documents\\GitHub\\open_model_zoo\\Retail\\object_attributes\\age_gender\\dldt\\age-gender-recognition-retail-0013-fp16.xml' ]

    #================================ INIT THREADS =======================================
    def _init_threads(self):
        vizTh = Thread (target = self.visualizer)
        self.threads.append(vizTh)
        
    #================================ RUN ================================================
    def run(self):
        self._init_threads()

        for th in self.threads:
            th.daemon = True
            th.start()

        for th in self.threads:
            th.join()

    def filter_bboxes(self, res, thresh = 0.85):
        faces = []
        for box in res:
            if box[2] > thresh and box[1] == 2:
                xMin = int(box[3] *  544)
                yMin = int(box[4] *  320)
                xMax = int(box[5] *  544)
                yMax = int(box[6] *  320)

                faces.append((xMin, yMin, xMax, yMax))
        return faces

    #================================ PUBLISH IMAGE ======================================
    def visualizer(self):
        self.inf = OpenVinoInferencer(1, self.xmlModels, mode= 'GPU')
        
        while True:
            frame = self.inPs[0].recv()
            
            frame = cv2.resize(frame, dsize=None, fx=0.85, fy=0.85)

            frame = frame[:320,:,: ]
            
            infFrame  =  np.expand_dims(frame, axis = 0)
            infFrame  =  np.transpose(infFrame, [0, 3, 1, 2])
            res       =  self.inf.predict_sync(infFrame, 0)[0,0,:,:]
            
            res = self.filter_bboxes(res)
            faces = []
            
            for face in res:
                fce = frame[face[0]:face[2], face[1]:face[3], :]
                fce = cv2.resize(fce, (62,62))
                faces.append(fce)

                frame = cv2.rectangle(frame, face[:2], face[2:], (255,213,10), 2)

            faces = np.array(faces)

            print( faces.shape ) 
            cv2.imshow('Detection', frame)
            if len(faces) >=1:
                cv2.imshow('Face', faces[0, :, :, :])
                

            if cv2.waitKey(1) & 0xff == 27:
                break
                
