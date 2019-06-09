import sys
sys.path.append('.')

import cv2

import threading
from threading import Thread

import multiprocessing
from multiprocessing import Process

class VideoCapture(Process):
    #================================ CAMERA PROCESS =====================================
    def __init__(self, inPs, outPs):
        Process.__init__(self)
        self.daemon = True
        
        self.inPs   = inPs
        self.outPs  = outPs

        self.threads = []

    #================================ INIT THREADS =======================================
    def _init_threads(self):
        pubTh = Thread (target = self.publish_image)
        self.threads.append(pubTh)
        
    #================================ RUN ================================================
    def run(self):
        self._init_threads()

        for th in self.threads:
            th.daemon = True
            th.start()

        for th in self.threads:
            th.join()

    #================================ PUBLISH IMAGE ======================================
    def publish_image(self):
        cap = cv2.VideoCapture(0)
    
        while True:
            ret, frame = cap.read()
            
            if  ret:
                for outP in self.outPs:
                    outP.send(frame)
                
