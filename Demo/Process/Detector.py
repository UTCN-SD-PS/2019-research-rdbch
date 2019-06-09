import sys
sys.path.append('.')

import time
from threading import Thread
from multiprocessing import Process
from Demo.utils.OpenVinoInferencer import OpenVinoInferencer

class Detector(Process):

    #================================ INIT ===============================================
    def __init__(self, inPs, outPs):
        Process.__init__(self)

        # in/out pipes
        self.inPs  = inPs
        self.outPs = outPs

        # configure inferencers
        self.noDevices  = 1
        self.xmlModels  = ['C:\\Users\\beche\\Documents\\GitHub\\open_model_zoo\\Retail\\object_detection\\face_pedestrian\\rmnet-ssssd-2heads\\0002\\dldt\\face-person-detection-retail-0002-fp16.xml',
                           'C:\\Users\\beche\\Documents\\GitHub\\open_model_zoo\\Retail\\object_attributes\\age_gender\\dldt\\age-gender-recognition-retail-0013-fp16.xml' ]
        self.noReq      = 2
        
        # jobs for the process
        self.threads = []

    #================================ RUN ================================================
    def run(self):
        self.inf = OpenVinoInferencer(1, self.xmlModels, mode = 'GPU')
        self._init_threads()

        for th in self.threads:
            th.daemon = True
            th.start()

        for th in self.threads:
            th.join()
    
    #================================ INIT THREADS ==============================================
    def _init_threads(self):
        
        laneTh = Thread(target = self._infer_lane_thread, args=(self.inPs[0], [self.outPs[0]], 0))
        self.threads.append(laneTh)

        objTh = Thread(target = self._infer_lane_thread, args=(self.inPs[1], [self.outPs[1]], 1))
        self.threads.append(objTh)

    #================================ NCS THREAD =================================================
    def _infer_lane_thread(self, inP, outPs, netNo):
        
        while True:
            
            image = inP.recv()

            res = self.inf.predict_async(image, netNo)
 
            if res is not None:
                for outP in outPs:
                    outP.send(res)