import os
import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IEPlugin


# ===================================== NCS INFERENCER ===================================
class OpenVinoInferencer:

    # ===================================== __INIT__ =====================================
    def __init__(self, id, xmlModels, noReq = 2, verbose = False, mode = 'MYRIAD'):
        """NCS Inferencer
        
        Arguments:
            id {int} -- id of the movidius stick(ex. 1,2,3... etc)
            xmlModels {list(Str)} -- a list of xml models of the OpenVino networks
        
        Keyword Arguments:
            noReq {int} -- the number of requests per device(threads openvino makes) 
                            (default: {2})
            verbose {bool} -- verbose for NCS platform (default: {False})
        """

        assert noReq >= 2, 'The number of requests should be grater than 2'
        
        self._id            =   id
        self.nets           =   []
        self.execNets       =   []
        self.noReq          =   noReq
        self.mode           =   mode
        self._load_models(xmlModels, noReq, verbose)

        self.inBlobs    = [next(iter(net.inputs))  for net in self.nets]
        # self.outBlobs   = [next(iter(net.outputs)) for net in self.nets]
        self.outBlobs = []
        for net in self.nets:
            for out in net.outputs:
                self.outBlobs.append(out)

        # ids of requests for running multithreadded
        self.cReqIds                 =   [0 for _ in range(len(self.nets))]
        self.nextReqIds              =   [1 for _ in range(len(self.nets))]

    
    # ===================================== LOAD MODEL ===================================
    def _load_models(self, xmlModels, noReq, verbose):
        """Create an OpenVino plugin and load the .xml models onto the NCS device
        
        Arguments:
            xmlModels {list(str)} -- list of OpenVino XML models
            noReq {int} -- the number of request per network
            verbose {bool} -- verbose NCS device
        """

        # Plugin initialization for specified device and load extensions library 
        if self.mode == 'MYRIAD':
            self.plugin = IEPlugin(device='GPU')

            if verbose:
                self.plugin.set_config({"VPU_LOG_LEVEL":"LOG_DEBUG"})

        elif self.mode == 'CPU':
            self.plugin = IEPlugin(device='CPU')
            
        elif self.mode == 'GPU':
            self.plugin = IEPlugin(device='GPU')

        for xmlModel in xmlModels:
            modelBin     =  os.path.splitext(xmlModel)[0] + ".bin"
            net          =  IENetwork(model=xmlModel, weights=modelBin)

            print( 'loaded', net ) 
            self.nets.append(net)
            self.execNets.append(self.plugin.load(network=net, num_requests=self.noReq))

    # ===================================== PREDICT ASYNC ================================
    def predict_async(self, image, netNo):
        """Predict on ncs device
        
        Arguments:
            image {mat} -- image
            netNo {int} -- the number of network that will make the inference
        """
        
        self.execNets[netNo].start_async(
                                request_id  =  self.nextReqIds[netNo], 
                                inputs      =  {self.inBlobs[netNo]: image})
    
        res = None

        # get return status from OpenVino Doc ( 0 means OK)
        if self.execNets[netNo].requests[self.cReqIds[netNo]].wait(-1) == 0:
            res = self.execNets[netNo]                      \
                            .requests[self.cReqIds[netNo]]   \
                            .outputs[self.outBlobs[netNo]]


        self.cReqIds[netNo]    =  self.cReqIds[netNo]   + 1 \
                                if self.cReqIds[netNo]   < self.noReq - 1 else 0

        self.nextReqIds[netNo]  =  self.nextReqIds[netNo] + 1 \
                                if self.nextReqIds[netNo] < self.noReq - 1 else 0
        
        return res

    #================================ PRED SYNC ==========================================
    def predict_sync(self, image, netNo, resNo = None):
        if not resNo:
            resNo = [netNo] 
        res = self.execNets[netNo].infer({self.inBlobs[netNo]: image})
        fRes = []

        for no in resNo:
            fRes.append(res[self.outBlobs[no]])

        return fRes

