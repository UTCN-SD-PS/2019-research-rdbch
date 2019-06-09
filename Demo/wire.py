#=========================================================================================
#  NOT YET COMPLETED
#=========================================================================================
import sys
sys.path.append('.')

from Demo.Process.Detector     import * 
from Demo.Process.VideoCapture import *
from Demo.Process.Visualizer   import *

import multiprocessing
from multiprocessing import Process, Pipe

import threading
from threading import Thread

#================================ PIPES ==================================================
capVizR,    capVizS     =   Pipe(duplex = False)
vizOv1R,    vizOv1S     =   Pipe(duplex = False)
ovViz1R,    ovViz1S     =   Pipe(duplex = False)

vizOv2R,    vizOv2S     =   Pipe(duplex = False)
ovViz2R,    ovViz2S     =   Pipe(duplex = False)

#================================ PROCESSES ==============================================
allProcesses = []

videoCap = VideoCapture([], [capVizS])
allProcesses.append(videoCap)

visProc = Visualizer([capVizR], [vizOv1S])
allProcesses.append(visProc)

if __name__ == '__main__':
    for proc in allProcesses:
        proc.daemon = True
        print(proc)
        proc.start()

    for proc in allProcesses:
        proc.join()