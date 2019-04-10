# FaceDetector

## Short Description

This is the repo of the FaceDetector project that I am curently doing in collaboration with Endava. In this repository one will be able to find the implemented code and the periodically activity report.

## Project status

- [x] Research Period
    - [x] Search the web for best suited object detection algorithm
    - [x] Read about the implementation of algorithms
    - [x] Succesfullt choose one(Single Shot Detector)
    
- [x] Implementation
    - [x] Make a dummy dataset to better understand the data
    - [x] Implement the neural network
    - [x] Implement the interpretation graph
    - [x] Implement the training algorithm
    - [x] Implement visualization 
    - [x] Experiment on the dummy dataset
    
- [ ] Training 
    - [x] Find and download a suited face dataset
    - [ ] Start training
    
- [ ] Deploying the model
    - [ ] Optimize the model for inference
    - [ ] Fine-tune the model for desired applications
    - [ ] Make a nice user interface
    
- [ ] Draw conclusion

## Depencies

<pre>
tensorflow>=1.13.0
keras>=2.2.4
numpy>=1.16.2
opencv-python>=4.0.0.21
easydict>=1.9
</pre>

## Installation guide

For running this project I reccomend creating a virtual environment after  clonning it.
<pre>
$ cd {REPOSITORY_ROOT}
$ virtualenv venv                # this command will create a new virtual environment in the folder venv
$ . .\venv\scripts\activate      # this command will activate your virtual env
</pre>

To install the dependencies, run:
<pre>
$ pip install -r Assets\requirements.txt
</pre>

This stepts should setup your environment.
 
## How to run an example

For the moment there isn't an example available.
