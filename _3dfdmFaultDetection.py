
# This program is created by Lanz De Guzman

#Initialization of dependencies 
import cv2
import numpy as np
from datetime import datetime
import time

#Initalization of time values
now = datetime.now()
timestr = str(now.strftime("%H:%M:%S"))
date = str(now.strftime("%d/%m/%Y"))
filename = str("results.txt")

with open(filename, "a") as f:
        f.write("\nSession: "+ date+timestr+"\n")
print('Initializing Data Output')
 

#Camera Initialization
cam = cv2.VideoCapture(0)

#Yolo Files Initalization 
folderpath = 'D:\\3dFDMYoloFiles\\obj.names'                                    # YOLO Name Fiile location
classNames = []
with open(folderpath, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

print('Loading Yolo Models')                                                     # Checkpoint Tracker

modelConfiguration = 'D:\\3dFDMYoloFiles\\custom-yolov4-tiny-detector.cfg'       # YOLO cfg file location
modelWeight = 'D:\\3dFDMYoloFiles\\custom-yolov4-tiny-detector_best.weights'     # YOLO weight file location
model = cv2.dnn.readNetFromDarknet(modelConfiguration,modelWeight)               # Loading of YOLO Models

# To run YOLO Models on GPU 
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)                             # Note that to use this OpenCV must be build within the GPU's Cuda Cores
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)                               # Will automatically switch to CPU whenever unconfigured. 

print('Yolo Initialization Successful')                                          # Model Loading Checkpoint

def recordData(name):
   currnow = datetime.now()
   with open(filename, "a") as f:
    timecurr = str(currnow.strftime("%H:%M:%S"))
    f.write(str(f'{name} - ') +timecurr+ "\n")


def findObjects(img):

    start_time = time.time()                                                     # Time initaialization to compute for FPS

    blob = cv2.dnn.blobFromImage(img, 1/255, (320,320), [0,0,0],1,crop =False)   # Converts video feed into blobs
    model.setInput(blob)

    #layerNames = model.getLayerNames()
    outputNames = model.getUnconnectedOutLayersNames()                           # Used for getting output layers

    # Object Detection Using Yolo
    detection = model.forward(outputNames)
    
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []

    confThreshold = 0.4                                                          # YOLO Confidence Treshold
    nmsThreshold = 0.5                                                           # lower the more agressive and less boxes
    
    for output in detection:
          for det in output:
              scores = det[5:]
              classId = np.argmax(scores)
              confidence = scores[classId]
              if confidence > confThreshold:
                  w,h = int(det[2]*wT) , int(det[3]*hT)
                  x,y = int((det[0]*wT)-w/2), int((det[1]*hT)-h/2)
                  bbox.append([x,y,w,h])
                  classIds.append(classId)
                  confs.append(float(confidence))
              

    indices = cv2.dnn.NMSBoxes(bbox,confs,confThreshold,nmsThreshold)
    indices = np.array(indices).flatten()                                        # Array list of detected objects
    
    # Drawing Bounding Box for every detection in indices
    for i in indices:
            i = i
            box = bbox[i]
            x,y,w,h = box[0],box[1],box[2],box[3]

            # Draws Bounding Box for every detection and display the detection type
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),5)
            cv2.putText(img,f'{classNames[classIds[i]].upper()}', (x,y-10),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0), 3)  
           
            recordData(classNames[classIds[i]].upper())                          # Calls the RecordData function purposed to record detected fault and the time it happened within a text file

    # FPS Calculation
    end_time = time.time()                                                      
    elapsed_time = end_time - start_time
    fps = 1/elapsed_time
    cv2.putText(img,str(round(fps,2)), (30,30),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0), 2)  #Displays the FPS to the video feed

while True:
       success, img = cam.read()
      
       if not success:
        break

       imgHeight, imgWidth, channels = img.shape
       #print(img.shape) ## checking the size of video feed
       
       img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
       
       findObjects(img)                                                          # Calling of Object Detection Function
       
       img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

       cv2.imshow('Image',img)

       #To stop the script press p
       if cv2.waitKey(1) & 0xFF == ord('p'):
        break