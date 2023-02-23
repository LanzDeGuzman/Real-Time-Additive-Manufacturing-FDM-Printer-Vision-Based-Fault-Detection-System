# Real-Time-Additive-Manufacturing-FDM-Printer-Vision-Based-Fault-Detection-System

Additive Manufacturing, commonly known as 3D printing, has gained popularity with its ability to produce highly versatile products. Consequent to this versatility is potency to various errors and thus low-reliability factor. In order to increase reliability and address print errors, a real-time vision-based fault detection system for FDM 3D printers is developed. The fault detection system is centered on detecting errors during the production or printing process, specifically the occurrence of blobs, cracks, spaghetti, stringing, and under-extrusion printing errors. The repository used CNN machine vision techniques, specifically YOLOv4 Tiny, to develop an FDM 3d printing fault detection model for real-time application. 

## Dataset Preparation 
In preparing the dataset for training, an open-source data set for 3d printing defects, together with additional images available from Google, were used. A total of 257 images comprised the dataset containing cases of **blobs, cracks, spaghetti, stringing, and under-extrusion printing defects.** VOTT & Roboflow was used to annontate images and configure the train test split.

## Training the Dataset 
In training the detection model, Google Collab, a cloud-based environment, was used. Using YOLOv4-Tiny’s darknet architecture, cfg, weights, and name files of the trained model were generated after 10000 iterations. 

Google Collab Training Notebook: https://colab.research.google.com/drive/1Qf0HmtpIOLA8eBm2M-YmHd4b_GGjd-be?usp=sharing
** Data Set Created is already attached to the notebook, can also be used to train other darknet based detection models

#### YOLOv4 Tiny Training Results 

<p align="center"> 
    <img src="https://user-images.githubusercontent.com/97860488/221009642-d7ecf97e-c952-438d-a1a3-e6f818aeaa78.png">
</p>

After 10,000 iterations of training, the finalized YOLOv4 model with 5 class detection rendered a **mean average precision(mAP) of 30.24% and 42.96% mAP** at its best weight file generated. To validate the model's accuracy and effectiveness, the model was executed in real-time using a web camera.

## How the code works 
Once YOLO files are initialized, the camera's feed is displayed, and whenever a fault is detected, a bounding box together with its corresponding class is displayed in the feed. Simultaneous to the detection and displaying of bounding boxes, detection results are recorded into a text file. The results text file includes the session when the program is initiated. The session includes information regarding what date and time the program script was executed.

<p align="center"> 
  <img src= "https://user-images.githubusercontent.com/97860488/221025586-d055d163-e6f7-4643-a107-f22324e8a4f2.PNG" height= "200"/> <img src="https://user-images.githubusercontent.com/97860488/221025430-87548c5a-929e-4c78-b892-98492363d6f2.PNG" width="200" height="200"/>
</p>

## Sample Results
<p align="center"> 
  <img src="https://user-images.githubusercontent.com/97860488/221027525-6e94f1a6-ceb2-4aac-9ea8-de85c1d312c3.PNG" height= "200"/> <img src="https://user-images.githubusercontent.com/97860488/221027527-a28a3d89-a600-4375-9def-043eb58dd41a.PNG" height= "200"/>
 <img src="https://user-images.githubusercontent.com/97860488/221027509-fbf21184-b47b-405a-b738-b0e21af380d1.PNG" height= "200"/> <img src="https://user-images.githubusercontent.com/97860488/221027499-da13a9b1-4ab6-4b05-9d3f-cac1cbe3298e.PNG" height= "200"/>
</p>

## Future Work 
To improve the groundwork, expounding and increasing the image database in training the model is recommended in order to increase the model’s performance and accuracy. Moreover, the use of newer YOLO architectures to evaluate the differences and improvements is also recommended. It is also recommended to pursue the initail goal of this repository to develop an automated fault detection and health monitoring system based on the FDM 3d Printers output, where after a print, whenever an error is  detected, the system would output what problems the 3d printer should address,  e.g., increasing heat, adjusting material supply, and motor calibration. Adding a real time graph visualiztion of the report can also be explored.

#### Collaborators
Joshua Andrei A. Jawod, Kristoffer Sean N. Wong, Denzhel Karl T. Rostata
