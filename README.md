# Not So Random Forest

Automating large-scale urban tree monitoring

#### Abstract

Locate trees and identify species in street view imagery 

Trees are an important part of a city’s landscape, providing a variety of benefits. Maintaining a city’s tree cover is a costly affair, requiring thousands of volunteer hours and several hundred thousand dollars every year. 

**Not So Random Forest** is an AI platform that aims to solve this problem by automating large-scale urban tree monitoring. Using deep learning techniques on Google street view imagery, the platform enables, at the click of a button, rapid localization, species classification and health monitoring of trees on a city-wide scale. 

Because labeled tree imagery is not freely available, I build and train CNN models on hand-labeled tree images. Installation instructions, labeling approaches and model description are given below.

[Slides](http://bit.ly/notsorandomforest)


## Models

Tree detection and species classification proceeds in two steps. First a detection model locates trees in an image of interest and outputs the detected trees as standalone images. The output of the detector are then piped to a classifier that predicts the species of the trees. 

### Object Detection

The project leverages RetinaNet, a state-of-the-art object detection model. RetinaNet improves upon previous architectures by implementing "focal loss", a weighted loss function, that addresses imbalance between objects of interest and the image background.
