# Clipper: A Dual-Stage Pipeline for Detecting and Classifying Bicycles and E-Bikes

**Clipper** is a performant, adaptable, yet simple pipeline for detecting bicycles and e-bikes. Its simplicity, with its core having open-vocabulary, pre-trained models, makes it versatile and ready to other domains. 

The proposed pipeline consists of two sequential stages:  

1. **Object Detection with YOLO**
   An input image is processed using a YOLO model (`yolov8s-worldv2`), trained to detect object instances belonging to the classes: [`electric-bike`, `electric-bicycle`, `e-bike`, `bicycle`]. At this stage, the primary focus is on accurately locating objects in the image. The specific class labels from YOLO are discarded, as the goal is to identify object bounding boxes in this stage.

2. **Classification with CLIP**
   Each detected object is cropped and passed to a CLIP-based classification model (`RN50-quickgelu`) with the following class labels: [`an e-bike`, `a bicycle`, `an electric-bike`, `an electric-bicycle`]. This stage classifies each detected object, leveraging the robustness and adaptability of pre-trained ViT models.  

This dual-stage approach uses state-of-the-art open-vocabulary object detection and image classification models, ensuring both accuracy and scalability. By using pre-trained models, **Clipper** achieves domain independence, allowing it to be easily adapted for other object detection and classification tasks. Future work includes fine-tuning both the YOLO and CLIP models to further optimize performance for bicycle and e-bike detection.

**Stay tuned for additional updates, R&D progress, and a detailed paper describing this pipeline in depth.**
