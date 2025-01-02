# Clipper: A Dual-Stage Solution for Detecting and Classifying Bicycles and E-Bikes

Clipper is a performant, adaptable yet simple pipeline for bicycle and e-bike detection.
The current pipeline is as follows:
- Pass the image to a YOLO model `yolov8s-worldv2` with the classes: [`electric-bike`, `electric-bicycle`, `e-bike`, `bicycle`]. We discard the classes completely in this step and focus on locating the objects.
- From each detected object, we crop it and feed it to a clip model, specifically `RN50-quickgelu` with the classes set to: [`an e-bike`, `a bicycle`, `an electric-bike`, `an electric-bicycle`]. In this step we finally get the class of the current object identified by the YOLO model.

It's that simple!

### More updates, R&D, paper coming very soon!