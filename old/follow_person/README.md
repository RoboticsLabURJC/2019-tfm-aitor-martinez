
# Follow Person

## Requirements
* All the necessary Python packages have been annotated for <code>pip</code> to install them automatically. To do so, run:
`pip3 install -r requirements.txt`

## Person Detector 

Application to detect persons from images of a webcam. It uses Deep Learning to do so: a detection CNN (_SSD_ Architecture).

The implementation (network models and device) can be customized using the YML file (`person_detector.yml`)

### How to use

**0. Tune your execution**

* Object Detection model: you can download a pre-trained network model from the [TensorFlow Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). Choose among those which output _boxes_ (not regions). Just download the .zip and keep the `.pb` file (which contains the frozen graph structure and weights). Place it into the `resources` directory, and indicate its name in the suitable YML file (in the `FollowPerson.Network.Model` node). In addition, you will have to indicate in the `FollowPerson.Network.Dataset` node which was the training dataset of that model (you can check it in the Model Zoo page).


**1. Launch the application**

`python3 person_detector.py person_detector.yml`