#
# Created on Sep, 2019
#
# @author: aitormf
#
# Based on Nacho Condes' TFG
# https://github.com/RoboticsLabURJC/2017-tfg-nacho_condes

import sys
import signal

import config_with_yaml as config

from camera.camera import Camera
from networks import TrackingNetwork
import cv2
import numpy as np
from datetime import datetime


if __name__ == '__main__':
    if len(sys.argv) > 1:
        cfg = config.load(sys.argv[1])
    else:
        cfg = config.load("person_detector.yml")
    

    camera_id = int(cfg.getPropertyWithDefault("Camera",0))
    
    network_model = cfg.getPropertyWithDefault("Network.Model",'ssdlite_mobilenet_v2_coco_2018_05_09_trt.pb') 
    

    # # The camera does not need a dedicated thread, the callbacks have their owns.
    cam = Camera(camera_id)
    network = TrackingNetwork(network_model)
    network.setCamera(cam)
    display_imgs = True

    while True:

        # Make an inference on the current image
        #start_time = datetime.now()
        network.predict()
        img = cam.get_rgb_image()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #elapsed = datetime.now() - start_time
        #print "elapsed {} ms. Framerate: {} fps".format(elapsed.microseconds/1000.0, 1e6/elapsed.microseconds)
        print ("inference output", network.predictions, network.boxes, network.scores)
        # Draw every detected person
        for idx, person in enumerate(network.boxes):
            [xmin, ymin, xmax, ymax] = person
            cv2.rectangle(img, (xmin, ymax), (xmax, ymin), (0,255,0), 5)
        if display_imgs:
            cv2.imshow("RGB", img)
            if cv2.waitKey(1) == 27:
                break
    if display_imgs:
        cv2.destroyAllWindows()