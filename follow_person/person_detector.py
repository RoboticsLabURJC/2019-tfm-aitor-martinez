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

from camera import getCamera
from networks import TrackingNetwork
import cv2
import numpy as np
from datetime import datetime
import time


if __name__ == '__main__':
    if len(sys.argv) > 1:
        cfg = config.load(sys.argv[1])
    else:
        cfg = config.load("person_detector.yml")
    

    
    camera_type = cfg.getPropertyWithDefault("Camera.type","webcam")
    if (camera_type == "webcam"): 
        camera_id = int(cfg.getPropertyWithDefault("Camera.device",0))
    else: 
        camera_id = cfg.getPropertyWithDefault("Camera.device",0)
    
    network_model = cfg.getPropertyWithDefault("Network.Model",'')
    label_map_file = cfg.getPropertyWithDefault("Network.Labels",'')
    net_type = cfg.getPropertyWithDefault("Network.Type",'')
    

    # # The camera does not need a dedicated thread, the callbacks have their owns.
    cam = getCamera(camera_id, camera_type)
    network = TrackingNetwork(network_model,label_map_file, net_type, True)
    network.setCamera(cam)
    display_imgs = True

    while True:

        # Make an inference on the current image
        #start_time = datetime.now()
        img = cam.get_rgb_image()
        (predictions, boxes, scores) = network.predict(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #elapsed = datetime.now() - start_time
        #print "elapsed {} ms. Framerate: {} fps".format(elapsed.microseconds/1000.0, 1e6/elapsed.microseconds)
        # print ("inference output", predictions, boxes, scores)
        # Draw every detected person
        for idx, person in enumerate(boxes):
            [xmin, ymin, xmax, ymax] = person
            cv2.rectangle(img, (xmin, ymax), (xmax, ymin), (0,255,0), 5)
        if display_imgs:
            cv2.imshow("RGB", img)
            if cv2.waitKey(1) == 27:
                break
        time.sleep(2)
    if display_imgs:
        cv2.destroyAllWindows()