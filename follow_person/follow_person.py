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
# from fake_drone import Drone
from tello.tello_wrapper import Drone

def bboxCenter (bbox): 
    x = (bbox[0] + bbox[2])/2
    y = (bbox[1] + bbox[3])/2
    return [x,y]

def bboxArea (bbox):
    x = abs(bbox[0] - bbox[2])
    y = abs(bbox[1] - bbox[3])
    return x * y

def getEPA (bboxArea, targetArea, totalArea):
    minArea = (targetArea - 3000)
    maxArea = (targetArea + 3000)


    
    if (minArea <= bboxArea < maxArea):
        return 0

    return - (bboxArea - targetArea)/1000

def calculate_vels(box, img_shape):
    kPX = 0.7
    kDX = 0

    kPZ = 3
    kDZ = 0

    kPA = 0.01
    kDA = 0

    ePX = 0
    eDX = 0
    velW = 0

    ePZ = 0
    eDZ = 0
    velZ = 0

    ePA = 0
    eDA = 0
    velV = 0

    height = img_shape[0]
    width = img_shape[1]
    img_center = [width/2, height/2]
    center = bboxCenter(box)

    # print(bboxArea(box))


    area = bboxArea(box)
    targetArea = 90000
    ePX = (center[0] - img_center[0])/img_center[0]
    ePZ = -((box[1] - height * 0.05)/img_center[1])
    if (ePZ > 0):
        ePZ + 0.3
    ePA = getEPA(area, targetArea, width*height)
    # print(area)

    velW = kPX * ePX + kDX * eDX
    velV = kPA * ePA + kDA * eDA
    velZ = kPZ * ePZ + kDZ * eDZ

    return (velV,0,velZ, velW)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        cfg = config.load(sys.argv[1])
    else:
        cfg = config.load("follow_person.yml")
    
    drone = Drone('', 9005)
    
    network_model = cfg.getPropertyWithDefault("Network.Model",'')
    label_map_file = cfg.getPropertyWithDefault("Network.Labels",'')
    net_type = cfg.getPropertyWithDefault("Network.Type",'')
    

    # # The camera does not need a dedicated thread, the callbacks have their owns.
    # cam = getCamera("project.mp4", "video")
    cam = getCamera(drone.drone, "Tello")
    network = TrackingNetwork(network_model,label_map_file, net_type, True)
    network.setCamera(cam)
    display_imgs = cfg.getPropertyWithDefault("show_imgs",False)

    print(drone.bateria_restante())
    drone.despegar()

    while True:

        # Make an inference on the current image
        # start_time = datetime.now()
        (predictions, boxes, scores) = network.predict()
        img = cam.get_rgb_image() 
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # elapsed = datetime.now() - start_time
        # print "elapsed {} ms. Framerate: {} fps".format(elapsed.microseconds/1000.0, 1e6/elapsed.microseconds)
        # print ("inference output", predictions, boxes, scores)
        if (len(boxes) > 0):
            (velx, vely, velz, velw) = calculate_vels(boxes[0], img.shape)
            drone.movimiento_libre(velx, vely, velz, velw)


        # Draw every detected person
        if display_imgs:
            for idx, person in enumerate(boxes):
                [xmin, ymin, xmax, ymax] = person
                cv2.rectangle(img, (xmin, ymax), (xmax, ymin), (0,255,0), 5)
            cv2.imshow("RGB", img)
            if cv2.waitKey(1) == 27:
                break

        # time.sleep(5)

    drone.aterrizar()
    drone.close()
    if display_imgs:
        cv2.destroyAllWindows()
        