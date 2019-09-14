#
# Created on May, 2018
#
# @author: naxvm
#
# Based on dl-objectdetector
# https://github.com/jderobot/dl-objectdetector

import sys
import signal

import config_with_yaml as config

from camera.camera import Camera
from networks import TrackingNetwork
import cv2
import numpy as np
from datetime import datetime



signal.signal(signal.SIGINT, signal.SIG_DFL)

if __name__ == '__main__':

    cfg = config.load(sys.argv[1])

    print(cfg)
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


    # mom_path = cfg.getProperty('FollowPerson.Mom.ImagePath')

    # siamese_network = SiameseNetwork(siamese_model, mom_path)


    # app = QtWidgets.QApplication(sys.argv)
    # window = GUIClass()

    # if device_type.lower() == 'kobuki':
    #     depth_proxy = jdrc.getCameraClient('FollowPerson.Depth')
    #     depth = Camera(depth_proxy)
    #     network.setDepth(depth)
    #     t_depth = ThreadCamera(depth)
    #     t_depth.start()
    #     window.setCamera(cam, t_cam)
    #     window.setDepth(depth, t_depth)
    # else:
    #     window.setCamera(cam, t_cam)


    # motors = Motors(motors_proxy)
    # motors.setNetworks(network, siamese_network)
    # motors.setCamera(cam)
    # if device_type.lower() == 'kobuki':
    #     motors.setDepth(depth)
    # t_motors = ThreadMotors(motors)
    # window.setMotors(motors, t_motors)
    # t_motors.start()


    # window.setNetwork(network, t_network)
    # window.show()

    # # Threading GUI
    # t_gui = ThreadGUI(window)
    # t_gui.start()


    # print("")
    # print("Requested timers:")
    # print("    Camera: %d ms" % (t_cam.t_cycle))
    # print("    GUI: %d ms" % (t_gui.t_cycle))
    # print("    Network: %d ms" % (t_network.t_cycle))
    # print("    Motors: %d ms" % (t_motors.t_cycle))
    # print("")

    # sys.exit(app.exec_())
