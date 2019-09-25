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
from networks import TrackingNetwork, SiameseNetwork, PersonTracker
import cv2
import numpy as np
from datetime import datetime
from cprint import cprint

class Follow_Person:
    def __init__(self, network, siamese_net, cam):
        self.network = network
        self.siamese_network = siamese_net


        self.last_center = (0, 0)
        self.threshold = 60
        self.face_thres = 1.0

        
        self.persons = []
        self.faces = []

        self.center_coords = [self.network.original_width/2, self.network.original_height/2]

        self.person_tracker = PersonTracker(same_person_thr=80)
        self.person_tracker.setSiameseNetwork(siamese_network)

        self.camera = cam



    def follow(self):
        full_image = self.camera.get_rgb_image()
        img2show = cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR)

        self.network.predict()
        self.detection_boxes = self.network.boxes
        self.detection_scores = self.network.scores

        for xmin, ymin, xmax, ymax in self.detection_boxes:
            cv2.rectangle(img2show, (xmin, ymax), (xmax, ymin), (0,0,255), 5)

        self.persons = self.person_tracker.evalPersons(self.detection_boxes, self.detection_scores, full_image)
        self.faces = self.person_tracker.getFaces(full_image)

        cprint.info('\t........%d/%d faces detected........' % (len(self.faces), len(self.persons)))

        mom_found_now = False
        # Iteration over all faces and persons...
        for person in self.persons:

            if person.is_mom:
                self.mom_coords = person.coords
                mom_found_now = True
                break
            else:
                faces = person.ftrk.tracked_faces
                if len(faces) > 0:
                    face = faces[0]
                    [f_width, f_height] = [face[2] - face[0], face[3] - face[1]]
                    f_total_box = np.zeros(4, dtype=np.int16)
                    f_total_box[:2] = person[:2] + face[:2]
                    f_total_box[2:4] = f_total_box[:2] + [f_width, f_height]
                    cropped_face = full_image[f_total_box[1]:f_total_box[3], f_total_box[0]:f_total_box[2], :]
                    # We compute the likelihood with mom...
                    dist_to_mom = self.siamese_network.distanceToMom(cropped_face)
                    if dist_to_mom < self.face_thres:
                        # Unset other moms
                        for idx2 in range(len(self.persons)):
                            self.person_tracker.tracked_persons[idx2].is_mom = False
                        # And set that person to mom.
                        self.person_tracker.tracked_persons[idx].is_mom = True
                        self.mom_coords = person.coords
                        mom_found_now = True
                        break

        if mom_found_now:
            cprint.ok("\t\t  Mom found")
            cprint.ok(str(self.mom_coords))            
            #[xmin, ymin, xmax, ymax] = self.mom_coords
            #cv2.rectangle(img2show, (xmin, ymax), (xmax, ymin), (0,255,0), 5)
        else:
            cprint.warn("\t\t  Looking for mom...")

        return img2show

    

signal.signal(signal.SIGINT, signal.SIG_DFL)

if __name__ == '__main__':

    if len(sys.argv) > 1:
        cfg = config.load(sys.argv[1])
    else:
        cfg = config.load("follow_person.yml")

    camera_id = int(cfg.getPropertyWithDefault("Camera",0))
    
    network_model = cfg.getPropertyWithDefault("Network.Model",'ssdlite_mobilenet_v2_coco_2018_05_09_trt.pb')
    siamese_model = cfg.getPropertyWithDefault("Network.SiameseModel",'siamese_model.pb')
    mom_path = cfg.getPropertyWithDefault('Mom.ImagePath','mom_img/sample_img.jpg')
    

    # # The camera does not need a dedicated thread, the callbacks have their owns.
    cam = Camera(camera_id)
    network = TrackingNetwork(network_model)
    network.setCamera(cam)

    siamese_network = SiameseNetwork(siamese_model, mom_path)


    follower = Follow_Person(network, siamese_network, cam)

    display_imgs = True

    while True:

        
        img = follower.follow()
        
        if display_imgs:
            cv2.imshow("RGB", img)
            if cv2.waitKey(1) == 27:
                break
    if display_imgs:
        cv2.destroyAllWindows()