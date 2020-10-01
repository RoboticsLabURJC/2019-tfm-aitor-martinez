#
# Created on Jan, 2019
#
# @author: naxvm
#
# Class which abstracts a RGBD Camera from ROS messages,
# and provides the methods to keep it constantly refreshed.

import traceback
import threading
import numpy as np
import rospy
from sensor_msgs.msg import Image
import cv_bridge
import cv2

from PIL import Image



class NotAdvertisedException(Exception):
    ''' Raised when the RGB/depth topics have not been already advertised. '''
    pass

class ROSCam:
    IMAGE_HEIGHT = 480
    IMAGE_WIDTH = 640

    def __init__ (self, topics):
        ''' Camera class gets images from live video and transform them
        in order to predict the digit in the image.

        A control thread is not necessary (the subscribers are controlled
        by rospy threads).
        '''

        

        # rospy.init_node('ROSCam', anonymous=True)
        # Check the existance of the topics
        topic_names = zip(*rospy.get_published_topics())[0]
        if topics['rgb'] not in topic_names or topics['depth'] not in topic_names:
            raise NotAdvertisedException

        # Two bridges for concurrency issues
        self.rgb_bridge = cv_bridge.CvBridge()
        self.depth_bridge = cv_bridge.CvBridge()

        # Placeholders
        self.rgb_img = np.zeros((self.IMAGE_HEIGHT, self.IMAGE_WIDTH))
        self.depth_img = np.zeros((self.IMAGE_HEIGHT, self.IMAGE_WIDTH))

        # Subscribers
        self.rgb_lst = rospy.Subscriber(topics['rgb'], Image, self.__rgbCallback, queue_size=1)
        self.d_lst = rospy.Subscriber(topics['depth'], Image, self.__depthCallback, queue_size=1)

        self.lock_rgb = threading.Lock()
        self.lock_depth = threading.Lock()
        # rospy.spin()


    def __rgbCallback(self, rgb_img):
        self.lock_rgb.acquire()
        self.rgb_img = self.rgb_bridge.imgmsg_to_cv2(rgb_img, rgb_img.encoding)
        rospy.logdebug("RGB updated")
        self.lock_rgb.release()

    def __depthCallback(self, depth_img):
        self.lock_depth.acquire()
        self.depth_img = self.depth_bridge.imgmsg_to_cv2(depth_img, depth_img.encoding)
        rospy.logdebug("Depth updated")
        self.lock_depth.release()

    def get_pil_image(self):
        self.lock_rgb.acquire()
        img = np.copy(self.rgb_img)
        self.lock_rgb.release()
        return Image.fromarray(img)
    
    def get_rgb_image(self):
        self.lock_rgb.acquire()
        img = np.copy(self.rgb_img)
        self.lock_rgb.release()
        return img

    def get_depth_image(self):
        self.lock_depth.acquire()
        img = np.copy(self.depth_img)
        self.lock_depth.release()
        return img
        
