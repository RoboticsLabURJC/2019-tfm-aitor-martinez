import cv2
import numpy as np

from PIL import Image

class Camera:
    IMAGE_HEIGHT = 480
    IMAGE_WIDTH = 640

    def __init__ (self, device):
        self._cap = cv2.VideoCapture(device)

    def get_rgb_image(self):
        ret, frame = self._cap.read()
        dim = (self.IMAGE_WIDTH, self.IMAGE_HEIGHT)
        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_CUBIC)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def get_pil_image(self):
        return Image.fromarray(self.get_rgb_image())

    
    def __del__(self):
        self._cap.release()