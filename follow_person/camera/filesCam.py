import cv2
import numpy as np
import glob

from PIL import Image

class FilesCam:
    IMAGE_HEIGHT = 480
    IMAGE_WIDTH = 640

    def __init__ (self, files):
        self._files = glob.glob(files)
        self._index = 0

    def get_rgb_image(self):
        filename = self._files[self._index]
        self._index = (self._index + 1) % len(self._files)
        frame = cv2.imread(filename)
        # dim = (self.IMAGE_WIDTH, self.IMAGE_HEIGHT)
        # frame = cv2.resize(frame, dim, interpolation = cv2.INTER_CUBIC)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def get_pil_image(self):
        return Image.fromarray(self.get_rgb_image())