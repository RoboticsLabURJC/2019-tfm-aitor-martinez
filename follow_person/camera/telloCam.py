import cv2
from PIL import Image

class TelloCam:
    IMAGE_HEIGHT = 480
    IMAGE_WIDTH = 640

    def __init__ (self, device):
        self._cap = device

    def get_rgb_image(self):
        frame = self._cap.get_image()
        dim = (self.IMAGE_WIDTH, self.IMAGE_HEIGHT)
        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_CUBIC)
        return frame

    def get_pil_image(self):
        return Image.fromarray(self.get_rgb_image())

    
    def __del__(self):
        self._cap.release()