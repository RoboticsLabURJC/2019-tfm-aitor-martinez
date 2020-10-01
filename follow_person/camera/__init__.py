from .camera import Camera
from .telloCam import TelloCam
from .filesCam import FilesCam

CLASSES = {
    'tello': TelloCam,
    'webcam': Camera,
    'images': FilesCam,
    'video': Camera
    }

def getCamera(device, device_type):
    return CLASSES[device_type.lower()](device)