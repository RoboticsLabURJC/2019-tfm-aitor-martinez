
import cv2
import numpy as np
from datetime import datetime
from datetime import timedelta
import time

from tello.tello_wrapper import Drone



if __name__ == '__main__':

    drone = Drone('', 9005)
    time.sleep(10)
    imagen_base = drone.dame_imagen()
    print(imagen_base.shape)
    height, width, layers = imagen_base.shape
    size = (width,height)
    out = cv2.VideoWriter('project.mp4',cv2.VideoWriter_fourcc(*"mp4v"), 10, size)
    mins = timedelta(minutes=1)
    
    start_time = datetime.now()
    while((datetime.now() - start_time) < mins):
        imagen = drone.dame_imagen()
        img = cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR)
        time.sleep(0.1)
        out.write(img)
        cv2.imshow("RGB", img)
        if cv2.waitKey(1) == 27:
            break
    out.release()
    drone.close()
