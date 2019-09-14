
from tello import Tello
import time
import numpy as np
import cv2

MIN  = np.array([0,0, 60],np.uint8)
MAX  = np.array([180, 255, 255],np.uint8)


def get_object(image, lower=MIN, upper=MAX):
        # convert to the HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # construct a mask for the color specified
    # then perform a series of dilations and erosions
    # to remove any small blobs left in the mask
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # find contours in the mask and
    # initialize the current center
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None
    area = None

    # only proceed if at least one contour was found
    if len(cnts) > 0:
    # find the largest contour in the mask, then use
    # it to compute the minimum enclosing circle and
    # centroid
        c = max(cnts, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        area=cv2.contourArea(c)
        center = (int((x+w+x)/2),int((y+h+y)/2))
        
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),1)

    return center, area, image



def main():
    vx = 0
    vy = 0
    vz = 0
    rot = 0
    vel = 30
    RED_MIN  = np.array([0,0, 90],np.uint8)
    RED_MAX  = np.array([50, 255, 150],np.uint8)

    tello = Tello('', 9005)
    time.sleep(2)
    print(tello.get_battery())
    #tello.takeoff()
    
    while(True):
        frame = tello.get_image()
        #print("height = {}, width = {}".format(frame.shape[0], frame.shape[1]))
        #720x960
        center, area, img = get_object(frame, RED_MIN, RED_MAX)
        print(tello.get_battery())

        

        # Display the resulting frame
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow('frame',img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release all
    tello.close()
    cv2.destroyAllWindows()
  
if __name__== "__main__":
  main()



