from pynput import keyboard
from tello import Tello
import time
import numpy as np
import cv2

vx = 0
vy = 0
vz = 0
rot = 0
vel = 30
tello = Tello('', 9005)

def on_press(key):
    global vx,vy,vz,rot, vel, tello
    #print(key)
    try:
        if key.char == 'w':
            vz = vel
        elif key.char == 's':
            vz = -vel
        elif key.char == 'a':
            rot = -vel
        elif key.char == 'd':
            rot  = vel
        elif key.char == 'p':
            print("p")
    except AttributeError:
        if key == keyboard.Key.up:
            vx = vel
        elif key == keyboard.Key.down:
            vx = -vel
        elif key == keyboard.Key.right:
            vy = vel
        elif key == keyboard.Key.left:
            vy = -vel

    tello.set_velocities(vy,vx,vz,rot)
    
    

def on_release(key):
    global vx,vy,vz,rot, vel, tello
    #print(key)
    try:
        if key.char == 'w':
            vz = 0
        elif key.char == 's':
            vz = 0
        elif key.char == 'a':
            rot = 0
        elif key.char == 'd':
            rot  = 0
    except AttributeError:
        if key == keyboard.Key.up:
            vx = 0
        elif key == keyboard.Key.down:
            vx = 0
        elif key == keyboard.Key.right:
            vy = 0
        elif key == keyboard.Key.left:
            vy = 0

        elif key == keyboard.Key.esc:
            tello.land()
            tello.close() 
            return False

    tello.set_velocities(vy,vx,vz,rot)
    

tello.takeoff()
time.sleep(6)

# Collect events until released
listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release)
listener.daemon = True
listener.start()


while(True):
    # Capture frame-by-frame
    #ret, frame = cap.read()
    frame = tello.get_image()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
#cap.release()
listener.stop()
tello.close()
cv2.destroyAllWindows()