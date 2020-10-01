from math import pi as PI
import math

V_MAX = 1.5 # m/s
W_MAX = PI/2 # rad/s

class Drone:
    def __init__(self, local_ip, local_port, command_timeout=.2, tello_ip='192.168.10.1', tello_port=8889):
        self._loca_ip = local_ip

    def to_rc(self, v):
        return math.floor((100 * v) / V_MAX)

    def to_rc_rot(self, w):
        return math.floor((100 * w) / W_MAX)

    def _rc(self, vy, vx, vz, rot):
        print("Enviando comandos de velocidad: {} cm/s [vx], {} cm/s [vy], {} cm/s [vz], {} grad/s [az]".format(vx, vy, vz, rot))

    def movimiento_libre(self, vx, vy, vz, vw):
        self._rc(self.to_rc(vy), self.to_rc(vx), self.to_rc(vz), self.to_rc_rot(vw))

    def aterrizar(self):
        return

    def close(self):
        return

    def despegar(self):
        return
    
    def bateria_restante(self):
        return 10
        
        

    