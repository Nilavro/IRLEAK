import numpy as np
import math
import cv2
from os import listdir

theta_l = 65 * math.pi / 180
f = 1 / math.tan(theta_l / 2)
f_pix = 360 * f # the 360 is pixels, half img width, not degrees
s = f_pix

theta_r = lambda: ((513 - 1) / NUM_PICTURES) * 2 * math.pi / 513
delta_col = lambda: 2*f_pix*math.sin(theta_r()/2)

x2xp = lambda x: s * math.atan(x / f_pix)
xp2x = lambda xp: f_pix * math.tan(xp / s)
y2yp = lambda x, y: s * y / math.sqrt((x**2) + (f_pix**2))
yp2y = lambda xp, yp: f_pix * yp * (1/math.cos(xp / s)) / s

mat = np.array([[  6.58800465e+01,   1.09266759e+01,   1.31393727e+02],
       [ -4.88417145e+01,  -3.25748418e+01,   7.08892286e+02],
       [ -7.33714277e-02,   3.45043730e-02,   1.00000000e+00]])

def parse_ir(temps):
    out = list()
    vals = [float(t) for t in temps]
    for i in range(0, 64, 4):
        out.append(vals[i:i+4])
    out = np.array(out)
    #out = np.fliplr(out)
    out = np.flipud(out)
    return out


def ir_image(im, max_temp, min_temp):
    therm_im = np.zeros(list(im.shape)+[3],dtype=np.uint8)
    mid_temp = (max_temp + min_temp)/2
    rng_temp = max_temp - min_temp
    #[H, S, V]
    therm_im[:,:,0] = (170 * (max_temp - im) / rng_temp) 
    therm_im[:,:,1] = 255
    therm_im[:,:,2] = 127
    return cv2.cvtColor(therm_im, cv2.COLOR_HSV2BGR)


def gs_image(im, max_temp, min_temp):
    rng_temp = max_temp-min_temp
    therm_im = np.zeros(im.shape, dtype=np.uint8)
    therm_im = 255*(im-min_temp)//rng_temp
    return therm_im

def reconvertTemp(thPan, max_temp, min_temp):
    return ((max_temp-min_temp) * thPan / 255) + min_temp
