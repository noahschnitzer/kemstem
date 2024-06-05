import numpy as np
def gaussian_2d(yx, amp, xo, yo, sigma_x, sigma_y, theta, offset):
    y,x = yx
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amp*np.exp(- (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return g.flatten()

def sin2D( yx, amp, qr, angle,phi):
    (y,x)= yx

    qx = qr*np.cos(angle)
    qy = qr*np.sin(angle)
    return amp*np.sin(qx*x + qy*y + phi).ravel()
