import numpy as np 
import cv2


def rover_to_camera():
    #trans_x [m], trans_y [m], trans_z [m], quat_x, quat_y, quat_z, quat_w
    Tmr = np.array([[0.305,-0.003,0.604,-0.579,0.584,-0.407,0.398]])

    angle = 2*np.arccos(Tmr[0,6])
    x = Tmr[0,3]/(1-Tmr[0,6]**2)**0.5
    y = Tmr[0,4]/(1-Tmr[0,6]**2)**0.5
    z = Tmr[0,5]/(1-Tmr[0,6]**2)**0.5
    rod_angles = np.array([[x/angle,y/angle,z/angle]])
    rodrigues_rot = cv2.Rodrigues(rod_angles)[0]
    return np.vstack((np.hstack((rodrigues_rot,-rodrigues_rot@Tmr[0,0:3].reshape(3,1))),np.array([[0,0,0,1]])))
