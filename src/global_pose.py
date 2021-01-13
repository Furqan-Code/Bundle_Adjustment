import numpy as np
from quaternion_to_rodriquez import rover_to_camera

def camera_position_ground_truth(input_dir):
    '''
    Read global-pose-utm.txt file and return list of time and camera position in camera reference frame
    '''

    with open(input_dir+'/run4_base_hr/global-pose-utm.txt','r') as file:
        pose_file= file.readlines()

    del pose_file[0]

    pose_list=[]
    for line in pose_file:
        line = line.split(',')[0:4]
        line[0] = float(line[0])-1.536099278442122698e+09
        line[1] = float(line[1])
        line[2] = float(line[2])
        line[3] = float(line[3])
        
        pose_list.append(line)

    pose_global = np.asarray(pose_list)
    pose = pose_global[:,1:4].reshape(-1,3)
    position_local = np.hstack((pose_global[:,0].reshape(-1,1),((rover_to_camera()@np.vstack((pose.T,np.ones((1,pose.shape[0])))))[:3,:].reshape(3,-1).T)))
    return position_local

def get_groundtruth_pose(frame_time,positions):
    '''
    Returns the camera position for a given time from a list of time stamped positions in 'positions'
    ''' 

    time_index = (positions[:,0]<=frame_time).nonzero()[0][-1]
    return positions[time_index,1:]