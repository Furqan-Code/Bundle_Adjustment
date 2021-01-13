import numpy as np
import cv2
from Feature_Matching import initial_guess
from velocity import velocity

def distance(velocity_estimate,frame_0_time,time_inc,base_frame,curr_frame):
    '''
    function to calculate distance travelled between frames

    Parameters:
    -----------
    velocity_estimate = Nx2 array of time,velocity
    frame_0_time = time at frame 0 
    curr_frame   = current frame number

    Returns:
    --------
    Distance travelled

    '''
    
    base_time = frame_0_time +time_inc*base_frame
    curr_time = frame_0_time +time_inc*curr_frame
    total_time = curr_time - base_time
    base_index = (velocity_estimate[:,0]<=base_time).nonzero()[0][-1]
    curr_index = (velocity_estimate[:,0]<=curr_time).nonzero()[0][-1]
    avg_velocity = velocity_estimate[base_index:curr_index+1,1].mean()
    return avg_velocity*total_time




def observations(frame_0_time,nth_frame,input_dir):  

    '''
    Retrieves:
    1. Feature correspondences across camera frames
    2. Initial guess for corresponding 3D points.
    3. Initial guess for camera transformation matrices

    All 3D points and transformations are defined w.r.t the first frame  


    Ouptut:
    -------
    point_list = Nx4 numpy array containing the observation points in the format - <Camera Frame><Feature index><x,y feature coordinates in current Camera Frame>
    model_params = 1x4 list containing <total number of camera frames>,<total number of feature points>,<total number of observations> 
    cam_transforms = Nx6 numpy array of camera extrensic parameters
    pt_3d_list = Nx3 numpy array of 3d points for matched features. The 3D points are defined w.r.t the first frame
    '''

    # list of run 4 image files
    images_root = [
                   input_dir+'/run4_base_hr/mono_image/frame000050_2018_09_04_18_14_44_143406.png',\
                input_dir+'/run4_base_hr/mono_image/frame000051_2018_09_04_18_14_44_342875.png',\
                input_dir+'/run4_base_hr/mono_image/frame000052_2018_09_04_18_14_44_543322.png',\
                input_dir+'/run4_base_hr/mono_image/frame000053_2018_09_04_18_14_44_745412.png',\
                input_dir+'/run4_base_hr/mono_image/frame000054_2018_09_04_18_14_44_947120.png',\
                input_dir+'/run4_base_hr/mono_image/frame000055_2018_09_04_18_14_45_145872.png',\
                input_dir+'/run4_base_hr/mono_image/frame000056_2018_09_04_18_14_45_347874.png',\
                input_dir+'/run4_base_hr/mono_image/frame000057_2018_09_04_18_14_45_548328.png',\
                input_dir+'/run4_base_hr/mono_image/frame000058_2018_09_04_18_14_45_745208.png',\
                input_dir+'/run4_base_hr/mono_image/frame000059_2018_09_04_18_14_45_942796.png',\
                input_dir+'/run4_base_hr/mono_image/frame000060_2018_09_04_18_14_46_143434.png',\
                input_dir+'/run4_base_hr/mono_image/frame000061_2018_09_04_18_14_46_343646.png',\
                input_dir+'/run4_base_hr/mono_image/frame000062_2018_09_04_18_14_46_543835.png',\
                input_dir+'/run4_base_hr/mono_image/frame000063_2018_09_04_18_14_46_747026.png',\
                input_dir+'/run4_base_hr/mono_image/frame000064_2018_09_04_18_14_46_949385.png',\
                input_dir+'/run4_base_hr/mono_image/frame000065_2018_09_04_18_14_47_146589.png',\
                input_dir+'/run4_base_hr/mono_image/frame000066_2018_09_04_18_14_47_345159.png',\
                input_dir+'/run4_base_hr/mono_image/frame000067_2018_09_04_18_14_47_544991.png',\
                input_dir+'/run4_base_hr/mono_image/frame000068_2018_09_04_18_14_47_742937.png',\
                input_dir+'/run4_base_hr/mono_image/frame000069_2018_09_04_18_14_47_942956.png',\
                input_dir+'/run4_base_hr/mono_image/frame000070_2018_09_04_18_14_48_143949.png',\
                input_dir+'/run4_base_hr/mono_image/frame000071_2018_09_04_18_14_48_343130.png',\
                input_dir+'/run4_base_hr/mono_image/frame000072_2018_09_04_18_14_48_547100.png',\
                input_dir+'/run4_base_hr/mono_image/frame000073_2018_09_04_18_14_48_748986.png',\
                input_dir+'/run4_base_hr/mono_image/frame000074_2018_09_04_18_14_48_944920.png',\
                input_dir+'/run4_base_hr/mono_image/frame000075_2018_09_04_18_14_49_145128.png',\
                input_dir+'/run4_base_hr/mono_image/frame000076_2018_09_04_18_14_49_343308.png',\
                input_dir+'/run4_base_hr/mono_image/frame000077_2018_09_04_18_14_49_543605.png',\
                input_dir+'/run4_base_hr/mono_image/frame000078_2018_09_04_18_14_49_742891.png',\
                input_dir+'/run4_base_hr/mono_image/frame000079_2018_09_04_18_14_49_945075.png',\
                input_dir+'/run4_base_hr/mono_image/frame000080_2018_09_04_18_14_50_146393.png',\
                input_dir+'/run4_base_hr/mono_image/frame000081_2018_09_04_18_14_50_346777.png',\
                input_dir+'/run4_base_hr/mono_image/frame000082_2018_09_04_18_14_50_546128.png',\
                input_dir+'/run4_base_hr/mono_image/frame000083_2018_09_04_18_14_50_743644.png',\
                input_dir+'/run4_base_hr/mono_image/frame000084_2018_09_04_18_14_50_943493.png',\
                input_dir+'/run4_base_hr/mono_image/frame000085_2018_09_04_18_14_51_144934.png',\
                input_dir+'/run4_base_hr/mono_image/frame000086_2018_09_04_18_14_51_343637.png',\
                input_dir+'/run4_base_hr/mono_image/frame000087_2018_09_04_18_14_51_544055.png',\
                input_dir+'/run4_base_hr/mono_image/frame000088_2018_09_04_18_14_51_743875.png',\
                input_dir+'/run4_base_hr/mono_image/frame000089_2018_09_04_18_14_51_946698.png',\
                input_dir+'/run4_base_hr/mono_image/frame000090_2018_09_04_18_14_52_145009.png',\
                input_dir+'/run4_base_hr/mono_image/frame000091_2018_09_04_18_14_52_345942.png',\
                input_dir+'/run4_base_hr/mono_image/frame000092_2018_09_04_18_14_52_543475.png',\
                input_dir+'/run4_base_hr/mono_image/frame000093_2018_09_04_18_14_52_743029.png',\
                input_dir+'/run4_base_hr/mono_image/frame000094_2018_09_04_18_14_52_942685.png',\
                input_dir+'/run4_base_hr/mono_image/frame000095_2018_09_04_18_14_53_144148.png',\
                input_dir+'/run4_base_hr/mono_image/frame000096_2018_09_04_18_14_53_343962.png',\
                input_dir+'/run4_base_hr/mono_image/frame000097_2018_09_04_18_14_53_547824.png',\
                input_dir+'/run4_base_hr/mono_image/frame000098_2018_09_04_18_14_53_744450.png',\
                input_dir+'/run4_base_hr/mono_image/frame000099_2018_09_04_18_14_53_945602.png',\
                input_dir+'/run4_base_hr/mono_image/frame000100_2018_09_04_18_14_54_145016.png',\
                input_dir+'/run4_base_hr/mono_image/frame000101_2018_09_04_18_14_54_343470.png',\
                input_dir+'/run4_base_hr/mono_image/frame000102_2018_09_04_18_14_54_543084.png',\
                input_dir+'/run4_base_hr/mono_image/frame000103_2018_09_04_18_14_54_743082.png',\
                input_dir+'/run4_base_hr/mono_image/frame000104_2018_09_04_18_14_54_943012.png',\
                input_dir+'/run4_base_hr/mono_image/frame000105_2018_09_04_18_14_55_145069.png',\
                input_dir+'/run4_base_hr/mono_image/frame000106_2018_09_04_18_14_55_343661.png',\
                input_dir+'/run4_base_hr/mono_image/frame000107_2018_09_04_18_14_55_546670.png',\
                input_dir+'/run4_base_hr/mono_image/frame000108_2018_09_04_18_14_55_745882.png',\
                input_dir+'/run4_base_hr/mono_image/frame000109_2018_09_04_18_14_55_945317.png',\
                input_dir+'/run4_base_hr/mono_image/frame000110_2018_09_04_18_14_56_142818.png'
            ]
        
    #Retrieve velocity profiles
    velocity_estimate = np.asarray(velocity(input_dir))
    
    time_inc = 0.2*nth_frame

    #images to be processed
    images = images_root[0::nth_frame]

    #Camera Calibration Matrix
    K =np.array([[904.04572636,0,645.74398382],[0,907.01811462,512.14951996],[0,0,1]])

    Data_assoc=[np.zeros((1,2))]*len(images)
    cam_transforms = [np.identity(4)]*len(images)

    for i in range(len(images)):
        # pont matching is done upto 6 frames forward
        for j in range(i+1,min(i+6,len(images))):
            img1 = cv2.imread(images[i],0)[:500,:]
            img2 = cv2.imread(images[j],0)[:500,:]
            dist = distance(velocity_estimate,frame_0_time,time_inc,i,j)
            src,dst,pt_3d,R_t_mat=initial_guess(img1,img2,dist)

            # transform camera parameters w.r.t to frame 0 
            if j-i==1:
                cam_transforms[j]= cam_transforms[j-1]@R_t_mat

            # transform 3D points w.r.t frame 0    
            pt_3d = (cam_transforms[i]@np.hstack((pt_3d,np.ones((pt_3d.shape[0],1)))).T).T

            
            #Logic for creating list of matching point correspondences across multiple frames 

            point_i_bool=(src[:,None]==Data_assoc[i]).all(-1).any(-1)

            if (~point_i_bool).all() == True:
                if Data_assoc[i].shape[0]==1:
                    Data_assoc[i]=src
                    Data_assoc[j]=dst
                    pt_3d_list = pt_3d
                    for iter in range(len(images)):
                        if iter != i and iter != j:
                            Data_assoc[iter]=np.zeros((src.shape[0],2)) 
            else:
                for row_ind,row in enumerate(point_i_bool):
                    if row.all() == False:
                        Data_assoc[i]=np.vstack((Data_assoc[i],src[row_ind]))
                        if Data_assoc[j].shape[0]<Data_assoc[i].shape[0]:
                            diff = Data_assoc[i].shape[0]-Data_assoc[j].shape[0]-1
                            Data_assoc[j]=np.vstack((Data_assoc[j],np.zeros((diff,2))))
                        Data_assoc[j]=np.vstack((Data_assoc[j],dst[row_ind]))
                        pt_3d_list = np.vstack((pt_3d_list,pt_3d[row_ind]))
                        for iter in range(len(images)):
                            if iter != i and iter != j:
                                Data_assoc[iter]=np.vstack((Data_assoc[iter],np.zeros((1,2))))
                    else:
                        j_ind = (Data_assoc[i]==src[row_ind]).all(axis=1).nonzero()
                        j_ind = j_ind[0][0]
                        if Data_assoc[j].shape[0]<=j_ind:
                            diff = j_ind-Data_assoc[j].shape[0]+1
                            Data_assoc[j]=np.vstack((Data_assoc[j],np.zeros((diff,2))))
                        Data_assoc[j][j_ind]=dst[row_ind]

    # Reorder data for returning to main funciton 
    point_list=[]
    for i in range(Data_assoc[0].shape[0]):
        for j in range(len(Data_assoc)):
            if len(Data_assoc[j][i].nonzero()[0]) != 0:
                point_list.append([j,i,Data_assoc[j][i][0],Data_assoc[j][i][1]])
    point_list = np.asarray(point_list)
    camera_frames = len(Data_assoc)
    num_points = Data_assoc[0].shape[0]
    num_observation = len(point_list)
    model_params = [camera_frames,num_points,num_observation]

    for index,matrix in enumerate(cam_transforms):
        cam_transforms[index] =  (np.vstack((cv2.Rodrigues(matrix[:3,:3])[0],matrix[:3,3].reshape(-1,1)))).T

    cam_transforms = np.asarray(cam_transforms).reshape(-1,6)

    return point_list,model_params,cam_transforms,pt_3d_list

        