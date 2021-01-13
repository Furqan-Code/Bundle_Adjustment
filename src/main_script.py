import numpy as np
from observations import observations
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import cv2
from global_pose import camera_position_ground_truth, get_groundtruth_pose
from quaternion_to_rodriquez import rover_to_camera



def read_data(frame_0_time,nth_frame,input_dir):
    '''
    Read points and initial guesses
    '''

    point_list,params,cam_transforms,pt_3d_list = observations(frame_0_time,nth_frame,input_dir)

    camera_indices=point_list[:,0].reshape(-1,1).astype(int)
    point_indices= point_list[:,1].reshape(-1,1).astype(int)
    points_2d = point_list[:,2:].astype('float32')

    camera_params = cam_transforms.astype('float32')
    points_3d = pt_3d_list[:,:3].reshape(-1,3).astype('float32')

    #fx [px], fy [px], cx [px], cy [px], k1, k2, p1, p2, k3
    camera_intrinsics = np.array([[904.04572636,907.01811462,645.74398382,512.14951996,-0.3329137,0.10161043,0.00123166,-0.00096204,0]]).astype('float32')

    return camera_params, points_3d, camera_indices,point_indices, points_2d,camera_intrinsics

def rotate(points, rot_vecs):
    """
    Rodrigues' rotation formula for vector rotation.
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v

def project(points, camera_params,camera_intrinsics):
    """Convert 3-D points to 2-D by projection onto images."""

    # Rotatrion and translation
    a = camera_params[:,:, :3]
    points_proj = rotate(points, camera_params[:,:, :3])
    points_proj += camera_params[:,:, 3:] 
    points_proj = points_proj[:,:,:2] / points_proj[:,:, 2, np.newaxis]
    
    #camera intrinsics
    f = camera_intrinsics[:,0:2]
    k1 = camera_intrinsics[:, 4]
    k2 = camera_intrinsics[:, 5]
    k3 = camera_intrinsics[:, 8]
    c = camera_intrinsics[:,2:4]
    

    #pass through camera intrinsic matric
    points_proj = (points_proj[:,:2]*f)+c
    points_proj = points_proj.reshape(-1,2)
    return points_proj

def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d,camera_intrinsics):
    """Compute residuals."""
    camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
    points_3d = params[n_cameras * 6:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices],camera_intrinsics)
    chc = (points_proj - points_2d).ravel()
    return (points_proj - points_2d).ravel()

def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    '''Generate sparcity indexes for jacobian matrix'''
    m = camera_indices.size * 2
    n = n_cameras * 6 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(6):
        A[2 * i, camera_indices * 6 + s] = 1
        A[2 * i + 1, camera_indices * 6 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 6 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 6 + point_indices * 3 + s] = 1

    A[2*i,0:6]=0
    A[2*i+1,0:6]=0
    

    return A

def get_ground_truth(n_cameras,frame_0_time,nth_frame,input_dir):
    '''Get camera position ground truth w.r.t to initial camera frame'''
    camera_positions_all = camera_position_ground_truth(input_dir)

    ground_truth =[]
    for camera in range(n_cameras):
        frame_time = frame_0_time + 0.2*nth_frame*camera
        if camera ==0:
            frame_position_0 = get_groundtruth_pose(frame_time,camera_positions_all)
            frame_position = np.array([[0,0,0]])
        else:
                
            frame_position_interim = get_groundtruth_pose(frame_time,camera_positions_all) - frame_position_0

            #rearrange global indexes to corresponding camera index (x is alt, y is -easting, z is -northing for camera)
            frame_position = np.vstack((frame_position,np.array([[frame_position_interim[2],-frame_position_interim[0],-frame_position_interim[1]]])))


    return frame_position.ravel()

def run_proj(input_dir,output):
    
    print('Beginning Bundle Adjustment. Can take upto 20-25 min!')
    # Change based on first frame time
    frame_0_time = 5

    # process every nth frame
    nth_frame = 5 

    # Get initial guesses and points
    camera_params, points_3d, camera_indices, point_indices, points_2d,camera_intrinsics = read_data(frame_0_time,nth_frame,input_dir)

    n_cameras = camera_params.shape[0]
    n_points = points_3d.shape[0]

    n = 6 * n_cameras + 3 * n_points
    m = 2 * points_2d.shape[0]

    print("n_cameras: {}".format(n_cameras))
    print("n_points: {}".format(n_points))
    print("Total number of parameters: {}".format(n))
    print("Total number of residuals: {}".format(m))


    #get camera position ground truth
    camera_ground_truth = get_ground_truth(n_cameras,frame_0_time,nth_frame,input_dir)

    # initial reprojections
    x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
    f0 = fun(x0, n_cameras, n_points, camera_indices, point_indices, points_2d,camera_intrinsics)

    plt.figure()
    plt.plot(f0)
    plt.xlabel('feature point index')
    plt.ylabel('reprojection error (no. of pixels)')


    A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)

    # Perform bundle adjustment 
    res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                        args=(n_cameras, n_points, camera_indices, point_indices, points_2d,camera_intrinsics))

    # Process results
    reprojection_error_summary = [res.fun.max(),res.fun.min(),np.sqrt(res.fun**2).mean()]

    print('Max. reprojection error (in pixels): {} '.format(reprojection_error_summary[0]))
    print('Min. reprojection error (in pixels): {} '.format(reprojection_error_summary[1]))
    print('RMS reprojection error (in pixels):  {}'.format(reprojection_error_summary[2]))

    plt.plot(res.fun)
    plt.savefig(output + '/reprojection error.png')

    with open(output + '/Bundle adjustment summary.txt','w') as results:
        results.write('Model summary:\n')
        results.write('number of cameras: ' + str(n_cameras) + ', number of points: ' + str(n_points) + ', total parameters: ' + str(n) + ',total residuals: ' + str(m) +'\n')
        results.write('Max. reprojection error (in pixels): '+str(reprojection_error_summary[0])+ '\n')
        results.write('Min. reprojection error (in pixels): '+str(reprojection_error_summary[1])+ '\n')
        results.write('RMS reprojection error (in pixels): '+str(reprojection_error_summary[2])+ '\n')
        
    # Point cloud plot
    fig = plt.figure()
    ax = fig.add_subplot(111,projection = '3d')
    x = res.x[6*n_cameras::3]
    y = res.x[6*n_cameras+1::3]
    z = res.x[6*n_cameras+2::3]
    ax.scatter3D(x,y,z,c='r',marker='o',s = 2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.xlim(-100,100)
    plt.ylim(-100,100)
    plt.savefig(output + '/point_cloud.png')


    # Camera frames plot ()
    fig=plt.figure()
    ax = fig.add_subplot(111,projection = '3d')
    camera_errors=[]
    camera_errors_rms=[]
    for camera in range(n_cameras):
        Rot = cv2.Rodrigues(res.x[6*camera:6*camera+3].reshape(1,3))[0]
        t = np.linalg.inv(Rot)@res.x[6*camera+3:6+6*camera].reshape(3,1)
        if camera == 0:
            t_0 = t
        #t = t-t_0
        if camera>0:
            position_error = np.abs(camera_ground_truth[3*camera:3+3*camera].reshape(3,-1)-t/t)*100
        else:
            position_error = np.zeros((3,1))
        position_error_rms = np.sqrt((camera_ground_truth[3*camera:3+3*camera].reshape(3,-1)-t)**2).mean()
        camera_errors.append(position_error)
        camera_errors_rms.append(position_error_rms)
        s = np.hstack((t,t,t))
        e =Rot+s
        ax.scatter3D(t[0,:],t[1,:],t[2,:], 'o', c = 'b')
        ax.plot3D([s[0, 0], e[0, 0]], [s[1, 0], e[1, 0]], [s[2, 0], e[2, 0]], 'r')
        ax.plot3D([s[0, 1], e[0, 1]], [s[1, 1], e[1, 1]], [s[2, 1], e[2, 1]], 'g')
        ax.plot3D([s[0, 2], e[0, 2]], [s[1, 2], e[1, 2]], [s[2, 2], e[2, 2]], 'b')
    ax.scatter3D(camera_ground_truth[0::3],camera_ground_truth[1::3],camera_ground_truth[2::3],c='r',marker = 'o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.savefig(output + '/camera_frame.png')


    plt.figure()
    plt.bar(range(n_cameras),camera_errors_rms)
    plt.xlabel('Camera number')
    plt.ylabel('RMS Error')
    plt.savefig(output + '/camera_frame_rms_error')


    camera_errors = np.asarray(camera_errors).reshape(n_cameras,3)

    plt.figure()
    plt.bar([i for i in range(n_cameras)],camera_errors[:,0],label='Altitiude error',alpha=0.5)
    plt.bar(range(n_cameras),camera_errors[:,1],label='Easting error',alpha=0.5)
    plt.bar(range(n_cameras),camera_errors[:,2],label='Northing error',alpha=0.5)
    plt.xlabel('Camera number')
    plt.ylabel('Errors %')
    plt.legend()
    plt.ylim(0,100)
    plt.savefig(output + '/camera_frame_error')



