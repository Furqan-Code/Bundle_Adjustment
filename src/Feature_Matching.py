import numpy as np
import cv2
from matplotlib import pyplot as plt
#from scipy.linalg import svd


def initial_guess(img1,img2,dist):

    '''
    Description: 
    Accepts 2 images and returns the following:
    1. SIFT pixel correspondences between left and right image - every 10th match is returned to reduce the number of generated points
    2. 3d Point coordinates for corresponding features. The coordinates are wih respect to img1 reference frame
    3. Initial guess for the transformation matrix between the 2 images. The matrix is for img2 w.r.t to img1 

    Params: 
    -------
    img1: 2xn image coordinates
    img2: 2xn image coordinates
    dist: Distance covered between 2 images

    Returns:
    --------
    src_pts: Nx2 feature matches in image 1
    dst_pts: Nx2 feature matches in image 2
          P: Nx3 3D point coordinates in img1 reference frame
    R_t_mat: 4x4 rotation matrix between img1 and img2
    ''' 

    #Camera Calibration Matrix
    K =np.array([[904.04572636,0,645.74398382],[0,907.01811462,512.14951996],[0,0,1]])

    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    #Find essential matrix
    E,mask = cv2.findEssentialMat(src_pts,dst_pts,cameraMatrix= K)
    mask=mask.any(axis=1)

    #Filter out outlier points as determined from findEssentialMat via RANSAC  
    src_pts = src_pts[mask]
    dst_pts = dst_pts[mask]
    
    #Convert pixel coordinates to camera coordinates (centered at camera projection centre). This is needed for triangulation function below.
    inv_K = np.linalg.inv(K)
    src_img_pts = src_pts*np.array([inv_K[0,0],inv_K[1,1]])+np.array([inv_K[0,2],inv_K[1,2]])
    dst_img_pts = dst_pts*np.array([inv_K[0,0],inv_K[1,1]])+np.array([inv_K[0,2],inv_K[1,2]])   

    
    P, Q = [], []

    #Retrieve pose from essential matrix - the function automatically performs tyhe chirality check and returns R and t that satisfy projection constraints. 
    retval,R_cv,t_cv1,mask = cv2.recoverPose(E,src_pts,dst_pts,cameraMatrix= K)
    #t_cv1 is a unit vector and this needs to be scaled by the distance between the 2 points. 
    t_cv = t_cv1*dist

    #check determinat of rotatoin matrix
    det = np.linalg.det(R_cv)
    if det<0:
        R_cv = -R_cv
        print('switiching rotation mat')

    #Retrieve 3D points in image 1 reference frame via triangulation
    P_cv, Q_cv = triangulate(src_img_pts, dst_img_pts, t_cv.reshape(-1), R_cv)
    P1=P_cv.T
    Q=Q_cv.T

    src_pts1 = src_pts.reshape(-1,2)
    dst_pts1 = dst_pts.reshape(-1,2)

    step_size = 15
    src_pts = src_pts1[0::step_size,:]
    dst_pts = dst_pts1[0::step_size,:]
    P = P1[0::step_size,:]
        
    R_t_mat = np.vstack((np.hstack((R_cv,-R_cv@t_cv.reshape(3,1))),np.array([[0,0,0,1]])))

    return src_pts,dst_pts,P,R_t_mat

def triangulate(p, q, t, R):
    '''
    Triangulate 3d point from point correpondences across image pairs.

    The logic used here is borrowed from Carlo Tomasi's work obatined from:
    https://www2.cs.duke.edu/courses/spring19/compsci527/notes/longuet-higgins.pdf

    '''

    p=(p.reshape(-1,2)).T
    q=(q.reshape(-1,2)).T
    n = p.shape[1]
    assert n == q.shape[1], 'p and q must have the same number of columns'
    P = np.zeros((3, n))
    
    i, j, k = R[0], R[1], R[2]
    kt = np.dot(k, t)
    proj = np.vstack((i, j))
    projt = np.dot(proj, t)
    
    C = np.array(((1, 0, 0), (0, 1, 0), (0, 0, 0), (0, 0, 0))).astype(float)
    c = np.zeros(4)
    
    for m in range(n):
        C[:2, 2] = -p[:2, m]
        C[2:, :] = np.outer(q[:2, m], k) - proj
        c[2:] = kt * q[:2, m] - projt
        
        x = np.linalg.lstsq(C, c, rcond=None)
        P[:, m] = x[0]
        
    Q = np.dot(R, P - np.outer(t, np.ones((1, n))))
    return P, Q

    