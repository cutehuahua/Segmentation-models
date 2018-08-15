import cv2
import numpy as np

def undistort(img):
    dist = np.array([[-0.3056, 0.1266, 1e-3*-0.4199, 1e-3*-0.1760 ,-0.0274]])
    mat = np.eye(3)
    mat[0,0] = 279.2681 
    mat[1,1] = 278.7655
    mat[1,0] = 0.0869
    mat[0,2] = 311.0627
    mat[1,2] = 231.7194
    dis = cv2.undistort(img, mat, dist)
    return dis
