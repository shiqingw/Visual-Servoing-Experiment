import numpy as np
from scipy.spatial.transform import Rotation as R
import pypose as pp
import torch

def get_homogeneous_transformation(p, R):
    H = np.eye(4)
    H[:3, :3] = R
    H[:3, 3] = p

    return H

def one_point_image_jacobian(coord_in_cam, fx, fy):
    X = coord_in_cam[0]
    Y = coord_in_cam[1]
    Z = coord_in_cam[2]
    J1 = np.array([-fx/Z, 0, fx*X/Z**2, fx*X*Y/Z**2, fx*(-1-X**2/Z**2), fx*Y/Z], dtype=np.float32)
    J2 = np.array([0, -fy/Z, fy*Y/Z**2, fy*(1+Y**2/Z**2), -fy*X*Y/Z**2, -fy*X/Z], dtype=np.float32)

    return np.vstack((J1, J2))

def one_point_depth_jacobian(coord_in_cam, fx, fy):
    X = coord_in_cam[0]
    Y = coord_in_cam[1]
    Z = coord_in_cam[2]
    J = np.array([0,0,-1,-Y,X,0], dtype=np.float32)
    return J

def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]], dtype=np.float32)

def skew_to_vector(S):
    return np.array([S[2,1], S[0,2], S[1,0]], dtype=np.float32)

def point_in_image(x, y, width, height):
    if (0 <= x and x < width):
        if (0 <= y and y < height):
            return True
    return False

def one_point_image_jacobian_normalized(x, y, Z):
    """
    x, y: normalized pixel coordinates (x = (x-cx)/fx, y = (y-cy)/fy)
    Z: depth of the point in camera frame
    """
    J1 = np.array([-1/Z, 0, x/Z, x*y, -(1+x**2), y], dtype=np.float32)
    J2 = np.array([0, -1/Z, y/Z, 1+y**2, -x*y, -x], dtype=np.float32)

    return np.vstack((J1, J2))

def one_point_depth_jacobian_normalized(x, y, Z):
    """
    x, y: normalized pixel coordinates (x = (x-cx)/fx, y = (y-cy)/fy)
    Z: depth of the point in camera frame
    """
    J = np.array([0, 0, -1, -y*Z, x*Z, 0], dtype=np.float32)
    return J

def normalize_one_image_point(x, y, fx, fy, cx, cy):
    x_norm = (x - cx) / fx
    y_norm = (y - cy) / fy
    return x_norm, y_norm

def normalize_corners(corners, fx, fy, cx, cy):
    """
    corners: 2D array of shape (N, 2)
    """
    corners_norm = np.zeros_like(corners)
    corners_norm[:, 0] = (corners[:, 0] - cx) / fx
    corners_norm[:, 1] = (corners[:, 1] - cy) / fy

    return corners_norm

def get_apriltag_corners_cam_and_world_homo_coord(half_apriltag_size, apriltag_pos, apriltag_ori, R_cam_to_world, P_cam_to_world):
    corners_in_apriltag = np.array([[-1,1,0],[1,1,0],[1,-1,0],[-1,-1,0]], dtype=np.float32)*half_apriltag_size
    corners_in_apriltag = np.concatenate([corners_in_apriltag, np.ones([4,1], dtype=np.float32)], axis=1)
    H_apriltag_to_cam = get_homogeneous_transformation(apriltag_pos, R.from_quat(apriltag_ori).as_matrix())
    _H = np.hstack((R_cam_to_world, np.reshape(P_cam_to_world,(3,1))))
    H_cam_to_world = np.vstack((_H, np.array([[0.0, 0.0, 0.0, 1.0]])))
    coord_in_cam = corners_in_apriltag @ H_apriltag_to_cam.T
    coord_in_world = coord_in_cam @ H_cam_to_world.T
    H_apriltag_to_world = H_apriltag_to_cam @ H_cam_to_world
    apriltag_SE3_in_world = pp.mat2SE3(torch.tensor(H_apriltag_to_world))

    return coord_in_cam, coord_in_world, apriltag_SE3_in_world

def dq_to_speeds_in_cam(dq_executed, J_camera, R_cam_to_world):
    speeds_in_world = J_camera @ dq_executed
    v_in_world = speeds_in_world[0:3]
    omega_in_world = speeds_in_world[3:6]
    R_world_to_cam = R_cam_to_world.T
    v_in_cam = R_world_to_cam @ v_in_world
    S_in_cam = R_world_to_cam @ skew(omega_in_world) @ R_world_to_cam.T
    omega_in_cam = skew_to_vector(S_in_cam)
    speeds_in_cam = np.hstack((v_in_cam, omega_in_cam))
    return speeds_in_cam

def compute_SE3_mean(SE3_measurements):
    """
    SE3_samples: np.array of size Nx7
    SE3 vector [tx, ty, tz, qx, qy, qz, qw]
    """
    SE3_measurements = pp.SE3(SE3_measurements)
    # Compute mean of SE3 measurements
    first_measurement = SE3_measurements[0,:] 
    deltas = np.zeros_like(SE3_measurements)
    for i in range(len(deltas)):
        deltas[i,:] = first_measurement.Inv() @ SE3_measurements[i,:]
    deltas = pp.SE3(deltas)
    deltas_in_log = pp.Log(deltas)
    mean_deltas_in_log = pp.se3(deltas_in_log.mean(dim=0))
    mean_delta = pp.Exp(mean_deltas_in_log)
    SE3_optimal = first_measurement @ mean_delta
    return SE3_optimal