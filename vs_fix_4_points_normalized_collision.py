import time

# print("==> Initializing Julia")
# time1 = time.time()
# from julia.api import Julia
# jl = Julia(compiled_modules=False)
# time2 = time.time()
# print("==> Initializing Julia took {} seconds".format(time2-time1))

import argparse
import json
import signal
import sys
import threading
from copy import deepcopy
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import tf

import apriltag
import cvxpy as cp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
import proxsuite
import rospy
import torch
import pypose as pp
from cv_bridge import CvBridge
from cvxpylayers.torch import CvxpyLayer
from PIL import Image
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image
from RobotModel_collision import RobotModel
from FR3Py.interfaces import FR3Real
from datetime import datetime
import os
import shutil
import pickle
import cv2
from all_utils.vs_utils import dq_to_speeds_in_cam, skew, skew_to_vector, compute_SE3_mean, change_quat_format
from all_utils.vs_utils import one_point_image_jacobian_normalized, one_point_depth_jacobian_normalized
from all_utils.vs_utils import normalize_one_image_point, normalize_corners, get_apriltag_corners_cam_and_world_homo_coord
from all_utils.proxsuite_utils import init_prosuite_qp
from all_utils.cvxpylayers_utils import init_cvxpylayer
from all_utils.joint_velocity_control_utils import bring_to_nominal_q
from ekf.ekf_ibvs_normalized import EKF_IBVS

try:
    from differentiable_collision_utils.dc_cbf import DifferentiableCollisionCBF
except:
    from differentiable_collision_utils.dc_cbf import DifferentiableCollisionCBF

def signal_handler(signal, frame):
    global target_thread_stop_global, image_thread_stop_global
    print("==> Ctrl+C received. Terminating threads...")
    target_thread_stop_global = True
    apriltag_thread.join()
    image_thread_stop_global = True
    image_thread.join()
    print("==> Threads terminated...")
    sys.exit()

def tf_listener_target_thread_func():

    global target_pos_global, target_ori_global
    global target_latest_timestamp_gloabl
    global target_thread_stop_global
    global camera_frame_global

    listener_target = tf.TransformListener()
    listener_target.waitForTransform(camera_frame_global, "/target", rospy.Time(), rospy.Duration(4.0))

    # Main loop to continuously update the pose
    while (not rospy.is_shutdown()) and (target_thread_stop_global == False):
        try:
            (trans, quat) = listener_target.lookupTransform(camera_frame_global, "/target", rospy.Time(0))
            target_pos_global = np.array(trans)
            target_ori_global = np.array(quat)
            target_latest_timestamp_gloabl = listener_target.getLatestCommonTime(camera_frame_global, "/target")

            rospy.sleep(0.001)  # Adjust the sleep duration as needed
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn("Error occurred while retrieving TF transform.") 

def tf_listener_obstacle_thread_func():

    global obstacle_pos_global, obstacle_ori_global
    global obstacle_thread_stop_global
    global camera_frame_global

    listener_obstacle = tf.TransformListener()
    listener_obstacle.waitForTransform(camera_frame_global, "/obstacle", rospy.Time(), rospy.Duration(4.0))

    # Main loop to continuously update the pose
    while (not rospy.is_shutdown()) and (obstacle_thread_stop_global == False):
        try:

            (trans, quat) = listener_obstacle.lookupTransform(camera_frame_global, "/obstacle", rospy.Time(0))
            obstacle_pos_global = np.array(trans)
            obstacle_ori_global = np.array(quat)

            rospy.sleep(0.001)  # Adjust the sleep duration as needed
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn("Error occurred while retrieving TF transform.") 

def image_thread_func():
    def image_callback(data):
        global detector_global, target_corners_global, cv_bridge_gloabl, gray_image_global
        try:
            # Process the image and perform AprilTag detection
            cv_image = cv_bridge_gloabl.imgmsg_to_cv2(data, desired_encoding="bgr8")

            # Convert the image to grayscale (required by the AprilTag detector)
            gray_image_global = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # Detect AprilTags in the image
            results = detector_global.detect(gray_image_global)

            for ii in range(len(results)):
                res = results[ii]
                if res.tag_id == 0:
                    target_corners_global = res.corners       

        except Exception as e:
            rospy.logerr("Error processing the image: %s", str(e))

    global image_thread_stop_global
    # Initialize the ROS subscriber
    rospy.Subscriber('/camera/infra1/image_rect_raw', Image, image_callback)
    # while (not rospy.is_shutdown()) and (image_thread_stop_global == False):
    rospy.spin()
    return 


if __name__ == '__main__':
    # Register the signal handler for SIGINT (Ctrl+C)
    signal.signal(signal.SIGINT, signal_handler)

    # Choose test settings
    parser = argparse.ArgumentParser(description="Visual servoing")
    parser.add_argument('--exp_num', default=21, type=int, help="test case number")

    # Set random seed
    seed_num = 0
    np.random.seed(seed_num)

    # Load test settings and create result_dir
    args = parser.parse_args()
    exp_num = args.exp_num
    now = datetime.now()
    results_dir = "{}/results/exp_{:03d}_{}".format(str(Path(__file__).parent), exp_num, now.strftime("%Y-%m-%d-%H-%M-%S"))
    test_settings_path = "{}/test_settings/test_settings_{:03d}.json".format(str(Path(__file__).parent), exp_num)

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    shutil.copy(test_settings_path, results_dir)

    with open(test_settings_path, "r", encoding="utf8") as f:
        test_settings = json.load(f)

    # Robot model
    print("==> Loading robot model...")
    pin_robot = RobotModel()
    
    # Real robot interface
    print("==> Loading real robot interface...")
    robot = FR3Real()
    # robot = FR3Real(interface_type="joint_torque")
        
    # Various configs
    camera_config = test_settings["camera_config"]
    controller_config = test_settings["controller_config"]
    observer_config = test_settings["observer_config"]
    CBF_config = test_settings["CBF_config"]
    collision_cbf_config = test_settings["collision_cbf_config"]
    optimization_config = test_settings["optimization_config"]
    obstacle_config = test_settings["obstacle_config"]
    target_config = test_settings["target_config"]
    ekf_config = test_settings["ekf_config"]

    # Joint limits
    joint_limits_config = test_settings["joint_limits_config"]
    joint_lb = np.array(joint_limits_config["lb"], dtype=np.float32)
    joint_ub = np.array(joint_limits_config["ub"], dtype=np.float32)

    # Camera parameters
    camera_frame_global = camera_config["camera_frame"]
    intrinsic_matrix = np.array(camera_config["intrinsic_matrix"], dtype=np.float32)
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    depth_scale = camera_config["depth_scale"]

    # Differentiable optimization layer
    nv = 2
    nc_target = optimization_config["n_cons_target"]
    nc_obstacle = optimization_config["n_cons_obstacle"]
    kappa = optimization_config["exp_coef"]
    cvxpylayer = init_cvxpylayer(nv, nc_target, nc_obstacle, kappa)

    # Proxsuite for CBF-QP
    n, n_eq, n_in = 6, 0, 1
    cbf_qp = init_prosuite_qp(n, n_eq, n_in)

    # Proxsuite for inverse kinematics with joint limits
    n, n_eq, n_in = 9, 0, 9
    joint_limits_qp = init_prosuite_qp(n, n_eq, n_in)

    # Move the robot to the nominal configuration
    q_nominal = np.array(test_settings["q_nominal"], dtype=np.float32)
    bring_to_nominal_q(robot, q_nominal, joint_lb, joint_ub)

    # Starting ros node
    print("==> Launch ros node...")
    rospy.init_node('apriltag_detection_node', anonymous=True)

    # Create and start the obstacle thread
    print("==> Starting obstacle thread...")
    obstacle_pos_global, obstacle_ori_global = [], []
    obstacle_thread_stop_global = False
    obstacle_thread = threading.Thread(target=tf_listener_obstacle_thread_func)
    obstacle_thread.daemon = True
    obstacle_thread.start()
    print("==> Turn camera toward the obstacle")
    input("==> Press Enter to if obstacle is in sight")
    while len(obstacle_pos_global) == 0:
        print("==> Wait a little bit for the obstacle thread...")
        time.sleep(1)

    # Capture obstacle world coordinates
    apriltag_size = obstacle_config["apriltag_size"]
    offset = target_config["offset"]
    num_sample = 20
    obstacle_corners_in_world_samples = np.zeros((num_sample, 4, 4), dtype=np.float32)
    obstacle_SE3_in_world_samples = np.zeros((num_sample, 7), dtype=np.float32)
    print("==> Collecting multiple samples of the obstacle...")
    for i in range(num_sample):
        # Collect several samples and use the mean
        obstacle_pos = deepcopy(obstacle_pos_global)
        obstacle_ori = deepcopy(obstacle_ori_global)
        state = robot.get_state()
        q, dq = state['q'], state['dq']
        info = pin_robot.getInfo(q,dq)
        _, obstacle_corners_in_world, obstacle_SE3_in_world = get_apriltag_corners_cam_and_world_homo_coord(
            apriltag_size/2 + offset, obstacle_pos, obstacle_ori, info["R_CAMERA"], info["P_CAMERA"])
        obstacle_corners_in_world_samples[i,:,:] = obstacle_corners_in_world
        obstacle_SE3_in_world_samples[i,:] = obstacle_SE3_in_world
        time.sleep(0.1)
    print("==> Obstacle world coordinates captured:")
    obstacle_corners_in_world = np.mean(obstacle_corners_in_world_samples, axis=0)
    print(obstacle_corners_in_world)
    obstacle_SE3_in_world = compute_SE3_mean(obstacle_SE3_in_world_samples)
    obstacle_SE3_in_world = np.array(obstacle_SE3_in_world)
    print("==> Obstacle SE3 vector in world:")
    print(obstacle_SE3_in_world)

    # Kill obstacle thread
    obstacle_thread_stop_global = True
    obstacle_thread.join()
    print("==> Obstacle thread terminated")

    # Initialize differentiable collision
    if collision_cbf_config["active"] == 1:
        half_length = obstacle_config["apriltag_size"]/2.0 + target_config["offset"]
        polygon_b_in_body = np.array([half_length, half_length, 0.0025, half_length, half_length, 0.0025])
        obstacle_r = obstacle_SE3_in_world[0:3]
        obstacle_q = change_quat_format(obstacle_SE3_in_world[3:7])
        print("==> Initializing differentiable collision (Julia)")
        time1 = time.time()
        try:
            collision_cbf = DifferentiableCollisionCBF(polygon_b_in_body, obstacle_r, obstacle_q, gamma=5.0, alpha_offset=1.03)
        except:
            collision_cbf = DifferentiableCollisionCBF(polygon_b_in_body, obstacle_r, obstacle_q, gamma=5.0, alpha_offset=1.03)
        vel = collision_cbf.filter_dq(np.zeros(9), info)
        time2 = time.time()
        print("==> Initializing differentiable collision (Julia) took {} seconds".format(time2-time1))

    # Create and start the apriltag thread
    print("==> Creating target thread...")
    target_pos_global, target_ori_global = [], []
    target_latest_timestamp_gloabl = 0
    target_thread_stop_global = False
    apriltag_thread = threading.Thread(target=tf_listener_target_thread_func)
    apriltag_thread.daemon = True
    apriltag_thread.start()
    while len(target_pos_global) == 0:
        print("==> Wait a little bit for the target thread...")
        time.sleep(1)

    # Start the image thread
    print("==> Starting image thread...")
    gray_image_global, target_corners_global = [], []
    cv_bridge_gloabl = CvBridge()
    detector_global = apriltag.Detector()
    image_thread_stop_global = False
    image_thread = threading.Thread(target=image_thread_func)
    image_thread.daemon = True
    image_thread.start()
    while len(target_corners_global) == 0:
        print("==> Wait a little bit for the image thread...")
        time.sleep(1)

    # History
    history = {"time": [],
                "q": [],
                "dq": [],
                "corners_raw_normalized": [],
                "corner_depths_raw": [],
                "obstacle_corner_in_world": [],
                "obstacle_corner_in_image": [],
                "error_position": [],
                "cbf": [],
                "joint_vel_command":[], 
                "info":[],
                "d_hat_dob": [],
                "d_true": [],
                "d_true_z":[],
                "loop_time": [],
                "ekf_estimates": [],
                "dob_dt":[],
                "ekf_dt":[]
                }
    
    # Adjust mean and variance target to num_points
    num_points = 4
    depth_target = controller_config["depth_target"]
    apriltag_size = target_config["apriltag_size"]
    desired_corners_in_cam = np.array([[1, 1],
                                        [-1, 1],
                                        [-1, -1],
                                        [1, -1]], dtype = np.float32)*apriltag_size/2
    desired_corners_in_cam = np.hstack((desired_corners_in_cam, depth_target*np.ones((desired_corners_in_cam.shape[0],1), dtype=np.float32)))
    desired_corners = desired_corners_in_cam @ intrinsic_matrix.T
    desired_corners = (desired_corners/desired_corners[:,2][:,np.newaxis])[:,0:2]
    desired_corners_normalized = normalize_corners(desired_corners, fx, fy, cx, cy)
    mean_target_normalized = np.mean(desired_corners_normalized[0:num_points,:], axis=0)
    variance_target_normalized = np.var(desired_corners_normalized[0:num_points,:], axis=0)
    J_image_cam_desired = np.zeros((2*num_points, 6), dtype=np.float32)
    for ii in range(len(desired_corners_normalized)):
        x, y = desired_corners_normalized[ii,:]
        Z = depth_target
        J_image_cam_desired[2*ii:2*ii+2] = one_point_image_jacobian_normalized(x,y,Z)
    
    # Check which observer to use
    if observer_config["active"] == 1 and ekf_config["active"] == 1:
        raise ValueError("Two observers cannot be active for controller at the same time.")
    if observer_config["active_for_cbf"] == 1 and ekf_config["active_for_cbf"] == 1:
        raise ValueError("Two observers cannot be active for cbf at the same time.")

    # Disturbance observer initialization
    last_dob_time = time.time()
    corners_raw = deepcopy(target_corners_global)
    corners_raw_normalized = normalize_corners(corners_raw, fx, fy, cx, cy)
    observer_gain = np.diag(observer_config["gain"]*num_points)
    epsilon = observer_gain @ np.reshape(corners_raw_normalized, (2*len(corners_raw_normalized),))
    d_hat_dob = observer_gain @ np.reshape(corners_raw_normalized, (2*len(corners_raw_normalized),)) - epsilon

    # EKF initialization
    current_sample_time = time.time()
    state = robot.get_state()
    q, dq = state['q'], state['dq']
    info = pin_robot.getInfo(q,dq)
    corners_raw = deepcopy(target_corners_global)
    target_pos = deepcopy(target_pos_global)
    target_ori = deepcopy(target_ori_global)
    corners_raw_normalized = normalize_corners(corners_raw, fx, fy, cx, cy)
    apriltag_size = target_config["apriltag_size"]
    coord_in_cam_raw, coord_in_world_raw, _ = get_apriltag_corners_cam_and_world_homo_coord(
        apriltag_size/2, target_pos, target_ori, info["R_CAMERA"], info["P_CAMERA"])
    corner_depths_raw = coord_in_cam_raw[:,2]
    ekf_init_val = np.zeros((num_points, 9), dtype=np.float32)
    ekf_init_val[:,0:2] = corners_raw_normalized[0:len(corners_raw),:]
    ekf_init_val[:,2] = corner_depths_raw[0:len(corners_raw)]
    P0_unnormalized = np.diag(ekf_config["P0_unnormalized"])
    P0 = P0_unnormalized @ np.diag([1/fx**2,1/fy**2,1,1,1,1,1,1,1])
    Q_cov = np.diag(ekf_config["Q"])
    R_unnormalized = np.diag(ekf_config["R_unnormalized"])
    R_cov = R_unnormalized @ np.diag([1/fx**2,1/fy**2,1])
    ekf = EKF_IBVS(num_points, ekf_init_val, P0, Q_cov, R_cov)
    last_ekf_time = current_sample_time

    # Start the control loop
    print("==> Start the control loop")
    designed_control_loop_time = test_settings["designed_control_loop_time"]
    dq_executed = np.zeros(9, dtype=np.float32)
    last_info = info
    last_d_true_time = current_sample_time
    last_d_true_z_time = current_sample_time
    last_corners_raw_normalized=  corners_raw_normalized
    last_corner_depths_raw = corner_depths_raw
    last_J_image_cam_raw = np.zeros((2*corners_raw.shape[0], 6), dtype=np.float32)
    last_J_depth_raw = np.zeros((corners_raw.shape[0], 6), dtype=np.float32)
    time_start = time.time()

    for i in range(100000):
        time_loop_start = time.time()

        # Break if TF info is too old
        target_latest_timestamp = deepcopy(target_latest_timestamp_gloabl)
        if np.abs(target_latest_timestamp.to_sec() - time_loop_start) > 0.2:
            print("==> TF timestamp too old. Break the loop...")
            robot.send_joint_command(np.zeros(7))
            break
        
        # Get the current state of the robot and corners
        current_sample_time = time.time()
        state = robot.get_state()
        q, dq = state['q'], state['dq']
        info = pin_robot.getInfo(q,dq)
        corners_raw = deepcopy(target_corners_global)
        target_pos = deepcopy(target_pos_global)
        target_ori = deepcopy(target_ori_global)
        corners_raw_normalized = normalize_corners(corners_raw, fx, fy, cx, cy)

        # Target corners and depths
        coord_in_cam_raw, coord_in_world_raw, _ = get_apriltag_corners_cam_and_world_homo_coord(
        apriltag_size/2, target_pos, target_ori, info["R_CAMERA"], info["P_CAMERA"])
        corner_depths_raw = coord_in_cam_raw[:,2]

        # Speeds excuted in the camera framexs
        last_speeds_in_cam = dq_to_speeds_in_cam(dq, last_info["J_CAMERA"], last_info["R_CAMERA"])

        # Speed contribution due to movement of the apriltag (x, y)
        current_d_true_time = current_sample_time
        d_true = np.zeros(2*len(corners_raw_normalized), dtype=np.float32)
        dx_dy_raw = (corners_raw_normalized - last_corners_raw_normalized)/(current_d_true_time - last_d_true_time)
        dx_dy_raw = np.reshape(dx_dy_raw, (2*len(corners_raw),))
        d_true = dx_dy_raw - last_J_image_cam_raw @ last_speeds_in_cam
        last_corners_raw_normalized = corners_raw_normalized
        last_d_true_time = current_d_true_time

        # Speed contribution due to movement of the apriltag (Z)
        current_d_true_z_time = current_sample_time
        dZ_raw = (corner_depths_raw - last_corner_depths_raw)/(current_d_true_z_time - last_d_true_z_time)
        d_true_z = dZ_raw - last_J_depth_raw @ last_speeds_in_cam
        last_corner_depths_raw = corner_depths_raw
        last_d_true_z_time = current_d_true_z_time

        # Update the disturbance observer
        current_dob_time = time.time()
        dob_dt = current_dob_time - last_dob_time
        epsilon += dob_dt * observer_gain @ (last_J_image_cam_raw @last_speeds_in_cam + d_hat_dob)
        d_hat_dob = observer_gain @ np.reshape(corners_raw_normalized, (2*len(corners_raw_normalized),)) - epsilon
        last_dob_time = current_dob_time
        if np.any(np.isnan(d_hat_dob)): 
            print("==> d_hat_dob is nan. Break the loop...")
            break
        
        # Compute image jaccobians due to camera speed
        last_J_image_cam_raw = np.zeros((2*corners_raw_normalized.shape[0], 6), dtype=np.float32)
        for ii in range(len(corners_raw_normalized)):
            x, y = corners_raw_normalized[ii,:]
            Z = corner_depths_raw[ii]
            last_J_image_cam_raw[2*ii:2*ii+2] = one_point_image_jacobian_normalized(x,y,Z)
            
        last_J_depth_raw = np.zeros((corners_raw_normalized.shape[0], 6), dtype=np.float32)
        for ii in range(len(corners_raw)):
            x, y = corners_raw_normalized[ii,:]
            Z = corner_depths_raw[ii]
            last_J_depth_raw[ii] = one_point_depth_jacobian_normalized(x, y, Z)

        # Step and update the EKF
        current_ekf_time = time.time()
        ekf_dt = current_ekf_time-last_ekf_time
        ekf.predict(ekf_dt, last_speeds_in_cam)
        mesurements = np.hstack((corners_raw_normalized, corner_depths_raw[:,np.newaxis]))
        ekf.update(mesurements)
        last_ekf_time = current_ekf_time

        # Image jacobian
        ekf_updated_states = ekf.get_updated_state()
        corners_ekf_normalized = ekf_updated_states[:,0:2]
        corner_depths_ekf = ekf_updated_states[:,2]
        d_hat_ekf = ekf_updated_states[:,3:5].reshape(-1)
        J_image_cam_ekf = np.zeros((2*corners_ekf_normalized.shape[0], 6), dtype=np.float32)
        for ii in range(len(corners_ekf_normalized)):
            x, y = corners_ekf_normalized[ii,:]
            Z = corner_depths_ekf[ii]
            J_image_cam_ekf[2*ii:2*ii+2] = one_point_image_jacobian_normalized(x,y,Z)
        
        # Performance controller
        # # Compute desired pixel velocity (mean)
        # mean_gain = np.diag(controller_config["mean_gain"])
        # J_mean = 1/num_points*np.tile(np.eye(2), num_points)
        # error_mean = np.mean(corners_ekf_normalized[0:num_points,:], axis=0) - mean_target_normalized
        # xd_yd_mean = - LA.pinv(J_mean) @ mean_gain @ error_mean

        # # Compute desired pixel velocity (variance)
        # variance_gain = np.diag(controller_config["variance_gain"])
        # J_variance = np.tile(- np.diag(np.mean(corners_ekf_normalized[0:num_points,:], axis=0)), num_points)
        # J_variance[0,0::2] += corners_ekf_normalized[0:num_points,0]
        # J_variance[1,1::2] += corners_ekf_normalized[0:num_points,1]
        # J_variance = 2/num_points*J_variance
        # error_variance = np.var(corners_ekf_normalized[0:num_points,:], axis = 0) - variance_target_normalized
        # xd_yd_variance = - LA.pinv(J_variance) @ variance_gain @ error_variance

        # # Compute desired pixel velocity (distance)
        # distance_gain = controller_config["distance_gain"]
        # tmp = corners_ekf_normalized - desired_corners_normalized
        # error_distance = np.sum(tmp**2, axis=1)[0:num_points]
        # J_distance = np.zeros((num_points, 2*num_points), dtype=np.float32)
        # for ii in range(num_points):
        #     J_distance[ii, 2*ii:2*ii+2] = tmp[ii,:]
        # J_distance = 2*J_distance
        # xd_yd_distance = - distance_gain * LA.pinv(J_distance) @ error_distance

        # Compute desired pixel velocity (position)
        fix_position_gain = controller_config["fix_position_gain"]
        error_position = (corners_ekf_normalized - desired_corners_normalized).reshape(-1)
        J_position = np.eye(2*num_points, dtype=np.float32)
        xd_yd_position = - fix_position_gain * LA.pinv(J_position) @ error_position

        # Map to the camera speed expressed in the camera frame
        # null_mean = np.eye(2*num_points, dtype=np.float32) - LA.pinv(J_mean) @ J_mean
        # null_position = np.eye(2*num_points, dtype=np.float32) - LA.pinv(J_position) @ J_position
        # xd_yd = xd_yd_mean + null_mean @ xd_yd_position
        xd_yd = xd_yd_position

        J_active = J_image_cam_ekf[0:2*num_points]
        if observer_config["active"] == 1 and time.time() - time_start > observer_config["dob_kick_in_time"]:
            speeds_in_cam_desired = LA.pinv(J_active + J_image_cam_desired) @ (xd_yd - d_hat_dob[0:2*num_points])/2
        elif ekf_config["active"] == 1 and time.time() - time_start > ekf_config["ekf_kick_in_time"]:
            speeds_in_cam_desired = LA.pinv(J_active + J_image_cam_desired) @ (xd_yd - d_hat_ekf)/2
        else:
            speeds_in_cam_desired = LA.pinv(J_active + J_image_cam_desired) @ xd_yd/2

        # Map obstacle vertices to image
        _H = np.hstack((info["R_CAMERA"], np.reshape(info["P_CAMERA"],(3,1))))
        H_cam_to_world = np.vstack((_H, np.array([[0.0, 0.0, 0.0, 1.0]])))
        obstacle_corner_in_cam = obstacle_corners_in_world @ LA.inv(H_cam_to_world).T 
        obstacle_corner_depths = obstacle_corner_in_cam[:,2]
        obstacle_corners_normalized = obstacle_corner_in_cam[:,0:2]/obstacle_corner_in_cam[:,2][:,np.newaxis]

        # Solve CBF constraints if it is active
        if CBF_config["active"] == 0 or time.time() - time_start < CBF_config["cbf_active_time"]: 
            speeds_in_cam = speeds_in_cam_desired
            CBF = 0
        else:
            # Construct CBF and its constraint
            target_coords = torch.tensor(corners_ekf_normalized, dtype=torch.float32, requires_grad=True)
            x_target = target_coords[:,0]
            y_target = target_coords[:,1]
            A_target_val = torch.vstack((-y_target+torch.roll(y_target,-1), -torch.roll(x_target,-1)+x_target)).T
            b_target_val = -y_target*torch.roll(x_target,-1) + torch.roll(y_target,-1)*x_target

            obstacle_coords = torch.tensor(obstacle_corners_normalized, dtype=torch.float32, requires_grad=True)
            x_obstacle = obstacle_coords[:,0]
            y_obstacle = obstacle_coords[:,1]
            A_obstacle_val = torch.vstack((-y_obstacle+torch.roll(y_obstacle,-1), -torch.roll(x_obstacle,-1)+x_obstacle)).T
            b_obstacle_val = -y_obstacle*torch.roll(x_obstacle,-1) + torch.roll(y_obstacle,-1)*x_obstacle

            # check if the obstacle is far to avoid numerical instability
            A_obstacle_np = A_obstacle_val.detach().numpy()
            b_obstacle_np = b_obstacle_val.detach().numpy()
            tmp = kappa*(corners_ekf_normalized @ A_obstacle_np.T - b_obstacle_np)
            tmp = np.max(tmp, axis=1)
            # print(tmp)

            if np.min(tmp) > CBF_config["threshold_lb"] or np.max(tmp) > CBF_config["threshold_ub"]: 
                speeds_in_cam = speeds_in_cam_desired
                CBF = 0
                print("CBF active but skipped")
            else:
                alpha_sol, p_sol = cvxpylayer(A_target_val, b_target_val, A_obstacle_val, b_obstacle_val, 
                                                solver_args=optimization_config["solver_args"])
                CBF = alpha_sol.detach().numpy().item() - CBF_config["scaling_lb"]
                print(CBF)
                alpha_sol.backward()

                target_coords_grad = np.array(target_coords.grad)
                obstacle_coords_grad = np.array(obstacle_coords.grad)
                grad_CBF = np.hstack((target_coords_grad.reshape(-1), obstacle_coords_grad.reshape(-1)))
                grad_CBF_disturbance = target_coords_grad.reshape(-1)

                actuation_matrix = np.zeros((len(grad_CBF), 6), dtype=np.float32)
                actuation_matrix[0:2*len(target_coords_grad)] = J_image_cam_ekf
                for ii in range(len(obstacle_coords_grad)):
                    x, y = obstacle_corners_normalized[ii,:]
                    Z = obstacle_corner_depths[ii]
                    actuation_matrix[2*ii+2*len(target_coords_grad):2*ii+2+2*len(target_coords_grad)] = one_point_image_jacobian_normalized(x,y,Z)
                
                A_CBF = (grad_CBF @ actuation_matrix)[np.newaxis, :]
                if ekf_config["active_for_cbf"] == 1:
                    d_hat_cbf = d_hat_ekf
                elif observer_config["active_for_cbf"] == 1:
                    d_hat_cbf = d_hat_dob
                else:
                    d_hat_cbf = np.zeros(2*num_points, dtype=np.float32)
                lb_CBF = [-CBF_config["barrier_alpha"]*CBF + CBF_config["compensation"]\
                        - grad_CBF_disturbance @ d_hat_cbf]
                H = np.eye(6)
                g = -speeds_in_cam_desired

                cbf_qp.update(g=g, C=A_CBF, l=lb_CBF)
                cbf_qp.solve()

                speeds_in_cam = cbf_qp.results.x

        # Transform the speed back to the world frame
        v_in_cam = speeds_in_cam[0:3]
        omega_in_cam = speeds_in_cam[3:6]
        R_cam_to_world = info["R_CAMERA"]
        v_in_world = R_cam_to_world @ v_in_cam
        S_in_world = R_cam_to_world @ skew(omega_in_cam) @ R_cam_to_world.T
        omega_in_world = skew_to_vector(S_in_world)
        u_desired = np.hstack((v_in_world, omega_in_world))

        # Secondary objective: encourage the joints to stay in the middle of joint limits
        W = np.diag(-1.0/(joint_ub-joint_lb)**2) /len(joint_lb)
        q = info["q"]
        grad_joint = controller_config["joint_limit_gain"]* W @ (q - (joint_ub+joint_lb)/2)
        
        # Map the desired camera speed to joint velocities
        J_camera = info["J_CAMERA"]
        pinv_J_camera = LA.pinv(J_camera)
        dq_nominal = pinv_J_camera @ u_desired + (np.eye(9) - pinv_J_camera @ J_camera) @ grad_joint

        # QP-for joint limits
        q = info["q"]
        H = np.eye(9)
        g = - dq_nominal
        C = np.eye(9)*designed_control_loop_time
        joint_limits_qp.update(H=H, g=g, l=joint_lb - q, u=joint_ub - q, C=C)
        joint_limits_qp.solve()
        vel = joint_limits_qp.results.x
        vel[-2:] = 0

        # Robot velocity control
        vel = np.clip(vel, -0.5*np.pi, 0.5*np.pi)
        # vel = np.zeros_like(vel)

        # CBF for collision
        if collision_cbf_config["active"] == 1:
            print(vel)
            vel = collision_cbf.filter_dq(vel, info)
            print(vel)
            
        if time.time() - time_start < ekf_config["wait_ekf"]:
            vel = np.zeros_like(vel)

        robot.send_joint_command(vel[:7])

        # Keep for next loop
        last_info = info

        # Time the loop 
        time_loop_end = time.time()
        loop_time = time_loop_end-time_loop_start

        # Record data to history
        history["time"].append(time_loop_start)
        history["q"].append(q)
        history["dq"].append(dq)
        history["corners_raw_normalized"].append(corners_raw_normalized)
        history["corner_depths_raw"].append(corner_depths_raw)
        history["obstacle_corner_in_world"].append(obstacle_corners_in_world)
        history["obstacle_corner_in_image"].append(obstacle_corner_in_cam)
        history["error_position"].append(error_position)
        history["joint_vel_command"].append(vel)
        history["info"].append(info)
        history["d_hat_dob"].append(d_hat_dob)
        history["loop_time"].append(loop_time)
        history["ekf_estimates"].append(ekf_updated_states)
        history["d_true"].append(d_true)
        history["d_true_z"].append(d_true_z)
        history["dob_dt"].append(dob_dt)
        history["ekf_dt"].append(ekf_dt)
        if CBF_config["active"] == 1:
            history["cbf"].append(CBF)

        if test_settings["save_scaling_function"]==1:
            img_infra1_gray = deepcopy(gray_image_global)
            A_target_val = A_target_val.detach().numpy()
            b_target_val = b_target_val.detach().numpy()
            A_obstacle_val = A_obstacle_val.detach().numpy()
            b_obstacle_val = b_obstacle_val.detach().numpy()
            for ii in range(camera_config["width"]):
                for jj in range(camera_config["height"]):
                    pp = np.array(normalize_one_image_point(ii,jj,fx,fy,cx,cy))
                    if np.sum(np.exp(kappa * (A_target_val @ pp - b_target_val))) <= 4:
                        x, y = ii, jj
                        img_infra1_gray = cv2.circle(img_infra1_gray, (int(x),int(y)), radius=1, color=(0, 0, 255), thickness=-1)
                    if np.sum(np.exp(kappa * (A_obstacle_val @ pp - b_obstacle_val))) <= 4:
                        x, y = ii, jj
                        img_infra1_gray = cv2.circle(img_infra1_gray, (int(x),int(y)), radius=1, color=(0, 0, 255), thickness=-1)
            cv2.imwrite(results_dir+'/scaling_functions_'+'{:04d}.{}'.format(i, test_settings["image_save_format"]), img_infra1_gray)
            print("==> Scaling function saved")

        # Wait for the next control loop
        time.sleep(max(designed_control_loop_time - loop_time, 0))
    
    # Finished
    robot.send_joint_command(np.zeros(7))
    target_thread_stop_global = True
    apriltag_thread.join()
    image_thread_stop_global = True
    rospy.signal_shutdown("User requested shutdown")
    image_thread.join()

    # Save history data to result_dir
    with open(os.path.join(results_dir, "history.pkl"), "wb") as f:
        pickle.dump(history, f)

    print("==> Done")


