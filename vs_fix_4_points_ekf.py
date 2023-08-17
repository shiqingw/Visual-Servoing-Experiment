import argparse
import json
import signal
import sys
import threading
import time
from copy import deepcopy
from pathlib import Path
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
from cv_bridge import CvBridge
from cvxpylayers.torch import CvxpyLayer
from PIL import Image
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image
from RobotModel import RobotModel
from FR3Py.interfaces import FR3Real
from datetime import datetime
import os
import shutil
import pickle
import cv2
from all_utils.vs_utils import get_homogeneous_transformation, one_point_image_jacobian, skew, skew_to_vector, point_in_image
from all_utils.proxsuite_utils import init_prosuite_qp
from all_utils.cvxpylayers_utils import init_cvxpylayer
from ekf.ekf_ibvs import EKF_IBVS

sys.path.append(str(Path(__file__).parent.parent))

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

    global target_pose_global, target_ori_global
    global target_latest_timestamp_gloabl
    global target_thread_stop_global
    global camera_frame_global

    listener_target = tf.TransformListener()
    listener_target.waitForTransform(camera_frame_global, "/target", rospy.Time(), rospy.Duration(4.0))

    # Main loop to continuously update the pose
    while (not rospy.is_shutdown()) and (target_thread_stop_global == False):
        try:
            (trans, quat) = listener_target.lookupTransform(camera_frame_global, "/target", rospy.Time(0))
            target_pose_global = np.array(trans)
            target_ori_global = np.array(quat)
            target_latest_timestamp_gloabl = listener_target.getLatestCommonTime(camera_frame_global, "/target")

            rospy.sleep(0.001)  # Adjust the sleep duration as needed
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn("Error occurred while retrieving TF transform.") 

def tf_listener_obstacle_thread_func():

    global obstacle_pose_global, obstacle_ori_global
    global obstacle_thread_stop_global
    global camera_frame_global

    listener_obstacle = tf.TransformListener()
    listener_obstacle.waitForTransform(camera_frame_global, "/obstacle", rospy.Time(), rospy.Duration(4.0))

    # Main loop to continuously update the pose
    while (not rospy.is_shutdown()) and (obstacle_thread_stop_global == False):
        try:

            (trans, quat) = listener_obstacle.lookupTransform(camera_frame_global, "/obstacle", rospy.Time(0))
            obstacle_pose_global = np.array(trans)
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
    parser.add_argument('--exp_num', default=1, type=int, help="test case number")

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
        
    # Various configs
    camera_config = test_settings["camera_config"]
    controller_config = test_settings["controller_config"]
    observer_config = test_settings["observer_config"]
    CBF_config = test_settings["CBF_config"]
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

    # Starting ros node
    rospy.init_node('apriltag_detection_node', anonymous=True)

    # Create and start the obstacle thread
    print("==> Starting obstacle thread...")
    obstacle_pose_global, obstacle_ori_global = [], []
    obstacle_thread_stop_global = False
    obstacle_thread = threading.Thread(target=tf_listener_obstacle_thread_func)
    obstacle_thread.daemon = True
    obstacle_thread.start()
    print("==> Turn camera toward the obstacle")
    input("==> Press Enter to if obstacle is in sight")
    while len(obstacle_pose_global) == 0:
        print("==> Wait a little bit for the obstacle thread...")
        time.sleep(1)

    # Capture obstacle world coordinates
    apriltag_size = obstacle_config["apriltag_size"]
    offset = target_config["offset"]
    obstacle_corners_in_obs = np.array([[-1,1,0],[1,1,0],[1,-1,0],[-1,-1,0]], dtype=np.float32)*(apriltag_size/2+offset)
    obstacle_corners_in_obs = np.concatenate([obstacle_corners_in_obs, np.ones([4,1], dtype=np.float32)], axis=1)
    obstacle_pose = deepcopy(obstacle_pose_global)
    obstacle_ori = deepcopy(obstacle_ori_global)
    H_obs_to_cam = get_homogeneous_transformation(obstacle_pose, R.from_quat(obstacle_ori).as_matrix())
    state = robot.get_state()
    q, dq = state['q'], state['dq']
    info = pin_robot.getInfo(q,dq)
    _H = np.hstack((info["R_CAMERA"], np.reshape(info["P_CAMERA"],(3,1))))
    H_cam_to_world = np.vstack((_H, np.array([[0.0, 0.0, 0.0, 1.0]])))
    H_obs_to_world = H_cam_to_world @ H_obs_to_cam
    obstacle_corner_in_world = obstacle_corners_in_obs @ H_obs_to_world.T
    print("==> Obstacle world coordinates captured:")
    print(obstacle_corner_in_world)

    # Kill obstacle thread
    obstacle_thread_stop_global = True
    obstacle_thread.join()
    print("==> Obstacle thread terminated")

    # Create and start the apriltag thread
    print("==> Creating target thread...")
    target_pose_global, target_ori_global = [], []
    target_latest_timestamp_gloabl = 0
    target_thread_stop_global = False
    apriltag_thread = threading.Thread(target=tf_listener_target_thread_func)
    apriltag_thread.daemon = True
    apriltag_thread.start()
    while len(target_pose_global) == 0:
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
                "corners_raw": [],
                "corner_depths_raw": [],
                "obstacle_corner_in_world": [],
                "obstacle_corner_in_image": [],
                "error_position": [],
                "cbf": [],
                "joint_vel_command":[], 
                "info":[],
                "d_hat_dob": [],
                "d_true": [],
                "loop_time": [],
                "ekf_estimates": []
                }
    
    # Adjust mean and variance target to num_points
    num_points = 4
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    old_mean_target = np.array([cx,cy], dtype=np.float32)
    old_variance_target = np.array(controller_config["variance_target"], dtype=np.float32)
    desired_coords = np.array([[1, 1],
                                [ -1, 1],
                                [ -1,  -1],
                                [1,  -1]], dtype = np.float32)
    desired_coords = desired_coords*np.sqrt(old_variance_target) + old_mean_target
    mean_target = np.mean(desired_coords[0:num_points,:], axis=0)
    variance_target = np.var(desired_coords[0:num_points,:], axis = 0)
    
    # Check which observer to use
    if observer_config["active"] == 1 and ekf_config["active"] == 1:
        raise ValueError("Two observers cannot be active for controller at the same time.")
    if observer_config["active_for_cbf"] == 1 and ekf_config["active_for_cbf"] == 1:
        raise ValueError("Two observers cannot be active for cbf at the same time.")

    # Disturbance observer initialization
    num_points = 4
    corners_raw = deepcopy(target_corners_global)
    observer_gain = np.diag(observer_config["gain"]*num_points)
    epsilon = observer_gain @ np.reshape(corners_raw, (2*len(corners_raw),))
    d_hat_dob = observer_gain @ np.reshape(corners_raw, (2*len(corners_raw),)) - epsilon
    last_dob_time = time.time()
    last_J_image_cam_raw = np.zeros((2*corners_raw.shape[0], 6), dtype=np.float32)

    # EKF initialization
    state = robot.get_state()
    q, dq = state['q'], state['dq']
    info = pin_robot.getInfo(q,dq)
    corners_raw = deepcopy(target_corners_global)
    target_pose = deepcopy(target_pose_global)
    target_ori = deepcopy(target_ori_global)
    apriltag_size = target_config["apriltag_size"]
    target_corners_in_target = np.array([[-1,1,0],[1,1,0],[1,-1,0],[-1,-1,0]], dtype=np.float32)*(apriltag_size/2)
    target_corners_in_target = np.concatenate([target_corners_in_target, np.ones([4,1], dtype=np.float32)], axis=1)
    H_target_to_cam = get_homogeneous_transformation(target_pose, R.from_quat(target_ori).as_matrix())
    _H = np.hstack((info["R_CAMERA"], np.reshape(info["P_CAMERA"],(3,1))))
    H_cam_to_world = np.vstack((_H, np.array([[0.0, 0.0, 0.0, 1.0]])))
    coord_in_cam_raw = target_corners_in_target @ H_target_to_cam.T
    last_coord_in_world_raw = coord_in_cam_raw @ H_cam_to_world.T
    corner_depths_raw = (coord_in_cam_raw[:,2]).reshape(-1,1)
    ekf_init_val = np.zeros((num_points, 9), dtype=np.float32)
    ekf_init_val[:,0:2] = corners_raw[0:len(corners_raw),:]
    ekf_init_val[:,2] = corner_depths_raw[0:len(corners_raw),0]
    P0 = np.diag(ekf_config["P0"])
    Q_cov = np.diag(ekf_config["Q"])
    R_cov = np.diag(ekf_config["R"])
    ekf = EKF_IBVS(num_points, ekf_init_val, P0, Q_cov, R_cov, fx, fy, cx, cy)
    last_ekf_time = time.time()

    # Start the control loop
    print("==> Start the control loop")
    dq_executed = np.zeros(9, dtype=np.float32)
    state = robot.get_state()
    q, dq = state['q'], state['dq']
    last_info = pin_robot.getInfo(q,dq)
    designed_control_loop_time = test_settings["designed_control_loop_time"]
    last_d_true_time = time.time()
    time_start = time.time()

    for i in range(100000):
        time_loop_start = time.time()

        # Break if TF info is too old
        target_latest_timestamp = deepcopy(target_latest_timestamp_gloabl)
        if np.abs(target_latest_timestamp.to_sec() - time_loop_start) > 0.2:
            print("==> TF timestamp too old. Break the loop...")
            robot.send_joint_command(np.zeros(7))
            break
        
        # Get the current state of the robot
        state = robot.get_state()
        q, dq = state['q'], state['dq']
        info = pin_robot.getInfo(q,dq)

        # Target corners and depths
        corners_raw = deepcopy(target_corners_global)
        target_pose = deepcopy(target_pose_global)
        target_ori = deepcopy(target_ori_global)
        target_corners_in_target = np.array([[-1,1,0],[1,1,0],[1,-1,0],[-1,-1,0]], dtype=np.float32)*(apriltag_size/2)
        target_corners_in_target = np.concatenate([target_corners_in_target, np.ones([4,1], dtype=np.float32)], axis=1)
        H_target_to_cam = get_homogeneous_transformation(target_pose, R.from_quat(target_ori).as_matrix())
        _H = np.hstack((info["R_CAMERA"], np.reshape(info["P_CAMERA"],(3,1))))
        H_cam_to_world = np.vstack((_H, np.array([[0.0, 0.0, 0.0, 1.0]])))
        coord_in_cam_raw = target_corners_in_target @ H_target_to_cam.T
        coord_in_world_raw = coord_in_cam_raw @ H_cam_to_world.T
        corner_depths_raw = coord_in_cam_raw[:,2]

        # Speeds excuted in the camera frame
        last_J_camera = last_info["J_CAMERA"]
        speeds_in_world = last_J_camera @ dq_executed
        v_in_world = speeds_in_world[0:3]
        omega_in_world = speeds_in_world[3:6]
        R_world_to_cam = last_info["R_CAMERA"].T
        v_in_cam = R_world_to_cam @ v_in_world
        S_in_cam = R_world_to_cam @ skew(omega_in_world) @ R_world_to_cam.T
        omega_in_cam = skew_to_vector(S_in_cam)
        last_speeds_in_cam = np.hstack((v_in_cam, omega_in_cam))

        # Update the disturbance observer
        current_dob_time = time.time()
        epsilon += (current_dob_time - last_dob_time) * observer_gain @ (last_J_image_cam_raw @last_speeds_in_cam + d_hat_dob)
        d_hat_dob = observer_gain @ np.reshape(corners_raw, (2*len(corners_raw),)) - epsilon
        last_dob_time = current_dob_time
        if np.any(np.isnan(d_hat_dob)): 
            print("==> d_hat_dob is nan. Break the loop...")
            break
        pixel_coord_raw = np.hstack((corners_raw, np.ones((corners_raw.shape[0],1), dtype=np.float32)))
        pixel_coord_denomalized_raw = pixel_coord_raw*corner_depths_raw[:,np.newaxis]
        coord_in_cam_raw = pixel_coord_denomalized_raw @ LA.inv(intrinsic_matrix.T)
        last_J_image_cam_raw = np.zeros((2*corners_raw.shape[0], 6), dtype=np.float32)
        fx = intrinsic_matrix[0, 0]
        fy = intrinsic_matrix[1, 1]
        for ii in range(len(corners_raw)):
            last_J_image_cam_raw[2*ii:2*ii+2] = one_point_image_jacobian(coord_in_cam_raw[ii], fx, fy)

        # Step and update the EKF
        current_ekf_time = time.time()
        ekf.predict(current_ekf_time-last_ekf_time, speeds_in_cam)
        mesurements = np.hstack((corners_raw, corner_depths_raw[:,np.newaxis]))
        ekf.update(mesurements)
        last_ekf_time = current_ekf_time

        # Image jacobian
        ekf_estimates = ekf.get_updated_state()
        corners = ekf_estimates[:,0:2]
        corner_depths = ekf_estimates[:,2]
        d_hat_ekf = ekf_estimates[0:num_points,3:5].reshape(-1)
        pixel_coord = np.hstack((corners, np.ones((corners.shape[0],1), dtype=np.float32)))
        pixel_coord_denomalized = pixel_coord*corner_depths[:,np.newaxis]
        coord_in_cam = pixel_coord_denomalized @ LA.inv(intrinsic_matrix.T)
        coord_in_cam = np.hstack((coord_in_cam, np.ones((coord_in_cam.shape[0],1), dtype=np.float32)))
        J_image_cam = np.zeros((2*corners.shape[0], 6), dtype=np.float32)
        fx = intrinsic_matrix[0, 0]
        fy = intrinsic_matrix[1, 1]
        for ii in range(len(corners)):
            J_image_cam[2*ii:2*ii+2] = one_point_image_jacobian(coord_in_cam[ii], fx, fy)

        # Speed contribution due to movement of the apriltag
        d_true = np.zeros(2*len(corners), dtype=np.float32)
        current_d_true_time = time.time()
        for ii in range(len(corners)):
            speed_of_corner_in_world = (coord_in_world_raw[ii,0:3] - last_coord_in_world_raw[ii,0:3])/(current_d_true_time - last_d_true_time)
            speed_of_corner_in_cam = info["R_CAMERA"].T @ speed_of_corner_in_world.squeeze()
            d_true[2*ii:2*ii+2] = -J_image_cam[2*ii:2*ii+2,0:3] @ speed_of_corner_in_cam
        last_coord_in_world_raw = coord_in_world_raw
        last_d_true_time = current_d_true_time

        # Performance controller
        # Compute desired pixel velocity (mean)
        mean_gain = np.diag(controller_config["mean_gain"])
        J_mean = 1/num_points*np.tile(np.eye(2), num_points)
        error_mean = np.mean(corners[0:num_points,:], axis=0) - mean_target
        xd_yd_mean = - LA.pinv(J_mean) @ mean_gain @ error_mean
        # Compute desired pixel velocity (position)
        fix_position_gain = controller_config["fix_position_gain"]
        tmp = corners - desired_coords
        error_position = np.sum(tmp**2, axis=1)[0:num_points]
        J_position = np.zeros((num_points, 2*num_points), dtype=np.float32)
        for ii in range(num_points):
            J_position[ii, 2*ii:2*ii+2] = tmp[ii,:]
        xd_yd_position = - fix_position_gain * LA.pinv(J_position) @ error_position

        # Map to the camera speed expressed in the camera frame
        # null_mean = np.eye(2*num_points, dtype=np.float32) - LA.pinv(J_mean) @ J_mean
        # null_position = np.eye(2*num_points, dtype=np.float32) - LA.pinv(J_position) @ J_position
        # xd_yd = xd_yd_position + null_position @ xd_yd_mean
        xd_yd = xd_yd_position
        J_active = J_image_cam[0:2*num_points]
        if observer_config["active"] == 1:
            speeds_in_cam_desired = J_active.T @ LA.inv(J_active @ J_active.T + 1*np.eye(2*num_points)) @ (xd_yd - d_hat_dob[0:2*num_points])
        elif ekf_config["active"] == 1:
            speeds_in_cam_desired = J_active.T @ LA.inv(J_active @ J_active.T + 1*np.eye(2*num_points)) @ (xd_yd - d_hat_ekf)
        else:
            speeds_in_cam_desired = J_active.T @ LA.inv(J_active @ J_active.T + 1*np.eye(2*num_points)) @ xd_yd

        # Map obstacle vertices to image
        obstacle_corner_in_cam = obstacle_corner_in_world @ LA.inv(H_cam_to_world).T 
        obstacle_corner_in_image = obstacle_corner_in_cam[:,0:3] @ intrinsic_matrix.T
        obstacle_corner_in_image = obstacle_corner_in_image/obstacle_corner_in_image[:,-1][:,np.newaxis]
        obstacle_corner_in_image = obstacle_corner_in_image[:,0:2]

        # Solve CBF constraints if it is active
        if CBF_config["active"] == 0: 
            speeds_in_cam = speeds_in_cam_desired
        else:
            # Construct CBF and its constraint
            target_coords = torch.tensor(corners, dtype=torch.float32, requires_grad=True)
            x_target = target_coords[:,0]
            y_target = target_coords[:,1]
            A_target_val = torch.vstack((-y_target+torch.roll(y_target,-1), -torch.roll(x_target,-1)+x_target)).T
            b_target_val = -y_target*torch.roll(x_target,-1) + torch.roll(y_target,-1)*x_target

            obstacle_coords = torch.tensor(obstacle_corner_in_image, dtype=torch.float32, requires_grad=True)
            x_obstacle = obstacle_coords[:,0]
            y_obstacle = obstacle_coords[:,1]
            A_obstacle_val = torch.vstack((-y_obstacle+torch.roll(y_obstacle,-1), -torch.roll(x_obstacle,-1)+x_obstacle)).T
            b_obstacle_val = -y_obstacle*torch.roll(x_obstacle,-1) + torch.roll(y_obstacle,-1)*x_obstacle

            # check if the obstacle is far to avoid numerical instability
            A_obstacle_np = A_obstacle_val.detach().numpy()
            b_obstacle_np = b_obstacle_val.detach().numpy()
            tmp = kappa*(corners @ A_obstacle_np.T - b_obstacle_np)
            tmp = np.max(tmp, axis=1)

            if np.min(tmp) > CBF_config["threshold_lb"] or np.max(tmp) > CBF_config["threshold_ub"]: 
                speeds_in_cam = speeds_in_cam_desired
                CBF = 0
                print("CBF active but skipped")
            else:
                alpha_sol, p_sol = cvxpylayer(A_target_val, b_target_val, A_obstacle_val, b_obstacle_val, 
                                                solver_args=optimization_config["solver_args"])
                CBF = alpha_sol.detach().numpy() - CBF_config["scaling_lb"]
                print(CBF)
                alpha_sol.backward()

                target_coords_grad = np.array(target_coords.grad)
                obstacle_coords_grad = np.array(obstacle_coords.grad)
                grad_CBF = np.hstack((target_coords_grad.reshape(-1), obstacle_coords_grad.reshape(-1)))
                grad_CBF_disturbance = target_coords_grad.reshape(-1)

                actuation_matrix = np.zeros((len(grad_CBF), 6), dtype=np.float32)
                actuation_matrix[0:2*len(target_coords_grad)] = J_image_cam
                for ii in range(len(obstacle_coords_grad)):
                    actuation_matrix[2*ii+2*len(target_coords_grad):2*ii+2+2*len(target_coords_grad)] = one_point_image_jacobian(obstacle_corner_in_cam[ii,0:3], fx, fy)
                
                A_CBF = (grad_CBF @ actuation_matrix)[np.newaxis, :]
                if ekf_config["active_for_cbf"] == 1:
                    d_hat_cbf = d_hat_ekf
                else:
                    d_hat_cbf = d_hat_dob
                lb_CBF = -CBF_config["barrier_alpha"]*CBF + CBF_config["compensation"]\
                        - grad_CBF_disturbance @ d_hat_cbf
                H = np.eye(6)
                g = -speeds_in_cam_desired

                cbf_qp.settings.initial_guess = (
                    proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
                )
                cbf_qp.update(g=g, C=A_CBF, l=lb_CBF)
                cbf_qp.settings.eps_abs = 1.0e-9
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
        joint_limits_qp.settings.initial_guess = (
                proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
            )
        joint_limits_qp.update(H=H, g=g, l=joint_lb - q, u=joint_ub - q, C=C)
        joint_limits_qp.settings.eps_abs = 1.0e-9
        joint_limits_qp.solve()
        vel = joint_limits_qp.results.x
        vel[-2:] = 0

        # Robot velocity control
        vel = np.clip(vel, -1.0*np.pi, 1.0*np.pi)
        print(vel)
        # if time.time() - time_start > 4:
        #     robot.send_joint_command(vel[:7])

        # Keep for next loop
        dq_executed = vel
        last_info = info

        # Time the loop 
        time_loop_end = time.time()
        loop_time = time_loop_end-time_loop_start
        # print("==> Loop time: ", loop_time)

        # Record data to history
        history["time"].append(time_loop_start)
        history["q"].append(q)
        history["dq"].append(dq)
        history["corners_raw"].append(corners_raw)
        history["corner_depths_raw"].append(corner_depths_raw)
        history["obstacle_corner_in_world"].append(obstacle_corner_in_world)
        history["obstacle_corner_in_image"].append(obstacle_corner_in_cam)
        history["error_position"].append(error_position)
        history["joint_vel_command"].append(vel)
        history["info"].append(info)
        history["d_hat_dob"].append(d_hat_dob)
        history["loop_time"].append(loop_time)
        history["ekf_estimates"].append(ekf_estimates)
        history["d_true"].append(d_true)
        if CBF_config["active"] == 1:
            history["cbf"].append(CBF)

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


