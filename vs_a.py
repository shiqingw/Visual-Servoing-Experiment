import argparse
import json
import signal
import sys
import threading
import time
from copy import deepcopy
from pathlib import Path

import apriltag
import cv2
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

sys.path.append(str(Path(__file__).parent.parent))

def one_point_image_jacobian(coord_in_cam, fx, fy):
    X = coord_in_cam[0]
    Y = coord_in_cam[1]
    Z = coord_in_cam[2]
    J1 = np.array([-fx/Z, 0, fx*X/Z**2, fx*X*Y/Z**2, fx*(-1-X**2/Z**2), fx*Y/Z], dtype=np.float32)
    J2 = np.array([0, -fy/Z, fy*Y/Z**2, fy*(1+Y**2/Z**2), -fy*X*Y/Z**2, -fy*X/Z], dtype=np.float32)

    return np.vstack((J1, J2))

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

def signal_handler(signal, frame):
    print("==> Ctrl+C received. Terminating threads...")
    # Set the exit_event to notify threads to stop
    sys.exit()

def apriltag_thread_func():
    def image_callback(data):
        global detector_global, target_corners_global, obstacle_corners_global
        try:
            # Process the image and perform AprilTag detection
            bridge = CvBridge()
            cv_image = bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")

            # Convert the image to grayscale (required by the AprilTag detector)
            gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # Detect AprilTags in the image
            results = detector_global.detect(gray_image)

            for ii in range(len(results)):
                res = results[ii]
                if res.tag_id == 0:
                    target_corners_global = res.corners
                elif res.tag_id == 1:
                    obstacle_corners_global = res.corners

            for ii in range(len(target_corners_global)):
                x, y = target_corners_global[ii,:]
                gray_image = cv2.circle(gray_image, (int(x),int(y)), radius=5, color=(0, 0, 255), thickness=-1)
            
            for ii in range(len(obstacle_corners_global)):
                x, y = obstacle_corners_global[ii,:]
                gray_image = cv2.circle(gray_image, (int(x),int(y)), radius=5, color=(255, 0, 0), thickness=-1)

            # cv2.imshow('debug', gray_image)
            # cv2.waitKey(1)                


        except Exception as e:
            rospy.logerr("Error processing the image: %s", str(e))

    # Initialize the ROS subscriber
    rospy.Subscriber('/camera/infra1/image_rect_raw', Image, image_callback)
    rospy.spin()
    return 

def depth_thread_func():
    def depth_callback(data):
        global depth_data_global
        try:
            bridge = CvBridge()
            depth_data_global = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")

            # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_data_global, alpha=0.03), cv2.COLORMAP_JET)
            # cv2.imshow('debug_depth', depth_colormap)
            # cv2.waitKey(1)

        except Exception as e:
            rospy.logerr("Error processing the image: %s", str(e))

    # Initialize the ROS subscriber
    rospy.Subscriber('/camera/depth/image_rect_raw', Image, depth_callback)
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

    # Joint limits
    joint_limits_config = test_settings["joint_limits_config"]
    joint_lb = np.array(joint_limits_config["lb"], dtype=np.float32)
    joint_ub = np.array(joint_limits_config["ub"], dtype=np.float32)

    # Camera parameters
    intrinsic_matrix = np.array(camera_config["intrinsic_matrix"], dtype=np.float32)
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    x0 = intrinsic_matrix[0, 2]
    y0 = intrinsic_matrix[1, 2]
    depth_scale = camera_config["depth_scale"]

    # Differential optimization layer
    nv = 2
    nc_target = optimization_config["n_cons_target"]
    nc_obstacle = optimization_config["n_cons_obstacle"]
    kappa = optimization_config["exp_coef"]

    _p = cp.Variable(nv)
    _alpha = cp.Variable(1)

    _A_target = cp.Parameter((nc_target, nv))
    _b_target = cp.Parameter(nc_target)
    _A_obstacle = cp.Parameter((nc_obstacle, nv))
    _b_obstacle = cp.Parameter(nc_obstacle)

    obj = cp.Minimize(_alpha)
    cons = [cp.sum(cp.exp(kappa*(_A_target @ _p - _b_target))) <= nc_target*_alpha, cp.sum(cp.exp(kappa*(_A_obstacle @ _p - _b_obstacle))) <= nc_obstacle*_alpha]
    problem = cp.Problem(obj, cons)
    assert problem.is_dpp()

    cvxpylayer = CvxpyLayer(problem, parameters=[_A_target, _b_target, _A_obstacle, _b_obstacle], variables=[_alpha, _p], gp=False)

    # Proxsuite for CBF-QP
    n = 6
    n_eq = 0
    n_in = 1
    cbf_qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
    # Randomly initialize the QP
    cbf_qp.init(np.eye(n), None, None, None, None, None, None)
    cbf_qp.settings.eps_abs = 1.0e-9
    cbf_qp.solve()

    # Proxsuite for inverse kinematics with joint limits
    n = 9
    n_eq = 0
    n_in = 9
    inv_kin_qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
    # Randomly initialize the QP
    inv_kin_qp.init(np.eye(n), None, None, None, None, None, None)
    inv_kin_qp.settings.eps_abs = 1.0e-9
    inv_kin_qp.solve()

    # Adjust mean and variance target to 3 points
    num_points = 3
    x0 = intrinsic_matrix[0, 2]
    y0 = intrinsic_matrix[1, 2]
    old_mean_target = np.array([x0,y0], dtype=np.float32)
    old_variance_target = np.array(controller_config["variance_target"], dtype=np.float32)
    desired_coords = np.array([[-1, -1],
                                [ 1, -1],
                                [ 1,  1],
                                [-1,  1]], dtype = np.float32)
    desired_coords = desired_coords*np.sqrt(old_variance_target) + old_mean_target
    mean_target = np.mean(desired_coords[0:num_points,:], axis=0)
    variance_target = np.var(desired_coords[0:num_points,:], axis = 0)

    # Initialize the global variables
    detector_global = apriltag.Detector()
    target_corners_global = []
    obstacle_corners_global = []
    depth_data_global = []

    # Create and start the apriltag thread
    rospy.init_node('apriltag_detection_node', anonymous=True)
    apriltag_thread = threading.Thread(target=apriltag_thread_func)
    apriltag_thread.daemon = True
    apriltag_thread.start()

    # Create and start the depth thread
    depth_thread = threading.Thread(target=depth_thread_func)
    depth_thread.daemon = True
    depth_thread.start()

    # Wait a little bit for the two threads
    print("==> Wait a little bit for the two threads...")
    time.sleep(1)

    # Capture obstacle world coordinates
    print("==> Turn camera toward the obstacle")
    input("==> Press Enter to if obstacle is in sight")
    obstacle_corners = deepcopy(obstacle_corners_global)
    depth_data = deepcopy(depth_data_global)
    obstacle_corner_depths = np.zeros([obstacle_corners.shape[0],1], dtype=np.float32)
    while np.min(obstacle_corner_depths) < 0.1:
        for ii in range(len(obstacle_corners)):
            x, y = obstacle_corners[ii,:]
            if not point_in_image(x, y, depth_data.shape[1], depth_data.shape[0]):
                x = np.clip(x, 0, depth_data.shape[1]-1)
                y = np.clip(y, 0, depth_data.shape[0]-1)
            obstacle_corner_depths[ii] = depth_data[int(y), int(x)]
        obstacle_corner_depths = depth_scale * obstacle_corner_depths
        if np.min(obstacle_corner_depths) < 0.1:
            print("==> Obstacle too close, please move it away")
            time.sleep(1)
            obstacle_corners = deepcopy(obstacle_corners_global)
            depth_data = deepcopy(depth_data_global)

    # Get the current state of the robot
    state = robot.get_state()
    q, dq = state['q'], state['dq']
    info = pin_robot.getInfo(q,dq)

    obstacle_pixel_coord = np.hstack((obstacle_corners, np.ones((obstacle_corners.shape[0],1), dtype=np.float32)))
    obstacle_pixel_coord_denomalized = obstacle_pixel_coord*obstacle_corner_depths
    
    obstalce_coord_in_cam = obstacle_pixel_coord_denomalized @ LA.inv(intrinsic_matrix.T)
    obstalce_coord_in_cam = np.hstack((obstalce_coord_in_cam, np.ones((obstalce_coord_in_cam.shape[0],1), dtype=np.float32)))

    # H = np.eye(4) # Homogeneous transformation matrix from camera to world frame
    _H = np.hstack((info["R_CAMERA"], np.reshape(info["P_CAMERA"],(3,1))))
    H = np.vstack((_H, np.array([[0.0, 0.0, 0.0, 1.0]])))
    obstacle_corner_in_world = obstalce_coord_in_cam @ H.T
    print("==> Obstacle world coordinates captured")
    print(obstacle_corner_in_world)

    # Observer initialization
    observer_gain = np.diag(observer_config["gain"]*observer_config["num_points"])

    # History
    history = {"time": [],
                "q": [],
                "dq": [],
                "corners": [],
                "corner_depths": [],
                "obstacle_corner_in_world": [],
                "obstacle_corner_in_image": [],
                "error_mean": [],
                "error_variance": [],
                "error_orientation": [],
                "cbf": [],
                "joint_vel_command":[], 
                "info":[],
                "d_hat": []
                }

    # Start the control loop
    print("==> Start the control loop")
    control_loop_wait_time = test_settings["control_loop_wait_time"]
    time_last = time.time()
    for i in range(100000):
        time_now = time.time()
        dt = time_now - time_last
        if len(target_corners_global) == 0 or len(depth_data_global) == 0: 
            time.sleep(1.0/10)
            print("==> No corners detected")
            continue
        
        # Get the current state of the robot
        state = robot.get_state()
        q, dq = state['q'], state['dq']
        info = pin_robot.getInfo(q,dq)

        corners = deepcopy(target_corners_global)
        depth_data = deepcopy(depth_data_global)

        # Initialize the observer such that d_hat = 0 at t = 0
        if not('epsilon' in locals()):
            epsilon = observer_gain @ np.reshape(corners, (2*len(corners),))
        
        corner_depths = np.zeros([corners.shape[0],1], dtype=np.float32)
        for ii in range(len(corners)):
            x, y = corners[ii,:]
            if not point_in_image(x, y, depth_data.shape[1], depth_data.shape[0]):
                x = np.clip(x, 0, depth_data.shape[1]-1)
                y = np.clip(y, 0, depth_data.shape[0]-1)
            corner_depths[ii] = depth_data[int(y), int(x)]
        corner_depths = depth_scale * corner_depths
        corner_depths = np.clip(corner_depths, 0.1, 10.0) # Clip the depth to be between 0.1 and 10 meters

        # Pixel coordinates to camera coordinates
        pixel_coord = np.hstack((corners, np.ones((corners.shape[0],1), dtype=np.float32)))
        pixel_coord_denomalized = pixel_coord*corner_depths
        
        coord_in_cam = pixel_coord_denomalized @ LA.inv(intrinsic_matrix.T)
        coord_in_cam = np.hstack((coord_in_cam, np.ones((coord_in_cam.shape[0],1), dtype=np.float32)))

        # Compute image jaccobian due to camera speed
        J_image_cam = np.zeros((2*corners.shape[0], 6), dtype=np.float32)
        fx = intrinsic_matrix[0, 0]
        fy = intrinsic_matrix[1, 1]
        for ii in range(len(corners)):
            J_image_cam[2*ii:2*ii+2] = one_point_image_jacobian(coord_in_cam[ii], fx, fy)

        # Compute desired pixel velocity (mean)
        mean_gain = np.diag(controller_config["mean_gain"])
        J_mean = 1/num_points*np.tile(np.eye(2), num_points)
        error_mean = np.mean(corners[0:num_points,:], axis=0) - mean_target
        xd_yd_mean = - LA.pinv(J_mean) @ mean_gain @ error_mean

        # Compute desired pixel velocity (variance)
        variance_gain = np.diag(controller_config["variance_gain"])
        J_variance = np.tile(- np.diag(np.mean(corners[0:num_points,:], axis=0)), num_points)
        J_variance[0,0::2] += corners[0:num_points,0]
        J_variance[1,1::2] += corners[0:num_points,1]
        J_variance = 2/num_points*J_variance
        error_variance = np.var(corners[0:num_points,:], axis = 0) - variance_target
        xd_yd_variance = - LA.pinv(J_variance) @ variance_gain @ error_variance

        # Compute desired pixel velocity (orientation)
        orientation_gain = np.diag([controller_config["horizontal_gain"], controller_config["vertical_gain"]])
        J_orientation = np.zeros((2, 2*len(corners)), dtype=np.float32)
        tmp1 = np.arange(0,len(corners),2,dtype=np.int_)
        tmp2 = np.arange(1,len(corners),2,dtype=np.int_)
        tmp = np.zeros(len(corners), dtype=np.int_)
        tmp[0::2] = tmp2
        tmp[1::2] = tmp1
        J_orientation[0,1::2] += corners[:,1] - corners[tmp,1]
        J_orientation[1,0::2] += corners[:,0] - np.flip(corners[:,0])
        J_orientation = 2*J_orientation
        J_orientation = J_orientation[:,0:2*num_points]
        J_orientation[1,0] = 0
        J_orientation[0,5] = 0
        error_orientation = np.sum(J_orientation**2, axis=1)/8.0
        xd_yd_orientation = - LA.pinv(J_orientation) @ orientation_gain @ error_orientation

        # Update the observer
        d_hat = observer_gain @ np.reshape(corners, (2*len(corners),)) - epsilon

        # Compute the desired speed in camera frame
        # xd_yd_mean and xd_yd_variance does not interfere each other, see Gans TRO 2011
        null_mean = np.eye(2*num_points, dtype=np.float32) - LA.pinv(J_mean) @ J_mean
        null_variance = np.eye(2*num_points, dtype=np.float32) - LA.pinv(J_variance) @ J_variance
        xd_yd = xd_yd_mean + xd_yd_variance + null_mean @ null_variance @ xd_yd_orientation
        J_active = J_image_cam[0:2*num_points]
        if observer_config["active"] == 1:
            speeds_in_cam_desired = J_active.T @ LA.inv(J_active @ J_active.T + 1*np.eye(2*num_points)) @ (xd_yd - d_hat[0:2*num_points])
        else:
            speeds_in_cam_desired = J_active.T @ LA.inv(J_active @ J_active.T + 1*np.eye(2*num_points)) @ xd_yd

        # Map obstacle vertices to image
        # H = np.eye(4) # Homogeneous transformation matrix from camera to world frame
        _H = np.hstack((info["R_CAMERA"], np.reshape(info["P_CAMERA"],(3,1))))
        H = np.vstack((_H, np.array([[0.0, 0.0, 0.0, 1.0]])))
        obstacle_corner_in_cam = obstacle_corner_in_world @ LA.inv(H).T 
        obstacle_corner_in_image = obstacle_corner_in_cam[:,0:3] @ intrinsic_matrix.T
        obstacle_corner_in_image = obstacle_corner_in_image/obstacle_corner_in_image[:,-1][:,np.newaxis]
        obstacle_corner_in_image = obstacle_corner_in_image[:,0:2]

        if CBF_config["active"] == 1:
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

            if np.min(tmp) <= CBF_config["threshold_lb"] and np.max(tmp) <= CBF_config["threshold_ub"]:
                time1 = time.time()
                alpha_sol, p_sol = cvxpylayer(A_target_val, b_target_val, A_obstacle_val, b_obstacle_val, 
                                                solver_args=optimization_config["solver_args"])
                CBF = alpha_sol.detach().numpy() - CBF_config["scaling_lb"]
                # print(alpha_sol, p_sol)
                print(CBF)
                alpha_sol.backward()
                time2 = time.time()
                # print("Time for CBF: ", time2-time1)

                target_coords_grad = np.array(target_coords.grad)
                obstacle_coords_grad = np.array(obstacle_coords.grad)
                grad_CBF = np.hstack((target_coords_grad.reshape(-1), obstacle_coords_grad.reshape(-1)))
                grad_CBF_disturbance = target_coords_grad.reshape(-1)

                actuation_matrix = np.zeros((len(grad_CBF), 6), dtype=np.float32)
                actuation_matrix[0:2*len(target_coords_grad)] = J_image_cam
                for ii in range(len(obstacle_coords_grad)):
                    actuation_matrix[2*ii+2*len(target_coords_grad):2*ii+2+2*len(target_coords_grad)] = one_point_image_jacobian(obstacle_corner_in_cam[ii,0:3], fx, fy)
                
                A_CBF = (grad_CBF @ actuation_matrix)[np.newaxis, :]
                lb_CBF = -CBF_config["barrier_alpha"]*CBF + CBF_config["compensation"]\
                        - grad_CBF_disturbance @ d_hat
                H = np.eye(6)
                g = -speeds_in_cam_desired

                cbf_qp.settings.initial_guess = (
                    proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
                )
                cbf_qp.update(g=g, C=A_CBF, l=lb_CBF)
                cbf_qp.settings.eps_abs = 1.0e-9
                cbf_qp.solve()

                speeds_in_cam = cbf_qp.results.x

            else: 
                speeds_in_cam = speeds_in_cam_desired
                CBF = 0
                print("CBF active but skipped")

        if np.any(np.isnan(d_hat)) or np.any(np.isnan(speeds_in_cam)):
            break

        # Transform the speed back to the world frame
        v_in_cam = speeds_in_cam[0:3]
        omega_in_cam = speeds_in_cam[3:6]
        R_cam_to_world = info["R_CAMERA"]
        # R_cam_to_world = np.eye(3)
        v_in_world = R_cam_to_world @ v_in_cam
        S_in_world = R_cam_to_world @ skew(omega_in_cam) @ R_cam_to_world.T
        omega_in_world = skew_to_vector(S_in_world)
        u_desired = np.hstack((v_in_world, omega_in_world))

        # Inverse kinematic with joint limits
        # q = np.zeros(9)
        # J_camera = np.zeros((6,9))
        J_camera = info["J_CAMERA"]
        H = J_camera.T @ J_camera
        g = - J_camera.T @ u_desired
        C = np.eye(9)*dt
        inv_kin_qp.settings.initial_guess = (
                proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
            )
        inv_kin_qp.update(H=H, g=g, l=joint_lb - q, u=joint_ub - q)
        inv_kin_qp.settings.eps_abs = 1.0e-9
        inv_kin_qp.solve()
        vel = inv_kin_qp.results.x
        vel[-2:] = 0

        # Robot velocity control
        robot.send_joint_command(vel[:7])

        # Step the observer
        epsilon += dt * observer_gain @ (J_image_cam @speeds_in_cam + d_hat)
        time_last = time_now
        
        # Record data to history
        history["time"].append(time_now)
        history["q"].append(q)
        history["dq"].append(dq)
        history["corners"].append(corners)
        history["corner_depths"].append(corner_depths)
        history["obstacle_corner_in_world"].append(obstacle_corner_in_world)
        history["obstacle_corner_in_image"].append(obstacle_corner_in_cam)
        history["error_mean"].append(error_mean)
        history["error_variance"].append(error_variance)
        history["error_orientation"].append(error_orientation)
        history["joint_vel_command"].append(vel)
        history["info"].append(info)
        history["d_hat"].append(d_hat)
        if CBF_config["active"] == 1:
            history["cbf"].append(CBF)

        # Wait for the next control loop
        time.sleep(control_loop_wait_time)

    # Save history data to result_dir
    with open(os.path.join(results_dir, "history.pkl"), "wb") as f:
        pickle.dump(history, f)

    # Finished
    print("==> Done")