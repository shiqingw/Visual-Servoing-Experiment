from ekf.ekf_ibvs_new import EKF_IBVS
import numpy as np
import numpy.linalg as LA
import pickle
import os
import json
import shutil
from all_utils.vs_utils import one_point_image_jacobian, skew, skew_to_vector, one_point_depth_jacobian
from pathlib import Path

exp_num = 2
source_file_exp_name = "exp_018_2023-08-19-23-35-02"

# Load test settings
test_settings_path = "{}/back_test_settings/test_settings_{:03d}.json".format(str(Path(__file__).parent), exp_num)
with open(test_settings_path, "r", encoding="utf8") as f:
    test_settings = json.load(f)

# Create results directory
results_dir = "{}/back_test_results/{}".format(str(Path(__file__).parent), source_file_exp_name)

if not os.path.exists(results_dir):
    os.makedirs(results_dir)
shutil.copy(test_settings_path, results_dir)

# Load history
root_dir = "{}/results/".format(str(Path(__file__).parent))
source_results_dir = root_dir + source_file_exp_name
file = source_results_dir +  "/history.pkl"
with open(file, 'rb') as handle:
    _history = pickle.load(handle)

# Get data
_times = np.array(_history["time"]) # size (N,)
_times = _times - _times[0]
_q = np.array(_history["q"]) # size (N, 9)
_dq = np.array(_history["dq"]) # size (N, 9)
_corners_raw = np.array(_history["corners_raw"]) # size (N, 4, 2)
_corner_depths_raw = np.array(_history["corner_depths_raw"]) # size (N, 4)
_obstacle_corner_in_world = np.array(_history["obstacle_corner_in_world"]) # size (N, 4, 3)
_obstacle_corner_in_image = np.array(_history["obstacle_corner_in_image"]) # size (N, 4, 2)
_error_position =  np.array(_history["error_position"]) # size (N, 4)
_cbf = np.array(_history["cbf"]) # size (N,) or 0
_joint_vel_command = np.array(_history["joint_vel_command"]) # size (N, 9)
_info = _history["info"] # size (N,)
_d_hat_dob = np.array(_history["d_hat_dob"]) # size (N, 8)
_d_true =  np.array(_history["d_true"]) # size (N, 8)
_loop_time = np.array(_history["loop_time"]) # size (N, 1)
_ekf_estimates = np.array(_history["ekf_estimates"]) # size (N, 4, 9)
_dob_dt = np.array(_history["dob_dt"]) # size (N, 1)
_ekf_dt = np.array(_history["ekf_dt"]) # size (N, 1)

# Various configs
camera_config = test_settings["camera_config"]
controller_config = test_settings["controller_config"]
observer_config = test_settings["observer_config"]
CBF_config = test_settings["CBF_config"]
optimization_config = test_settings["optimization_config"]
obstacle_config = test_settings["obstacle_config"]
target_config = test_settings["target_config"]
ekf_config = test_settings["ekf_config"]

# Get camera parameters
intrinsic_matrix = np.array(camera_config["intrinsic_matrix"], dtype=np.float32)
fx = intrinsic_matrix[0, 0]
fy = intrinsic_matrix[1, 1]
cx = intrinsic_matrix[0, 2]
cy = intrinsic_matrix[1, 2]

# Record
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
            "ekf_estimates": [],
            "d_true_z": [],
            "dob_dt":[],
            "ekf_dt":[]
            }

# Disturbance observer initialization
num_points = 4
corners_raw =  _corners_raw[0,:,:]
observer_gain = np.diag(observer_config["gain"]*num_points)
epsilon = observer_gain @ np.reshape(corners_raw, (2*len(corners_raw),))
d_hat_dob = observer_gain @ np.reshape(corners_raw, (2*len(corners_raw),)) - epsilon

# Initialize EKF
num_points = 4
P0_unnormalized = np.diag(ekf_config["P0_unnormalized"])
P0 = P0_unnormalized @ np.diag([1/fx**2,1/fy**2,1,1,1,1,1,1])
Q_cov = np.diag(ekf_config["Q"])
R_unnormalized = np.diag(ekf_config["R_unnormalized"])
R_cov = R_unnormalized @ np.diag([1/fx**2,1/fy**2,1])
ekf_init_val = np.zeros((num_points, 8), dtype=np.float32)
ekf_init_val[:,0:2] = _corners_raw[0,:,:]
ekf_init_val[:,2] = _corner_depths_raw[0,:]
ekf = EKF_IBVS(num_points, ekf_init_val, P0, Q_cov, R_cov, fx, fy, cx, cy)

# Start loop
last_J_image_cam_raw = np.zeros((2*corners_raw.shape[0], 6), dtype=np.float32)
last_J_depth_raw = np.zeros((corners_raw.shape[0], 6), dtype=np.float32)
last_info = _info[0]
last_corners_raw = _corners_raw[0,:,:]
last_corner_depths_raw = _corner_depths_raw[0,:]

for i in range(1,len(_times)):
    # Last control input
    dq_executed = _dq[i,:]
    last_J_camera = last_info["J_CAMERA"]
    speeds_in_world = last_J_camera @ dq_executed
    v_in_world = speeds_in_world[0:3]
    omega_in_world = speeds_in_world[3:6]
    R_world_to_cam = last_info["R_CAMERA"].T
    v_in_cam = R_world_to_cam @ v_in_world
    S_in_cam = R_world_to_cam @ skew(omega_in_world) @ R_world_to_cam.T
    omega_in_cam = skew_to_vector(S_in_cam)
    last_speeds_in_cam = np.hstack((v_in_cam, omega_in_cam))

    # Current measurements
    corners_raw = _corners_raw[i,:,:]
    corner_depths_raw = _corner_depths_raw[i,:]

    # Calculate d_true_z
    dt = (_dob_dt[i]+_ekf_dt[i])/2.0
    dZ = (last_corner_depths_raw - corner_depths_raw)/dt
    d_true_z = dZ - last_J_depth_raw @ last_speeds_in_cam

    # Update the disturbance observer
    dob_dt = _dob_dt[i]
    epsilon += dob_dt * observer_gain @ (last_J_image_cam_raw @last_speeds_in_cam + d_hat_dob)
    d_hat_dob = observer_gain @ np.reshape(corners_raw, (2*len(corners_raw),)) - epsilon

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
    last_J_depth_raw = np.zeros((corners_raw.shape[0], 6), dtype=np.float32)
    for ii in range(len(corners_raw)):
        last_J_depth_raw[ii] = one_point_depth_jacobian(coord_in_cam_raw[ii], fx, fy)

    # Step and update EKF
    ekf_dt = _ekf_dt[i]
    ekf.predict(ekf_dt, last_speeds_in_cam)
    mesurements = np.hstack((corners_raw, corner_depths_raw[:,np.newaxis]))
    ekf.update(mesurements)

    # Pass for next loop
    last_info = _info[i]
    last_corner_depths_raw = _corner_depths_raw[i,:]

    # Record
    ekf_estimates = ekf.get_updated_state()
    history["ekf_estimates"].append(ekf_estimates)
    history["d_hat_dob"].append(d_hat_dob)
    history["d_true_z"].append(d_true_z)    

# Save history data to result_dir
history["time"] = _times[1:]
history["q"] = _q[1:,:]
history["dq"] = _dq[1:,:]
history["corners_raw"] = _corners_raw[1:,:,:]
history["corner_depths_raw"] = _corner_depths_raw[1:,:]
history["obstacle_corner_in_world"] = _obstacle_corner_in_world[1:,:,:]
history["obstacle_corner_in_image"] = _obstacle_corner_in_image[1:,:,:]
history["error_position"] = _error_position[1:,:]
history["joint_vel_command"] = _joint_vel_command[1:,:]
history["info"] = _info[1:]
history["loop_time"] = _loop_time[1:]
history["d_true"] = _d_true[1:,:]
history["dob_dt"] = _dob_dt[1:]
history["ekf_dt"] = _ekf_dt[1:]

with open(os.path.join(results_dir, "history.pkl"), "wb") as f:
    pickle.dump(history, f)

print("==> Done")
