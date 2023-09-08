import numpy as np
import os
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
import copy
from scipy.spatial.transform import Rotation
from pathlib import Path


class RobotModel:
    def __init__(self):
        package_directory = str(Path(__file__).parent)
        robot_URDF = package_directory + "/robots/fr3_with_camera_and_bounding_boxes.urdf"
        self.robot = RobotWrapper.BuildFromURDF(robot_URDF, package_directory)

        # Get frame ID for grasp target
        self.jacobian_frame = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED

        # Get frame ids
        self.FR3_LINK3_BB_FRAME_ID = self.robot.model.getFrameId("fr3_link3_bounding_box")
        self.FR3_LINK4_BB_FRAME_ID = self.robot.model.getFrameId("fr3_link4_bounding_box")
        self.FR3_LINK5_1_BB_FRAME_ID = self.robot.model.getFrameId("fr3_link5_1_bounding_box")
        self.FR3_LINK5_2_BB_FRAME_ID = self.robot.model.getFrameId("fr3_link5_2_bounding_box")
        self.FR3_LINK6_BB_FRAME_ID = self.robot.model.getFrameId("fr3_link6_bounding_box")
        self.FR3_LINK7_BB_FRAME_ID = self.robot.model.getFrameId("fr3_link7_bounding_box")
        self.FR3_HAND_BB_FRAME_ID = self.robot.model.getFrameId("fr3_hand_bounding_box")
        self.FR3_CAMERA_FRAME_ID = self.robot.model.getFrameId("fr3_camera")
        self.frame_names_and_ids = {
            "LINK3_BB": self.FR3_LINK3_BB_FRAME_ID,
            "LINK4_BB": self.FR3_LINK4_BB_FRAME_ID,
            "LINK5_1_BB": self.FR3_LINK5_1_BB_FRAME_ID,
            "LINK5_2_BB": self.FR3_LINK5_2_BB_FRAME_ID,
            "LINK6_BB": self.FR3_LINK6_BB_FRAME_ID,
            "LINK7_BB": self.FR3_LINK7_BB_FRAME_ID,
            "HAND_BB": self.FR3_HAND_BB_FRAME_ID,
            "CAMERA": self.FR3_CAMERA_FRAME_ID,
        }

        self.base_R_offset = np.eye(3)
        self.base_p_offset = np.zeros((3,1))
        

    def compute_crude_location(self, base_R_offset, base_p_offset, frame_id):
        # get link orientation and position
        _p = self.robot.data.oMf[frame_id].translation
        _Rot = self.robot.data.oMf[frame_id].rotation

        # compute link transformation matrix
        _T = np.hstack((_Rot, _p[:, np.newaxis]))
        T = np.vstack((_T, np.array([[0.0, 0.0, 0.0, 1.0]])))

        # compute link offset transformation matrix
        _TW = np.hstack((base_R_offset, base_p_offset))
        TW = np.vstack((_TW, np.array([[0.0, 0.0, 0.0, 1.0]])))
        
        # get transformation matrix
        T_mat = TW @ T 

        # compute crude model location
        p = (T_mat @ np.array([[0.0], [0.0], [0.0], [1.0]]))[:3, 0]

        # compute crude model orientation
        Rot = T_mat[:3, :3]

        # quaternion
        q = Rotation.from_matrix(Rot).as_quat()

        return p, Rot, q

    def getInfo(self, q, dq):
        """
        info contains:
        -------------------------------------
        q: joint position
        dq: joint velocity
        J_{frame_name}: jacobian of frame_name
        P_{frame_name}: position of frame_name
        R_{frame_name}: orientation of frame_name
        """
        assert q.shape == (9,), "q vector should be 9,"
        assert dq.shape == (9,), "dq vector should be 9,"
        self.robot.computeJointJacobians(q)
        self.robot.framesForwardKinematics(q)
        self.robot.centroidalMomentum(q, dq)
        # Get Jacobian from grasp target frame
        # preprocessing is done in get_state_update_pinocchio()
        info = {"q": q,
                "dq": dq}
        for frame_name, frame_id in self.frame_names_and_ids.items():
            # Frame jacobian
            info[f"J_{frame_name}"] = self.robot.getFrameJacobian(frame_id, self.jacobian_frame)
            # Frame position and orientation
            (
                info[f"P_{frame_name}"],
                info[f"R_{frame_name}"],
                info[f"q_{frame_name}"],
            ) = self.compute_crude_location(
                self.base_R_offset, self.base_p_offset, frame_id
            )

        return info
