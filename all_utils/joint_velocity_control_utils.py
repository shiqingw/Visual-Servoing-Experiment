import numpy as np

def bring_to_nominal_q(robot, q0, joint_lb, joint_ub):
    """
    Bring the robot to the nominal configuration q0
    
    Args:   
    robot: robot interface
    q0 (numpy array): nominal configuration, size (robot.nq, )
    joint_lb (numpy array): lower bound of the joint, size (robot.nq, )
    joint_ub (numpy array): upper bound of the joint, size (robot.nq, )
    """
    W = np.diag(-1.0/(joint_ub-joint_lb)**2) /len(joint_lb)
    state = robot.get_state()
    q, dq = state['q'], state['dq']

    print("==> Moving the robot to the nominal configuration")
    while np.abs(q-q0).max() > 0.05:
        state = robot.get_state()
        q = state['q']
        grad_joint = 0.1* W @ (q - q0)
        robot.send_joint_command(grad_joint[:7])
    
    robot.send_joint_command(np.zeros(7))
    print("==> Robot is at the nominal configuration")
    return


