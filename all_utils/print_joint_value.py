from FR3Py.interfaces import FR3Real
import sys
import signal
import time

def signal_handler(signal, frame):
    print("==> Ctrl+C received. Terminating script...")
    sys.exit()

if __name__ == '__main__':
    # Register the signal handler for SIGINT (Ctrl+C)
    signal.signal(signal.SIGINT, signal_handler)

    # Real robot interface
    print("==> Loading real robot interface...")
    robot = FR3Real(interface_type="joint_torque")

    print("==> Starting reading robot values...")
    for i in range(100000):
        state = robot.get_state()
        q = state['q']
        print("q: ", q)
        time.sleep(0.5)
