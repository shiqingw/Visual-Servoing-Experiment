import rosbag
import cv2
from cv_bridge import CvBridge
import os

def extract_images_from_rosbag(rosbag_path, topic_name, output_folder):
    """
    Extract images from a rosbag.

    Parameters:
    - rosbag_path: Path to the rosbag.
    - topic_name: Name of the topic containing the images.
    - output_folder: Folder where the extracted images will be saved.
    """

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    bridge = CvBridge()  # To convert ROS image messages to OpenCV format
    with rosbag.Bag(rosbag_path, "r") as bag:
        for idx, (topic, msg, ts) in enumerate(bag.read_messages(topics=[topic_name])):
            if topic == topic_name:
                try:
                    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                    image_filename = os.path.join(output_folder, "{:05d}.png".format(idx))
                    cv2.imwrite(image_filename, cv_image)
                except Exception as e:
                    print(f"Error processing image {idx}: {e}")

if __name__ == "__main__":
    BAG_PATH = '/home/mocap/visual_servo_ws/2023-09-08-16-19-07.bag'
    TOPIC_NAME = '/camera/color/image_raw'  # replace with the correct topic name
    OUTPUT_FOLDER = '/home/mocap/visual_servo_ws/extracted_images/robot'
    extract_images_from_rosbag(BAG_PATH, TOPIC_NAME, OUTPUT_FOLDER)