import cv2
import os
import time
from pathlib import Path

def renmae_images(image_folder):
    images_raw = [img for img in os.listdir(image_folder) if img.endswith(".bmp")]
    print("renaming images...")
    for img in images_raw:
        new_name = "{:04d}.bmp".format(int(img[0:-4]))
        os.rename(os.path.join(image_folder, img), os.path.join(image_folder, new_name))
    print("Done!")
    return

def create_video(image_folder, video_name, frame_rate, repeat_to, rotate_by = None):
    images = [img for img in os.listdir(image_folder) if img.endswith(".bmp")] 
    images.sort()
    # print(images)

    print("Total images for the video: {}".format(len(images)))
    if len(images) < repeat_to:
        # Repeat the last image to make the video longer
        for i in range(repeat_to - len(images)):
            images.append(images[-1])
    # print the number of total images
    print("Total images with repeated last frame: {}".format(len(images)))

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    if rotate_by is not None:
            frame = cv2.rotate(frame, rotate_by)
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_name, fourcc, frame_rate, (width, height))

    print("Creating video...")
    time_start = time.time()
    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        if rotate_by is not None:
            frame = cv2.rotate(frame, rotate_by)
        video.write(frame)
    time_end = time.time()
    print("Done!")
    print("Time elapsed: {}s".format(time_end - time_start))
    video.release()

if __name__ == "__main__":
    # subfolder = 'external'
    # frame_rate = 30
    # repeat_to = 0
    # rotate_by = cv2.ROTATE_90_COUNTERCLOCKWISE

    subfolder = 'robot'
    frame_rate = 90
    repeat_to = 0
    rotate_by = None

    image_folder = str(Path(__file__).parent) + '/' + subfolder
    video_name = image_folder + '/' + subfolder + '_video.mp4'
    create_video(image_folder, video_name, frame_rate, repeat_to, rotate_by)