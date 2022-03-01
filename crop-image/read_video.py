import cv2, os
import random

i = 0  # frame index to save frames
random.seed(0)
root = "/home/muyu/Downloads/video/"


def findfile(base):
    for root, dirs, files in os.walk(base):
        for f in files:
            yield os.path.join(root, f)


for file in findfile(root):
    # define the video path

    # capture the video
    cap = cv2.VideoCapture(file)
    # extract and save the video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
            break
        a = random.random()
        if a < 0.95:
            continue
        print("hah")
        cv2.imwrite("test_frame_" + str(i) + ".png", frame)
        i += 1

    cap.release()
    cv2.destroyAllWindows()
