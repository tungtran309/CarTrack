# import the necessary packages
from transform import four_point_transform
import numpy as np
import argparse
import cv2


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image file")
ap.add_argument("-v", "--video", help = "path to the video file")
ap.add_argument("-c", "--coords",
	help = "comma seperated list of source points")
args = vars(ap.parse_args())
# load the image and grab the source coordinates (i.e. the list of
# of (x, y) points)
# NOTE: using the 'eval' function is bad form, but for this example
# let's just roll with it -- in future posts I'll show you how to
# automatically determine the coordinates without pre-supplying them

cap = cv2.VideoCapture(args["video"])

pts = []
def add_point(event, x, y, flags, param):
    global pts
    if event == cv2.EVENT_LBUTTONDBLCLK:
        pts += [(x, y)]
        print(pts)
        if len(pts) == 4:
            # show(np.asarray(pts))
            show_video()


def first_frame():
    ret, frame = cap.read()
    frame = cv2.resize(frame, (900, 600))
    cv2.imshow("frame", frame)
    cv2.setMouseCallback("frame", add_point)
    cv2.waitKey(0)


def show(pts):
    warped = four_point_transform(image, pts)
    # show the original and warped images
    cv2.imshow("Warped", warped)
    cv2.waitKey(0)

def show_video():
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (900, 600))
        warped = four_point_transform(frame, pts)
        cv2.imshow("video", warped)
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break
    cap.release()
    cv2.destroyAllWindows()

first_frame()