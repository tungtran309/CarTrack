# import the necessary packages
from transform import *
import numpy as np
import argparse
import cv2

pts = []

transformer = Transformer()

# max_width = 1449
# max_height = 2048
max_width = 483
max_height = 682


def show(pts):
    height, width, channels = image.shape
    transformer.four_point_transform(pts, max_height, max_width)
    warped = transformer.get_warp(image)
    print(warped.shape)
    # show the original and warped images

    cv2.imshow("Warped", warped)
    cv2.waitKey(0)
    cv2.imwrite("Transformed_" + args["image"], warped)


def add_point(event, x, y, flags, param):
    global pts
    if event == cv2.EVENT_LBUTTONDBLCLK:
        pts += [(x, y)]
        print(pts)
        if len(pts) == 4:
            show(np.asarray(pts))

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image file")
ap.add_argument("-c", "--coords",
	help = "comma seperated list of source points")
args = vars(ap.parse_args())
# load the image and grab the source coordinates (i.e. the list of
# of (x, y) points)
# NOTE: using the 'eval' function is bad form, but for this example
# let's just roll with it -- in future posts I'll show you how to
# automatically determine the coordinates without pre-supplying them
image = cv2.imread(args["image"])
image = cv2.resize(image, (max_width, max_height))
cv2.imshow("image", image)

cv2.setMouseCallback("image", add_point)

# pts = np.array(eval(args["coords"]), dtype = "float32")
# apply the four point tranform to obtain a "birds eye view" of
# the image
cv2.waitKey(0)