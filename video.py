# import the necessary packages
import numpy as np
import argparse
import cv2
import time

import torch

import norfair
from models import Yolov4
from norfair import Detection, Tracker, Video
from tool.torch_utils import do_detect
from tool.utils import load_class_names, plot_boxes_cv2
from transform.transform import *

max_distance_between_points = 30


class YOLO:
    def __init__(self, weightfile, use_cuda=False):
        self.use_cuda = use_cuda
        self.model = Yolov4(yolov4conv137weight=None, n_classes=80, inference=True)
        pretrained_dict = torch.load(
            weightfile, map_location=torch.device("cuda" if use_cuda else "cpu")
        )
        self.model.load_state_dict(pretrained_dict)

        if self.use_cuda:
            self.model.cuda()

    def __call__(self, img):
        width, height = 416, 416
        sized = cv2.resize(img, (width, height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        boxes = do_detect(self.model, sized, 0.4, 0.6, self.use_cuda)
        return boxes[0]


def euclidean_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)


def get_centroid(yolo_box, img_height, img_width):
    x1 = yolo_box[0] * img_width
    y1 = yolo_box[1] * img_height
    x2 = yolo_box[2] * img_width
    y2 = yolo_box[3] * img_height
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image file")
ap.add_argument("-v", "--video", help = "path to the video file")
ap.add_argument("-c", "--coords",
	help = "comma seperated list of source points")
ap.add_argument("-p", "--points", help="")
ap.add_argument("-2d", "--2dpoints", help="")
args = vars(ap.parse_args())
# load the image and grab the source coordinates (i.e. the list of
# of (x, y) points)
# NOTE: using the 'eval' function is bad form, but for this example
# let's just roll with it -- in future posts I'll show you how to
# automatically determine the coordinates without pre-supplying them

max_width = 600
max_height = 600
green_color = (0, 255, 0)
blue_color =  (255, 0, 0)

cap = cv2.VideoCapture(args["video"])

pts = []

def write_point(pts, out_path):
    if (out_path != None):
        out_file = open(out_path, "w")
        for i in range(0, 4):
            write_string = str(pts[i][0]) + " " + str(pts[i][1])
            if (i != 3):
                write_string += " "
            out_file.write(write_string)

def add_point(event, x, y, flags, param):
    global pts
    if event == cv2.EVENT_LBUTTONDBLCLK:
        pts += [(x, y)]
        print(pts)
        if len(pts) == 4:
            # show(np.asarray(pts))
            write_point(pts, args["points"])
            show_video()

def read_point_file(point_path):
    try:
        point_file = open(point_path, "r")
        line = point_file.readline()
        coor = line.split(" ")
        points = []
        if (len(coor) == 8):
            for i in range(0, 8, 2):
                    points += [(int(coor[i]), int(coor[i + 1]))]
        return points
    except:
        return []

def first_frame():
    global pts
    point_file = args["points"]
    print(point_file)
    if (point_file != None):
        pts = read_point_file(args["points"])
        if (len(pts) == 4):
            show_video()
        else:
            ret, frame = cap.read()
            frame = cv2.resize(frame, (max_width, max_height))
            cv2.imshow("frame", frame)
            cv2.setMouseCallback("frame", add_point)
            cv2.waitKey(0)
    else:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (max_width, max_height))
        cv2.imshow("frame", frame)
        cv2.setMouseCallback("frame", add_point)
        cv2.waitKey(0)


def show(pts):
    warped = four_point_transform(image, pts)
    # show the original and warped images
    cv2.imshow("Warped", warped)
    cv2.waitKey(0)


def show_video():
    tracker = Tracker(
        distance_function=euclidean_distance,
        distance_threshold=max_distance_between_points,
    )
    model = YOLO("yolov4.pth")  # set use_cuda=False if using CPU
    transformer = Transformer()
    transformer.four_point_transform(pts, max_width, max_height)
    cnt = 0

    points_2d = read_point_file(args["2dpoints"])
    points_2d = np.array(points_2d)

    while True:
        ret, frame = cap.read()
        cnt = cnt + 1
        frame = cv2.resize(frame, (max_width, max_height))
        for point in pts:
            frame = cv2.circle(np.float32(frame), (point[0], point[1]), 5, blue_color, 2)
        original_frame = frame

        time1 = time.time()
        detections = model(frame)
        detections = [
            Detection(get_centroid(box, frame.shape[0], frame.shape[1]), data=box)
            for box in detections
            if box[-1] == 2
        ]
        time2 = time.time()
        print("time : ", cnt, " ", time2 - time1)
        norfair.draw_points(frame, detections)
        tracked_objects = tracker.update(detections=detections)
        frame = transformer.get_warp(frame)
        frame_2d = np.zeros((max_width, max_height, 3))
        ratio = points_2d.max(axis=0) - points_2d.min(axis=0)
        print("ratio : ", ratio)
        for point in points_2d:
            frame_2d = cv2.circle(np.float32(frame_2d), (point[0], point[1]), 5, blue_color, 2)
        for tracked_object in tracked_objects:
            box_points = tracked_object.last_detection.points
            transform_point = transformer.convert_point(box_points)
            base_point = transformer.convert_point(pts[0])
            transform_point = [int(transform_point[0]), int(transform_point[1])]
            frame = cv2.circle(np.float32(frame), (transform_point[0], transform_point[1]), 5, green_color, 2)
            map_point = [(transform_point[0] - base_point[0]) / transformer.get_max_width(), (transform_point[1] - base_point[1]) / transformer.get_max_height()]
            map_point = [int(points_2d[0][0] + map_point[0] * ratio[0]), int(points_2d[0][1] + map_point[1] * ratio[1])]
            frame_2d = cv2.circle(np.float32(frame_2d), (map_point[0], map_point[1]), 5, green_color, 2)

        # norfair.draw_tracked_objects(frame, tracked_objects)

       	frame = frame.astype(np.uint8)
        frame_2d = frame_2d.astype(np.uint8)
        original_frame = original_frame.astype(np.uint8)
        cv2.imshow("original vid", original_frame)
        # cv2.imshow("transform vid", frame)
        cv2.imshow("2d vid", frame_2d)

        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break
    cap.release()
    cv2.destroyAllWindows()

first_frame()