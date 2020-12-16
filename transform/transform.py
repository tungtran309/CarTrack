# import the necessary packages
import numpy as np
import cv2

class Transformer:
    def __init__(self):
        self.M = None

    def order_points(self, pts):
        return np.asarray(pts, dtype="float32")
        # initialzie a list of coordinates that will be ordered
        # such that the first entry in the list is the top-left,
        # the second entry is the top-right, the third is the
        # bottom-right, and the fourth is the bottom-left
        rect = np.zeros((4, 2), dtype = "float32")
        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        s = pts.sum(axis = 1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        # now, compute the difference between the points, the
        # top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        # return the ordered coordinates
        return rect
    def get_max_width(self):
        return 900 / self.b
    def get_max_height(self):
        return 600 / self.b


    def convert_point(self, point):
        homg_point = [point[0], point[1], 1]
        trans_point = self.M.dot(homg_point)
        trans_point /= trans_point[2]
        # return trans_point[0:2]
        # return [trans_point[0] / self.maxWidth, trans_point[1] / self.maxHeight]
        return [trans_point[0] / self.maxWidth / self.b * 900, trans_point[1] / self.maxHeight / self.b * 600];


    def get_warp(self, image):
        # print(" ============== b : ", self.b)
        warped = cv2.warpPerspective(image, self.M, (self.maxWidth * self.b, self.maxHeight * self.b))
        # return the warped image
        return cv2.resize(warped, (900, 600))


    def four_point_transform(self, pts):
        # obtain a consistent order of the points and unpack them
        # individually
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect
        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # print(maxWidth, maxHeight)
        self.maxWidth = maxWidth
        self.maxHeight = maxHeight
        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left
        # order
        b = 9
        dst = np.array([
            [b // 2 * maxWidth, b//2 * maxHeight],
            [b // 2 * maxWidth + maxWidth, b//2 * maxHeight],
            [b // 2 * maxWidth + maxWidth, b//2 * maxHeight + maxHeight],
            [b // 2 * maxWidth, b//2 * maxHeight + maxHeight]
        ], dtype = "float32")

        self.b = b

        # dst = np.array([
        #     [maxWidth, maxHeight],
        #     [maxWidth * 2 - 1, maxHeight],
        #     [maxWidth * 2 - 1, maxHeight * 2 - 1],
        #     [maxWidth, maxHeight * 2 - 1]], dtype = "float32")
        # compute the perspective transform matrix and then apply it
        self.M = cv2.getPerspectiveTransform(rect, dst)

