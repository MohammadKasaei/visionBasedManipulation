import cv2
import numpy as np
import os
import sys
import math

from skimage.morphology import skeletonize, medial_axis
from skimage import data
import matplotlib.pyplot as plt
from skimage.util import invert
from sklearn import linear_model

def getCurvature(contour,stride=1):
    curvature=[]
    assert stride<len(contour),"stride must be shorther than length of contour"

    for i in range(len(contour)):
        
        before=i-stride+len(contour) if i-stride<0 else i-stride
        after=i+stride-len(contour) if i+stride>=len(contour) else i+stride
        
        f1x,f1y=(contour[after]-contour[before])/stride
        f2x,f2y=(contour[after]-2*contour[i]+contour[before])/stride**2
        denominator=(f1x**2+f1y**2)**3+1e-11
        
        curvature_at_i=np.sqrt(4*(f2y*f1x-f2x*f1y)**2/denominator) if denominator > 1e-12 else -1

        curvature.append(curvature_at_i)
    
    return curvature



"""
NOTES:
1. Find specific shape: circle, rectangle
    1.1 For circle-like objects: find the center -> use it as grasp position -> get radius of circle -> random rotation
    1.2 For rectangle: Find minAreaRect -> Calculate the rotation -> generate grasp candidates

"""




if __name__ == "__main__":
    idx = 1
    num_obj = 1
    rgb = cv2.imread("./dataset/{:04d}/{}_objects_rgb.png".format(idx, num_obj), cv2.COLOR_BGR2RGB)
    depth = cv2.imread("./dataset/{:04d}/{}_objects_depth.png".format(idx, num_obj), cv2.IMREAD_UNCHANGED)
    seg = cv2.imread("./dataset/{:04d}/{}_objects_seg.png".format(idx, num_obj), cv2.IMREAD_UNCHANGED)

    with open("./dataset/{:04d}/{}_objects_info.txt".format(idx, num_obj), 'r') as f:
        info = f.readlines()
    
    print(rgb.shape, depth.shape, seg.shape)

    reg = linear_model.LinearRegression()

    for line in info:
        rgb_c = np.copy(rgb)
        print(line.strip("\n").split(":"))
        idx = int(line.strip("\n").split(":")[-1])
        if line.strip("\n").split(":")[0] != 'YcbScissors':
            continue

        mask = (seg == idx).astype('uint8')
        # kernel = np.ones((5, 5), np.uint8)
        # mask = cv2.dilate(mask, kernel, iterations=1)
        cv2.imwrite("test_{}.png".format(idx), mask*255)

        # cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # for cnt in cnts:
        #     print
        #     approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
        #     print(approx)

        # # Erosion
        # kernel = np.ones((3, 3), np.uint8)
        # image = cv2.erode(mask, kernel) 
        # cv2.imwrite("test_erosion_{}.png".format(idx), image*255)

        edges = cv2.Canny(mask*255, 100, 200)
        cv2.imwrite("test_edges_{}.png".format(idx), edges)

        skeleton = (skeletonize(mask, method='lee')).astype('uint8')

        cv2.imwrite("test_skeleton_zha_{}.png".format(idx), skeleton*255)

        skel, distance = medial_axis(mask, return_distance=True)
        dist_on_skel = distance * skel

        cv2.imwrite("test_skeleton_medial_{}.png".format(idx), dist_on_skel*255)

        skeleton_lee = (skeletonize(mask, method='lee')).astype('uint8')*255

        cv2.imwrite("test_skeleton_lee_{}.png".format(idx), skeleton_lee)

        edge_with_skel = (skeleton + edges/255).astype('bool').astype('uint8')

        cv2.imwrite("test_edge_skel_{}.png".format(idx), edge_with_skel*255)

        edge_pts_y, edge_pts_x = np.nonzero(edges)
        skel_pts_y, skel_pts_x = np.nonzero(skeleton)

        slope_window = 10

        rects = []


        for i in range(len(skel_pts_x)):
            if i+5 >= len(skel_pts_x):
                break

            if i%5 == 0:
                
                x = skel_pts_x[i]
                y = skel_pts_y[i]

                x_ = skel_pts_x[i+5]
                y_ = skel_pts_y[i+5]


                if x != x_:
                    slope = (y_ - y) / (x_ - x)
                
                if slope == 0:
                    rect_slope = (-1)/1e-4
                else:
                    rect_slope = (-1)/slope


                # Point-Slope equation: y- y1 = m(x - x1)
                edge_y_pred   = rect_slope*(edge_pts_x - x) + y
                edge_y_offset = np.absolute(edge_pts_y - edge_y_pred)
                min_idx = np.argmin(edge_y_offset)
                pair_edge_x, pair_edge_y = edge_pts_x[min_idx], edge_pts_y[min_idx]

                width = 2 * np.sqrt((pair_edge_x - x)*(pair_edge_x - x) + (pair_edge_y - y)*(pair_edge_y - y)) + 10
                print(width)
                anlge_of_rect = np.arctan((-1)/slope) / np.pi * 180

                if width < 100:
                    rects.append([x, y, anlge_of_rect, width, 10])
        
        print(rects)


        for rect in rects:
            center_x, center_y, theta, width, height = rect
            box = ((int(center_x), int(center_y)), (width, height), theta)
            box = cv2.boxPoints(box)
            box = np.int0(box)
            cv2.drawContours(rgb_c, [box], 0, [255,0,0], 2)

        cv2.imwrite("test_rects_{}.png".format(idx), rgb_c)


        print(skel_pts_y, skel_pts_x)
