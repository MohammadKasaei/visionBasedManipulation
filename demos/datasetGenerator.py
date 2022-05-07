from dis import dis
import time

from cv2 import getTrackbarPos
from environment.datasetEnv import DatasetEnvironment
from environment.camera.camera import Camera
from graspGenerator.grasp_generator import GraspGenerator

import pybullet as p
import numpy as np
import sys
import random
import os
import cv2

from skimage.morphology import skeletonize, medial_axis

class GraspControl():

    def __init__(self,env):

        self.env = env
        if self.env.robotType == "UR5":
            self.GRIPPER_MOVING_HEIGHT = 1.2
            self.GRIP_REDUCTION = 0.3
            self.GRIPPER_INIT_ORN = [-0*np.pi*0.25, np.pi/2, 0.0]
        elif self.env.robotType == "Panda":
            self.GRIPPER_MOVING_HEIGHT = 1.2
            self.GRIP_REDUCTION = 0.3
            self.GRIPPER_INIT_ORN = [np.pi, 0, 0]
        else:
            self.GRIPPER_MOVING_HEIGHT = 1.3
            self.GRIP_REDUCTION = 0.3
            self.GRIPPER_INIT_ORN = [0, 0, 0]
        
        
        self.gState = "Stop"
        self.lastStateUpdate = self.getTimeMs()
        self.updateState("Stop")
        self.pgState = self.gState 
        self.gPos = np.array([0,0,0])
        self.gOrn = np.array([0,np.pi/2,0])
        self.gWidth = 0.1
        self.depth_radius = 2

        self.EnableGraspingProcess = False    

        self.target = []

        self.TIMEOUT_MS = {
            "Stop": 100,
            "GoHome": 20000,
            "GraspDetection": 2000,
            "MoveOnTopofBox": 10000,
            "ReadyToGrasp": 20000,
            "Grasp": 5000,
            "Pickup": 5000,
            "MoveObjectIntoTarget": 10000,
            "DropObject": 5000
            }

    
    """
     TODO:
        
        1- rank grasp points: 
            select the best one (x,y,z)
            fail handeler:
                select another gp : partially done
        2- no grasp detection -> change the env
           > use next grasp point once it fails two times
        3- multiple robot type: DONE
        4- visualize prasp configuration 
        5- visualize camera pos and orientation

    """

    def updateState(self,newState):
        self.pgState = self.gState
        self.gState  = newState
        print(f"Prev: {self.pgState:20}\t Current: {self.gState:20}\t Time: {self.getTimeMs()-self.lastStateUpdate:20}")
        print ("-----------------------------------------------------------------------------------------------")

        self.lastStateUpdate =  self.getTimeMs()
        self.cnt = 0
     

    def timeoutControl(self):
        if (self.gState is not "Stop") and (self.getTimeMs() - self.lastStateUpdate > self.TIMEOUT_MS[self.gState]):
            self.updateState ("GoHome") 


    def getTimeMs(self):
        return round(time.time()*1000)
         
    def getTargetOrientation(self):
        
        if self.env.robotType == "UR5":
            # return   p.getQuaternionFromEuler([self.gOrn, np.pi/2, 0.0]) 
            return   p.getQuaternionFromEuler([0, np.pi/2, self.gOrn-np.pi/2]) 
        elif self.env.robotType == "Panda":
            return   p.getQuaternionFromEuler([-np.pi, 0, self.gOrn-np.pi/2]) 
        
    
    def graspAttempt(self, target):
        
        grasp_x, grasp_y, grasp_z, gripper_width, obj_height, theta, target_angle = target
        self.env.visualizePredictedGrasp([[grasp_x, grasp_y, grasp_z, theta, gripper_width, obj_height]])

        self.gPos = [grasp_x, grasp_y, np.clip(grasp_z+self.env.finger_length, *self.env.ee_position_limit[2])]
        self.gOrn = -theta
        self.gState = "GoHome"
        self.cnt = 0

        grasp_flag = True
        
        for i in range(50000):
            eeState   = self.env.getEEState()
            if self.gState  == "GoHome":
                targetPos    = self.env.TARGET_ZONE_POS[:]
                targetPos[2] = self.GRIPPER_MOVING_HEIGHT
                targetOrn    = p.getQuaternionFromEuler(self.GRIPPER_INIT_ORN)
                self.env.moveEE(targetPos, targetOrn)
                dist = np.linalg.norm(np.array(targetPos)-eeState[0])
                if (dist<0.3):
                    if (self.cnt>10):
                        self.updateState("MoveOnTopofBox")
                    else:
                        self.cnt += 1
                else:
                    self.cnt = 0

            elif self.gState == "MoveOnTopofBox":
                # Move to the top of box
                targetPos    = np.array(self.gPos)
                targetPos[2] = self.GRIPPER_MOVING_HEIGHT
                self.env.moveGripper(0.2)
                targetOrn    = self.getTargetOrientation() 
                self.env.moveEE(targetPos, targetOrn)
                dist = np.linalg.norm(np.array(targetPos)-eeState[0])
                if (dist<0.05):
                    if (self.cnt>10):
                        self.updateState("ReadyToGrasp")
                    else:
                        self.cnt += 1
                else:
                    self.cnt = 0

            elif self.gState == "ReadyToGrasp":
                targetPos    = np.array(self.gPos[:])
                targetOrn    = self.getTargetOrientation() 

                self.env.moveEE(targetPos, targetOrn)
                dist = np.linalg.norm(targetPos-eeState[0])
                thr = 0.14 if self.env.robotType == "UR5" else 0.02
                if (dist < thr): 
                    if (self.cnt>20):
                        self.updateState("Grasp")
                    else:
                        self.cnt += 1
                else:
                    self.cnt = 0
            
            elif self.gState  == "Grasp":
                succes_grasp = False
                self.env.moveGripper(0.01)
                time.sleep(0.025)
                grasped_id = self.env.checkGraspedID()
                if len(grasped_id) >= 1:
                    succes_grasp = True
                    grasped_obj_id = grasped_id[0]
            
                if succes_grasp:
                    self.cnt += 1
                    if self.cnt>50:
                        self.updateState("Pickup")   
                else:
                    self.cnt = 0
                    grasp_flag = False
                    # Failed to grasp object
                    break
                    
            
            elif self.gState  == "Pickup":
                targetPos    = np.array([self.gPos[0],self.gPos[1], self.GRIPPER_MOVING_HEIGHT])
                targetOrn    = self.getTargetOrientation() 
                self.env.moveEE(targetPos, targetOrn)
                dist = np.linalg.norm(targetPos-eeState[0])

                # Check if the obj is still graspsed
                grasped_id = self.env.checkGraspedID()
                if len(grasped_id) >= 1:
                    if (dist<0.14):
                        if (self.cnt>100):
                            self.updateState("MoveObjectIntoTarget")   
                        else:
                            self.cnt += 1
                    else:
                        self.cnt = 0
                else:
                    # Failed to pickup object
                    grasp_flag = False
                    break
            
            elif self.gState  == "MoveObjectIntoTarget":
                z = self.GRIPPER_MOVING_HEIGHT
                targetPos    = np.array([self.env.TARGET_ZONE_POS[0],self.env.TARGET_ZONE_POS[1],z])
                targetOrn    = self.getTargetOrientation() 
                self.env.moveEE(targetPos, targetOrn)
                dist = np.linalg.norm(targetPos-eeState[0])
                # Check if the obj is still graspsed
                if (dist<0.15):
                    self.updateState("DropObject")

            elif self.gState  == "DropObject":
                z = self.env.TARGET_ZONE_POS[2]+0.1
                targetPos    = np.array([self.env.TARGET_ZONE_POS[0],self.env.TARGET_ZONE_POS[1],z])
                targetOrn    = self.getTargetOrientation()#p.getQuaternionFromEuler([self.gOrn[0], np.pi/2, 0.0])
                self.env.moveEE(targetPos, targetOrn)
                dist = np.linalg.norm(targetPos-eeState[0])
                if (dist<0.15):
                    if (self.cnt>10):
                        self.env.moveGripper(0.1)
                        break
                    else:
                        self.cnt += 1
                else:
                    self.cnt = 0
        
        return grasp_flag



        
    def graspStateMachine(self):
        self.timeoutControl()
        eeState   = self.env.getEEState()
        
        if  self.gState  == "Stop":
            self.env.moveGripper(0.1)
            self.env.moveEE(eeState[0], eeState[1],max_step=1)
            
        elif self.gState  == "GoHome":
            targetPos    = self.env.TARGET_ZONE_POS[:]
            targetPos[2] = self.GRIPPER_MOVING_HEIGHT
            targetOrn    = p.getQuaternionFromEuler(self.GRIPPER_INIT_ORN)
            self.env.moveEE(targetPos, targetOrn)
            dist = np.linalg.norm(np.array(targetPos)-eeState[0])
            if (dist<0.15):
               if (self.cnt>10):
                if (self.EnableGraspingProcess and self.env.isThereAnyObject()):
                    self.updateState("WaitForGraspCandidate")
                else:
                    print ("Accomplished...")
                    # self.env.creatPileofTube(1)
                    # self.updateState("Stop")
               else:
                self.cnt += 1
            else:
                self.cnt = 0

        elif self.gState  == "WaitForGraspCandidate":
            if (len(self.target) == 0):
                print ("Receive no grasp candidates")
                self.cnt+=1
                if self.cnt > 3:
                    self.updateState("GoHome")
            else:
                grasp_x, grasp_y, grasp_z, gripper_width, obj_height, theta = self.target
                env.visualizePredictedGrasp([[grasp_x, grasp_y, grasp_z, theta, gripper_width, obj_height]])

                self.gPos = [grasp_x, grasp_y, np.clip(grasp_z+self.env.finger_length, *self.env.ee_position_limit[2])]

                self.gOrn = theta
                self.updateState("MoveOnTopofBox")

        
        elif self.gState  == "MoveOnTopofBox":
            targetPos    = np.array(self.gPos[:])
            targetPos[2] = self.GRIPPER_MOVING_HEIGHT
            self.env.moveGripper(0.2)
            targetOrn    = self.getTargetOrientation() 
            self.env.moveEE(targetPos, targetOrn)
            dist = np.linalg.norm(np.array(targetPos)-eeState[0])
            if (dist<0.05):
               if (self.cnt>10):
                self.updateState("ReadyToGrasp")
               else:
                self.cnt += 1
            else:
                self.cnt = 0

                
            
        elif self.gState  == "ReadyToGrasp":
            targetPos    = np.array(self.gPos[:])
            targetOrn    = self.getTargetOrientation() 
            self.env.moveEE(targetPos, targetOrn)
            dist = np.linalg.norm(targetPos-eeState[0])
            thr = 0.14 if self.env.robotType == "UR5" else 0.02
            if (dist < thr): 
               if (self.cnt>20):
                self.updateState("Grasp")
               else:
                self.cnt += 1
            else:
                self.cnt = 0


        elif self.gState  == "Grasp":

            succes_grasp = False
            self.env.moveGripper(0.01)
            time.sleep(0.025)
            grasped_id = self.env.checkGraspedID()
            if len(grasped_id) >= 1:
                succes_grasp = True
                grasped_obj_id = grasped_id[0]
           
            if succes_grasp:
                self.cnt += 1
                if self.cnt>50:
                    self.updateState("Pickup")   
            else:
                self.cnt = 0
        
        elif self.gState  == "Pickup":

            targetPos    = np.array([self.gPos[0],self.gPos[1], self.GRIPPER_MOVING_HEIGHT])
            targetOrn    = self.getTargetOrientation() 
            self.env.moveEE(targetPos, targetOrn)
            dist = np.linalg.norm(targetPos-eeState[0])
            if (dist<0.14):
               if (self.cnt>100):
                 self.updateState("MoveObjectIntoTarget")   
               else:
                 self.cnt += 1
            else:
                self.cnt = 0
            
        elif self.gState  == "MoveObjectIntoTarget":
            z = self.GRIPPER_MOVING_HEIGHT
            targetPos    = np.array([self.env.TARGET_ZONE_POS[0],self.env.TARGET_ZONE_POS[1],z])
            targetOrn    = self.getTargetOrientation() 
            self.env.moveEE(targetPos, targetOrn)
            dist = np.linalg.norm(targetPos-eeState[0])
            if (dist<0.15):
                self.updateState("DropObject")
                
        elif self.gState  == "DropObject":
            z = self.env.TARGET_ZONE_POS[2]+0.1
            targetPos    = np.array([self.env.TARGET_ZONE_POS[0],self.env.TARGET_ZONE_POS[1],z])
            targetOrn    = self.getTargetOrientation()#p.getQuaternionFromEuler([self.gOrn[0], np.pi/2, 0.0])
            self.env.moveEE(targetPos, targetOrn)
            dist = np.linalg.norm(targetPos-eeState[0])
            if (dist<0.15):
                if (self.cnt>10):
                    self.env.moveGripper(0.1)
                    self.updateState("GoHome")   
                else:
                    self.cnt += 1
            else:
                self.cnt = 0

            

def generate_grasp_candidates(rgb, seg, obj_info, prefix, save_results=False):
    data_dict = dict()
    for (obj_name, obj_idx) in obj_info:
        rgb_c = np.copy(rgb)

        mask = (seg == obj_idx).astype('uint8')

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        edges = cv2.Canny(mask*255, 100, 200)
        skeleton = (skeletonize(mask, method='lee')).astype('uint8')
        contours, _ = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        shape_results = []
        for cnt in contours:
            approx = cv2.approxPolyDP(
                    cnt, 0.01 * cv2.arcLength(cnt, True), True)
            
            cv2.drawContours(rgb_c, [cnt], 0, (0, 0, 255), 5)

            shape_results.append(approx)


        if len(shape_results) == 1:

            if len(shape_results[0]) <= 6:
                # print("Detect specific shape")

                # Find orthogonal line pair
                norm = []
                num_pts = shape_results[0].shape[0]
                last_slope = 0
                for i in range(num_pts):
                    if i == (num_pts -1):
                        next_i = 0
                        pts_1 = shape_results[0][i].reshape(2)
                        pts_2 = shape_results[0][next_i].reshape(2)
                    else:
                        next_i = i+1
                        pts_1 = shape_results[0][i].reshape(2)
                        pts_2 = shape_results[0][next_i].reshape(2)
                    
                    # Visualize codes
                    # cv2.circle(rgb_c, (pts_1[0], pts_1[1]), 1, (255,0,0), 3)
                    # cv2.putText(rgb_c, "{}".format(i), (pts_1[0], pts_1[1]), cv2.FONT_HERSHEY_SIMPLEX, 
                    #             1, (0,0,255), 1, cv2.LINE_AA)

                    # For the first point, record the slope of current line
                    if i == 0:
                        # Deal with vertical line
                        if (pts_2[0] - pts_1[0]) == 0:
                            slope = 9999
                            last_slope = 9999
                        else:
                            slope = (pts_2[1] - pts_1[1]) / (pts_2[0] - pts_1[0])
                            last_slope = slope
                            # print("idx: {}-{}, slope: {}".format(i, next_i, slope))
                            # cv2.putText(rgb_c, "idx: {}-{}, slope: {}".format(i, next_i, slope), (pts_1[0], pts_1[1]), cv2.FONT_HERSHEY_SIMPLEX, 
                            #     0.3, (255,0,0), 1, cv2.LINE_AA)
                    else:
                        # Deal with vertical line
                        if (pts_2[0] - pts_1[0]) == 0:
                            slope = 9999
                            if abs(last_slope) < 0.3:
                                # print("Find orthogonal lines")
                                norm = [slope, last_slope]
                                break
                        else:
                            slope = (pts_2[1] - pts_1[1]) / (pts_2[0] - pts_1[0])
                            tmp = last_slope * slope
                            if abs(abs(tmp)-1) < 0.15:
                                # print("Find orthogonal lines")
                                norm = [slope, last_slope]
                                break
                            last_slope = slope

                            # print("idx: {}-{}, slope: {}".format(i, next_i, slope))
                            # cv2.putText(rgb_c, "idx: {}-{}, slope: {}".format(i, next_i, slope), (pts_1[0], pts_1[1]), cv2.FONT_HERSHEY_SIMPLEX, 
                            #     0.3, (255,0,0), 1, cv2.LINE_AA)

                # Once we get two orthogonal lines, we uses slopes of them as the slopes of two object central axis                
                edge_pts_y, edge_pts_x = np.nonzero(edges)
                center_y = np.mean(edge_pts_y)
                center_x = np.mean(edge_pts_x)

                rects = []
                if len(norm) > 0:

                    for i in range(len(norm)):
                        print(norm)
                        # Slope: slope of current axis
                        # Grasp rectangles should be arthogonal to the current axis
                        slope = norm[i]
                        edge_y_pred   = slope*(edge_pts_x - center_x) + center_y
                        edge_y_offset = np.absolute(edge_pts_y - edge_y_pred)
                        min_idx = np.argmin(edge_y_offset)

                        # Get in point in edge
                        pair_edge_x, pair_edge_y = edge_pts_x[min_idx], edge_pts_y[min_idx]
                        delta_x = abs(center_x-pair_edge_x) 

                        # cv2.line(rgb_c, (int(center_x), int(center_y)), (int(pair_edge_x), int(pair_edge_y)), (0,0,255), 2)

                        if pair_edge_x < center_x:
                            sample_range = [pair_edge_x, center_x+delta_x]
                        else:
                            sample_range = [center_x-delta_x, pair_edge_x]

                        edge_y_pred   = norm[1-i]*(edge_pts_x - center_x) + center_y
                        edge_y_offset = np.absolute(edge_pts_y - edge_y_pred)
                        min_idx = np.argmin(edge_y_offset)
                        pair_edge_x, pair_edge_y = edge_pts_x[min_idx], edge_pts_y[min_idx]
                        # Calculate width
                        width = 2 * np.sqrt((pair_edge_x - center_x)*(pair_edge_x - center_x) + (pair_edge_y - center_y)*(pair_edge_y - center_y)) + 20

                        # Calculate angle
                        angle_of_grasp = np.arctan(norm[1-i])
                        # uniformly sample from two orthogonal axis

                        for k in range(int(sample_range[0]+5),int(sample_range[1]-5), 6):
                            # Avoid grasp candidates with extreme width
                            if width < 300:
                                cx = k
                                cy = slope*(cx - center_x) + center_y
                                rects.append([cx, cy, width, 30, angle_of_grasp])
                        
                        data_dict[obj_name] = rects
                else:

                    edge_pts_y, edge_pts_x = np.nonzero(edges)
                    skel_pts_y, skel_pts_x = np.nonzero(skeleton)

                    rects = []
                    for i in range(3, len(skel_pts_x)-3):
                        if i%5 == 0:
                            
                            pts_x = skel_pts_x[i-3: i+3]
                            pts_y = skel_pts_y[i-3: i+3]

                            x = skel_pts_x[i]
                            y = skel_pts_y[i]

                            z = np.poly1d(np.polyfit(pts_x,pts_y,2))

                            slope = x * (2 * z.c[0]) + z.c[1]
                            norm = (-1) / slope
                            
                            # Point-Slope equation: y- y1 = m(x - x1)
                            edge_y_pred   = norm*(edge_pts_x - x) + y
                            edge_y_offset = np.absolute(edge_pts_y - edge_y_pred)
                            min_idx = np.argmin(edge_y_offset)
                            pair_edge_x, pair_edge_y = edge_pts_x[min_idx], edge_pts_y[min_idx]

                            width = 2 * np.sqrt((pair_edge_x - x)*(pair_edge_x - x) + (pair_edge_y - y)*(pair_edge_y - y)) + 10
                            anlge_of_rect = np.arctan(norm)

                            # Abandon those grasp rects with hugh width
                            if width < 100:
                                rects.append([x, y, width, 30, anlge_of_rect])
                            
                    data_dict[obj_name] = rects 

                # for rect in rects:
                #     center_x, center_y, width, height, theta = rect
                #     box = ((int(center_x), int(center_y)), (width, height), theta)
                #     box = cv2.boxPoints(box)
                #     box = np.int0(box)
                #     cv2.drawContours(rgb_c, [box], 0, [255,0,0], 2)


                # cv2.circle(rgb_c, (int(center_x), int(center_y)), 1, (255,0,0), 3)

                # cv2.imwrite("test_rectangle_{}.png".format(idx), rgb_c)
            else:
                # Try to find circle in image
                edge_pts_y, edge_pts_x = np.nonzero(edges)
                center_y = np.mean(edge_pts_y)
                center_x = np.mean(edge_pts_x)

                dist_x = (edge_pts_x-center_x)*(edge_pts_x-center_x)
                dist_y = (edge_pts_y-center_y)*(edge_pts_y-center_y)
                dist = np.sqrt((dist_x+dist_y))
                dist_std = np.std(dist)

                if dist_std < 5:
                    radius = np.mean(dist)
                    rects = []
                    for k in range(0, 180, 30):
                        angle = k / 180 * np.pi
                        rects.append([center_x, center_y, radius*2+20, 30, k])
                    
                    data_dict[obj_name] = rects
                    
                    # for rect in rects:
                    #     center_x, center_y, width, height, theta = rect
                    #     box = ((int(center_x), int(center_y)), (width, height), theta)
                    #     box = cv2.boxPoints(box)
                    #     box = np.int0(box)
                    #     cv2.drawContours(rgb_c, [box], 0, [255,0,0], 2)
                    # cv2.imwrite("test_circle_{}.png".format(idx), rgb_c)

                else:
                    edge_pts_y, edge_pts_x = np.nonzero(edges)
                    skel_pts_y, skel_pts_x = np.nonzero(skeleton)

                    rects = []
                    for i in range(3, len(skel_pts_x)-3):
                        if i%5 == 0:
                            
                            pts_x = skel_pts_x[i-3: i+3]
                            pts_y = skel_pts_y[i-3: i+3]

                            x = skel_pts_x[i]
                            y = skel_pts_y[i]

                            z = np.poly1d(np.polyfit(pts_x,pts_y,2))

                            slope = x * (2 * z.c[0]) + z.c[1]
                            norm = (-1) / slope
                            
                            # Point-Slope equation: y- y1 = m(x - x1)
                            edge_y_pred   = norm*(edge_pts_x - x) + y
                            edge_y_offset = np.absolute(edge_pts_y - edge_y_pred)
                            min_idx = np.argmin(edge_y_offset)
                            pair_edge_x, pair_edge_y = edge_pts_x[min_idx], edge_pts_y[min_idx]

                            width = 2 * np.sqrt((pair_edge_x - x)*(pair_edge_x - x) + (pair_edge_y - y)*(pair_edge_y - y)) + 10
                            anlge_of_rect = np.arctan(norm)
                            # Abandon those grasp rects with hugh width
                            if width < 100:
                                rects.append([x, y, width, 30, anlge_of_rect])
                    
                    data_dict[obj_name] = rects

        

                                

                # minDist = 100
                # param1 = 30 #500
                # param2 = 50 #200 #smaller value-> more false circles
                # minRadius = 5
                # maxRadius = 200 #10
                # # Detect circle using HoughCircle
                # rgb_g = cv2.cvtColor(rgb_c, cv2.COLOR_BGR2GRAY)
                # cv2.imwrite("test_gray_{}.png".format(idx), rgb_g)
                # circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
                # print(circles)

                # if circles is not None:
                #     print("detected circle")
                # else:
                #     print("other shape")
        else:
            edge_pts_y, edge_pts_x = np.nonzero(edges)
            skel_pts_y, skel_pts_x = np.nonzero(skeleton)

            rects = []
            for i in range(3, len(skel_pts_x)-3):
                if i%5 == 0:
                    
                    pts_x = skel_pts_x[i-3: i+3]
                    pts_y = skel_pts_y[i-3: i+3]

                    x = skel_pts_x[i]
                    y = skel_pts_y[i]

                    z = np.poly1d(np.polyfit(pts_x,pts_y,2))

                    slope = x * (2 * z.c[0]) + z.c[1]
                    norm = (-1) / slope
                    
                    # Point-Slope equation: y- y1 = m(x - x1)
                    edge_y_pred   = norm*(edge_pts_x - x) + y
                    edge_y_offset = np.absolute(edge_pts_y - edge_y_pred)
                    min_idx = np.argmin(edge_y_offset)
                    pair_edge_x, pair_edge_y = edge_pts_x[min_idx], edge_pts_y[min_idx]

                    width = 2 * np.sqrt((pair_edge_x - x)*(pair_edge_x - x) + (pair_edge_y - y)*(pair_edge_y - y)) + 10
                    anlge_of_rect = np.arctan(norm)

                    # Abandon those grasp rects with hugh width
                    if width < 100:
                        rects.append([x, y, width, 30, anlge_of_rect])
                    
            data_dict[obj_name] = rects


        if save_results:
            for obj in data_dict.keys():
                rects = data_dict[obj]
                for rect in rects:
                    center_x, center_y, width, height, theta = rect
                    box = ((int(center_x), int(center_y)), (width, height), theta / np.pi * 180)
                    box = cv2.boxPoints(box)
                    box = np.int0(box)
                    cv2.drawContours(rgb_c, [box], 0, [255,0,0], 2)

            cv2.imwrite("{}_{}.png".format(prefix, obj_name), rgb_c)
        
    return data_dict



# def grasp_to_robot_frame(camera, grasp, depth, depth_radius, img_width):
#     """
#     return: x, y, z, roll, opening length gripper, object height
#     """
#     pixel_to_meter = 0.365 / img_width
#     camera_pose = np.array([camera.x,camera.y,camera.z])
#     image_rotation = -np.pi * 0.5
#     MAX_GRASP = 100
#     PIX_CONVERSION = 277 * img_width/544
#     DIST_BACKGROUND = 1.115

#     # Get x, y, z of center pixel
#     x_p, y_p = grasp[0], grasp[1]
#     grasp_length = grasp[2]
#     grasp_angle = grasp[4]


#     # Get area of depth values around center pixel
#     x_min = np.clip(x_p-depth_radius, 0, img_width)
#     x_max = np.clip(x_p+depth_radius, 0, img_width)
#     y_min = np.clip(y_p-depth_radius, 0, img_width)
#     y_max = np.clip(y_p+depth_radius, 0, img_width)
#     depth_values = depth[x_min:x_max, y_min:y_max]

#     # Get minimum depth value from selected area
#     z_p = np.amin(depth_values)

#     # Convert pixels to meters
#     x_p *= pixel_to_meter
#     y_p *= pixel_to_meter
#     z_p = camera.far * camera.near / (camera.far - (camera.far - camera.near) * z_p)
    

#     img_center = img_width / 2 - 0.5
#     imgOriginRelative = np.array([-img_center*pixel_to_meter,img_center*pixel_to_meter,0])
#     imgWorldOrigin  = p.multiplyTransforms(camera_pose, p.getQuaternionFromEuler([0,0,0]), imgOriginRelative, p.getQuaternionFromEuler([0,0,image_rotation]))
#     robot_xyz = p.multiplyTransforms(imgWorldOrigin[0],imgWorldOrigin[1], np.array([x_p,y_p,-z_p]), p.getQuaternionFromEuler([0,0,grasp_angle]))

#     yaw = p.getEulerFromQuaternion(robot_xyz[1])[2]

#     # Covert pixel width to gripper width
#     opening_length = (grasp_length / int(MAX_GRASP * PIX_CONVERSION)) * MAX_GRASP

#     obj_height = DIST_BACKGROUND - z_p

#     return robot_xyz[0][0], robot_xyz[0][1], robot_xyz[0][2], yaw, opening_length, obj_height


def grasp_to_robot_frame(camera, grasp, depth, img_width):

    MAX_GRASP = 100
    PIX_CONVERSION = 277 * img_width/544
    DIST_BACKGROUND = 1.115

    center_x, center_y, width, length, theta = grasp


    # create a 4x4 transform matrix that goes from pixel coordinates (and depth values) to world coordinates
    proj_matrix = np.asarray(camera.projection_matrix).reshape([4,4], order="F")
    view_matrix = np.asarray(camera.view_matrix).reshape([4,4], order="F")
    tran_pix_world = np.linalg.inv(np.matmul(proj_matrix, view_matrix))


    # create a grid with pixel coordinates and depth values
    y, x = np.mgrid[-1:1:2 / img_width, -1:1:2 / img_width]
    y *= -1.
    x, y, z = x.reshape(-1), y.reshape(-1), depth.reshape(-1)
    h = np.ones_like(z)

    pixels = np.stack([x, y, z, h], axis=1)
    # filter out "infinite" depths
    pixels = pixels[z < 0.99]
    pixels[:, 2] = 2 * pixels[:, 2] - 1

    points = np.matmul(tran_pix_world, pixels.T).T
    points /= points[:, 3: 4]
    points = points[:, :3]

    grasp_x, grasp_y, grasp_z = points[int(center_y)*img_width+int(center_x),:]

    gripper_width = (width / int(MAX_GRASP * PIX_CONVERSION)) * MAX_GRASP

    obj_height = DIST_BACKGROUND - grasp_z

    # Convert camera to robot angle
    angle = np.asarray([0, 0, theta/180*np.pi])
    angle.shape = (3, 1)
    target_angle = np.dot(tran_pix_world[0:3, 0:3], angle)

    return grasp_x, grasp_y, grasp_z, gripper_width, obj_height, theta, target_angle
    



if __name__ == '__main__':
    from tqdm import tqdm
  
    # env = BaiscEnvironment(GUI = True,robotType ="Panda",img_size= IMG_SIZE)
    env = DatasetEnvironment(GUI = True,robotType ="Panda",img_size = 544)
    # env.createTempBox(0.35, 1)
    # env.createTempBox(0.2, 1)
    env.updateBackgroundImage(1)
    env.dummySimulationSteps(500)
    
    pbar = tqdm(range(100))

    gc = GraspControl(env)

    # Data Collection
    # 1. Put objects in the table
    # 2. Get the semantic mask of each object
    # 3. For each object, erode its mask and get central part of object
    # 4. Randomly generate grasp candidates for each object according to the eroded mask
    # 5. Conduct grasp experiments based on the generated grasp candidates
    # 6. Keep all valid grasp candidates and refine them(position, angle and gripper width)

    for idx in pbar:
        env.objects.shuffle_objects()
        if not os.path.exists("./dataset/{:04d}".format(idx)):
            os.makedirs("./dataset/{:04d}".format(idx))
        # for num in range(1, 4):
        print("Current load {} objects".format(1))
        info = env.objects.get_n_first_obj_info(1)
        # info = [('objects/ycb_objects/YcbBanana/model.urdf', False, False),]
        obj_names = [obj[0].split("/")[2] for obj in info]

        env.createPile(info)
        obj_info = zip(obj_names, env.obj_ids)

        rgb, depth, seg = env.captureImage(removeBackground=0, getSegment=True)

        data_dict = generate_grasp_candidates(rgb, seg, obj_info=obj_info, prefix="{}".format(idx), save_results=True)


        for obj in data_dict.keys():
            print("Current attempt: {}, {} grasp candidates in total".format(obj, len(data_dict[obj])))
            grasp_rects = data_dict[obj]
            valid_rects = []
            for rect in grasp_rects:
                # x, y, z, yaw, opening_len, obj_height = grasp_to_robot_frame(
                #     env.camera, rect, depth, depth_radius=2, img_width=544
                # )
                grasp_x, grasp_y, grasp_z, gripper_width, obj_height, theta, target_angle = grasp_to_robot_frame(env.camera, rect, depth, 544)
                # print(x, y, z, yaw, opening_len, obj_height)

                grasp_flag = gc.graspAttempt([grasp_x, grasp_y, grasp_z, gripper_width, obj_height, theta, target_angle])
                print(grasp_flag)
                if grasp_flag:
                    valid_rects.append(rect)

                env.reset_all_obj()
        
        env.removeAllObject()
                
                
        env.removeAllObject()
        time.sleep(1)
            
            # print("Get objects info: ")
            # with open("./dataset/{:04d}/{}_objects_info.txt".format(idx, num), 'w') as f:
            #     for obj in obj_info:
            #         data = "{}:{}\n".format(obj[0], obj[1])
            #         f.write(data) 

            # print("Writing images (rgb, depth, seg)")
            # rgb, depth, seg = env.captureImage(removeBackground=0, getSegment=True)

            # # grasps, save_name = gg.predict(rgb, depth, n_grasps=20, show_output=True, min_distance=20, threshold_abs=0.5)
            # print(rgb.shape, depth.shape, seg.shape)
            # depth = (depth * 1000).astype(np.uint8)
            # cv2.imwrite("./dataset/{:04d}/{}_objects_rgb.png".format(idx, num), rgb)
            # cv2.imwrite("./dataset/{:04d}/{}_objects_depth.png".format(idx, num), depth)
            # cv2.imwrite("./dataset/{:04d}/{}_objects_seg.png".format(idx, num), seg)

            # # print("Write predicted grasps")
            # # with open("./dataset/{:04d}/{}_objects_grasp.txt".format(idx, num), 'w') as f:
            # #     for grasp in grasps:
            # #         data = "{},{},{},{},{}\n".format(grasp.center[0], grasp.center[1], grasp.angle, grasp.length, grasp.quality)
            # #         f.write(data)

            # env.removeAllObject()
            # time.sleep(1)
  

