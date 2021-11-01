import time

from cv2 import getTrackbarPos
from environment.basicEnv import BaiscEnvironment
from environment.utilities import Camera
from graspGenerator.grasp_generator import GraspGenerator

import pybullet as p
import numpy as np
import sys

class GraspControl():

    def __init__(self,env,network_model):
        self.env = env
        self.GRIPPER_MOVING_HEIGHT = 1.2
        self.GRIP_REDUCTION = 0.3
        self.gState = "Stop"
        self.gPos = np.array([0,0,0])
        self.gOrn = np.array([0,np.pi/2,0])
        self.gWidth = 0.1
        self.gg = GraspGenerator(network_path, self.env.camera, depth_radius, self.env.camera.width, network_model)
            
    """
     TODO:
        1- state timeout
        2- rank grasp points: 
            select the best one (x,y,z)
            fail handeler:
                select another gp

        3- add a function for updating state 
         
    """

    def graspStateMachine(self):
        eeState   = self.env.getEEState()

        if  self.gState  == "Stop":
            self.env.moveGripper(0.1)
            self.env.moveEE(eeState[0], eeState[1],max_step=1)
            
        elif self.gState  == "GoHome":
            targetPos    = self.env.TARGET_ZONE_POS[:]
            targetPos[2] = self.GRIPPER_MOVING_HEIGHT
            targetOrn    = p.getQuaternionFromEuler([-np.pi*0.25, np.pi/2, 0.0])
            self.env.moveEE(targetPos, targetOrn)
            dist = np.linalg.norm(np.array(targetPos)-eeState[0])
            if (dist<0.1):
               if (self.cnt>10):
                if (self.env.isThereAnyObject()):
                    self.gState = "GraspDetection"
                else:
                    print ("Accomplished...")
                    self.gState = "Stop"
                self.cnt = 0
               else:
                self.cnt += 1
            else:
                self.cnt = 0


                
        elif self.gState  == "GraspDetection":
            
            rgb ,depth = self.env.captureImage(1)
            number_of_predict = 3
            output = False
            grasps, save_name = self.gg.predict_grasp( rgb, depth, n_grasps=number_of_predict, show_output=output)
            if (grasps == []):
                print ("can not predict any grasp point")
                self.cnt+=1
                if self.cnt > 3:
                    self.gState  = "GoHome"
                    self.cnt = 0
                    
            else:
                env.visualizePredictedGrasp(grasps,color=[1,1,0],visibleTime=0.1)
                x, y, z, roll, opening_len, obj_height = grasps[0]
                self.gPos = [x, y, np.clip(z+self.env.finger_length, *self.env.ee_position_limit[2])]
                self.gOrn = [roll,0,0]
                self.gState  = "MoveOnTopofBox"
                self.cnt = 0

        
        elif self.gState  == "MoveOnTopofBox":
            targetPos    = np.array(self.gPos[:])
            targetPos[2] = self.GRIPPER_MOVING_HEIGHT
            self.env.moveGripper(0.1)
            targetOrn    = p.getQuaternionFromEuler([self.gOrn[0], np.pi/2, 0.0])
            self.env.moveEE(targetPos, targetOrn)
            dist = np.linalg.norm(np.array(targetPos)-eeState[0])
            if (dist<0.05):
               if (self.cnt>10):
                self.gState  = "ReadyToGrasp"
                self.cnt = 0
               else:
                self.cnt += 1
            else:
                self.cnt = 0

                
            print (f"{dist:2.3}")

        elif self.gState  == "ReadyToGrasp":
            targetPos    = np.array(self.gPos[:])
            targetOrn    = p.getQuaternionFromEuler([self.gOrn[0], np.pi/2, 0.0])
            self.env.moveEE(targetPos, targetOrn)
            dist = np.linalg.norm(targetPos-eeState[0])
            print(dist)
            if (dist<0.14):
               if (self.cnt>20):
                self.gState  = "Grasp"
                self.cnt = 0
               else:
                self.cnt += 1
            else:
                self.cnt = 0



        elif self.gState  == "Grasp":

            succes_grasp = False
            self.env.moveGripper(0.024)
            time.sleep(0.025)
                # If the object has been grasped and lifted off the table
            grasped_id = self.env.checkGraspedID()
            if len(grasped_id) >= 1:
                succes_grasp = True
                grasped_obj_id = grasped_id[0]
           
            if succes_grasp:
                self.cnt += 1
                if self.cnt>30:
                    self.gState  = "Pickup"   
            else:
                self.cnt = 0
        
        elif self.gState  == "Pickup":

            targetPos    = np.array([self.gPos[0],self.gPos[1], self.GRIPPER_MOVING_HEIGHT])
            targetOrn    = p.getQuaternionFromEuler([self.gOrn[0], np.pi/2, 0.0])
            self.env.moveEE(targetPos, targetOrn)
            dist = np.linalg.norm(targetPos-eeState[0])
            print(dist)
            if (dist<0.14):
               if (self.cnt>10):
                self.gState  = "MoveObjectIntoTarget"   
                self.cnt = 0
               else:
                self.cnt += 1
            else:
                self.cnt = 0
            
        elif self.gState  == "MoveObjectIntoTarget":
            z = self.GRIPPER_MOVING_HEIGHT
            targetPos    = np.array([self.env.TARGET_ZONE_POS[0],self.env.TARGET_ZONE_POS[1],z])
            targetOrn    = p.getQuaternionFromEuler([self.gOrn[0], np.pi/2, 0.0])
            self.env.moveEE(targetPos, targetOrn)
            dist = np.linalg.norm(targetPos-eeState[0])
            print(dist)
            if (dist<0.15):
                self.gState  = "DropObject"
                self.cnt = 0
            
        elif self.gState  == "DropObject":
            z = self.env.TARGET_ZONE_POS[2]+0.2  
            targetPos    = np.array([self.env.TARGET_ZONE_POS[0],self.env.TARGET_ZONE_POS[1],z])
            targetOrn    = p.getQuaternionFromEuler([self.gOrn[0], np.pi/2, 0.0])
            self.env.moveEE(targetPos, targetOrn)
            dist = np.linalg.norm(targetPos-eeState[0])
            print(dist)
            if (dist<0.15):
                if (self.cnt>10):
                    self.env.moveGripper(0.1)
                    self.gState  = "GoHome"   
                    self.cnt = 0
                else:
                    self.cnt += 1
            else:
                self.cnt = 0
            
        

if __name__ == '__main__':

    networkName = "GR_ConvNet"
    if (networkName == "GGCNN"):
            ##### GGCNN #####
            network_model = "GGCNN"
            IMG_SIZE = 300
            network_path = 'trained_models/GGCNN/ggcnn_weights_cornell/ggcnn_epoch_23_cornell'
            sys.path.append('trained_models/GGCNN')
    elif (networkName == "GR_ConvNet"):
            ##### GR-ConvNet #####
            network_model = "GR_ConvNet"
            IMG_SIZE = 224
            network_path = 'trained_models/GR_ConvNet/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98'
            sys.path.append('trained_models/GR_ConvNet')
    depth_radius = 2

    env = BaiscEnvironment(GUI = True,img_size= IMG_SIZE)
    env.createTempBox(0.45, 1)
    env.creatPileofTube(10)
    env.dummySimulationSteps(300)
    
    # gg = GraspGenerator(network_path, env.camera, depth_radius, IMG_SIZE, network_model)
    # rgb ,depth = env.captureImage(1)
    # number_of_predict = 3
    # output = True
    # grasps, save_name = gg.predict_grasp( rgb, depth, n_grasps=number_of_predict, show_output=output)
    # env.visualizePredictedGrasp(grasps,color=[1,1,0],visibleTime=1)
    
    gc = GraspControl(env,network_model)
    # x, y, z, roll, opening_len, obj_height = grasps[0]
  
    # # gc.grasp([x, y, z], yaw, opening_len, obj_height)
    # gc.gState = "Stop"
    # gc.graspStateMachine()
    # env.dummySimulationSteps(100)   
    
         
    # gc.gPos = np.array([x, y, np.clip(z+env.finger_length, *env.ee_position_limit[2])])
    # gc.gOrn = np.array([roll,0,0])
    # gc.gPos = [x, y, np.clip(z+env.finger_length, *env.ee_position_limit[2])]
    # gc.gOrn = [roll,0,0]
    # gc.gState  = "MoveOnTopofBox" 
    gc.gState = "GoHome"
    for _ in range(50000):
        gc.graspStateMachine()
    
    # gc.gPos = np.array([x, y, np.clip(z+env.finger_length, *env.ee_position_limit[2])])
    # gc.gOrn = np.array([roll,0,0])
    # gc.gState  = "Grasp"     
    # for _ in range(1000):
    #     gc.graspStateMachine()
  