import time

from cv2 import getTrackbarPos
from environment.basicEnv import BaiscEnvironment
from environment.utilities import Camera
from graspGenerator.grasp_generator import GraspGenerator

import pybullet as p
import numpy as np
import sys

class GraspControl():

    def __init__(self,env):
        self.env = env
        self.GRIPPER_MOVING_HEIGHT = 1.25
        self.GRIP_REDUCTION = 0.3
        self.gState = "Stop"
        self.gPos = np.array([0,0,0])
        self.gOrn = np.array([0,np.pi/2,0])
        self.gWidth = 0.1
        self.gg = GraspGenerator(network_path, env.camera, depth_radius, IMG_SIZE, network_model)
            


    def graspStateMachine(self):
                        
        if  self.gState  == "Stop":
            eeState   = self.env.getEEState()
            self.env.moveGripper(0.1)
            self.env.moveEE(eeState[0], eeState[1],max_step=1)
            
        elif self.gState  == "GoHome":
            targetPos    = self.env.TARGET_ZONE_POS
            targetPos[2] = self.GRIPPER_MOVING_HEIGHT
            targetOrn    = p.getQuaternionFromEuler([-np.pi*0.25, np.pi/2, 0.0])
            self.env.moveEE(targetPos, targetOrn)
            if (self.env.isThereAnyObject()):
                self.gState = "GraspDetection"
            else:
                self.gState = "Stop"
                
        elif self.gState  == "GraspDetection":
            
            rgb ,depth = env.captureImage(1)
            number_of_predict = 3
            output = True
            grasps, save_name = self.gg.predict_grasp( rgb, depth, n_grasps=number_of_predict, show_output=output)
            env.visualizePredictedGrasp(grasps,color=[1,1,0],visibleTime=1)
            
        
        elif self.gState  == "MoveOnTopofBox":
            targetPos    = np.array(self.gPos[:])
            targetPos[2] = self.GRIPPER_MOVING_HEIGHT
            self.env.moveGripper(0.1)
            targetOrn    = p.getQuaternionFromEuler([self.gOrn[0], np.pi/2, 0.0])
            self.env.moveEE(targetPos, targetOrn)
            eeState   = self.env.getEEState()
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
            eeState   = self.env.getEEState()
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
            self.env.moveGripper(0.01)
            time.sleep(0.025)
                # If the object has been grasped and lifted off the table
            grasped_id = self.env.checkGraspedID()
            if len(grasped_id) == 1:
                succes_grasp = True
                grasped_obj_id = grasped_id[0]
            # else:
            #     return succes_target, succes_grasp

            if succes_grasp:
                self.cnt += 1
                if self.cnt>30:
                    self.gState  = "BringObject"   
            else:
                self.cnt = 0
        
        elif self.gState  == "BringObject":

            targetPos    = np.array([self.gPos[0],self.gPos[1], 1.25])
            targetOrn    = p.getQuaternionFromEuler([self.gOrn[0], np.pi/2, 0.0])
            self.env.moveEE(targetPos, targetOrn)
            eeState   = self.env.getEEState()
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
            z = self.env.TARGET_ZONE_POS[2] + obj_height + 0.5
            targetPos    = np.array([self.env.TARGET_ZONE_POS[0],self.env.TARGET_ZONE_POS[1],z])
            targetOrn    = p.getQuaternionFromEuler([self.gOrn[0], np.pi/2, 0.0])
            self.env.moveEE(targetPos, targetOrn)
            eeState   = self.env.getEEState()
            dist = np.linalg.norm(targetPos-eeState[0])
            print(dist)
            if (dist<0.15):
                self.gState  = "DropObject"
                self.cnt = 0
            
        elif self.gState  == "DropObject":
            z = self.env.TARGET_ZONE_POS[2] + obj_height 
            targetPos    = np.array([self.env.TARGET_ZONE_POS[0],self.env.TARGET_ZONE_POS[1],z])
            targetOrn    = p.getQuaternionFromEuler([self.gOrn[0], np.pi/2, 0.0])
            self.env.moveEE(targetPos, targetOrn)
            eeState   = self.env.getEEState()
            dist = np.linalg.norm(targetPos-eeState[0])
            print(dist)
            if (dist<0.2):
                if (self.cnt>10):
                    self.env.moveGripper(0.1)
                    self.gState  = "GoHome"   
                    self.cnt = 0
                else:
                    self.cnt += 1
            else:
                self.cnt = 0
            
        

        

                
                z_drop = self.env.TARGET_ZONE_POS[2] + obj_height + 0.5
                targetPos    = np.array([self.env.TARGET_ZONE_POS[0],self.env.TARGET_ZONE_POS[1],z_drop])
                targetOrn    = p.getQuaternionFromEuler([self.gOrn[0], np.pi/2, 0.0])
                self.env.moveEE(targetPos, targetOrn)
                    

    def grasp(self, pos: tuple, roll: float, gripper_opening_length: float, obj_height: float):
        """
        Method to perform grasp
        pos [x y z]: The axis in real-world coordinate
        roll: float,   for grasp, it should be in [-pi/2, pi/2)
        """
        succes_grasp, succes_target = False, False
        grasped_obj_id = None

        x, y, z = pos
    
        # Substracht gripper finger length from z
        z += self.env.finger_length
        z  = np.clip(z, *self.env.ee_position_limit[2])

        self.env.moveGripper(0.1)
        orn = p.getQuaternionFromEuler([roll, np.pi/2, 0.0])
        self.env.moveEE([x, y, self.GRIPPER_MOVING_HEIGHT, orn])

        # Reduce grip to get a tighter grip
        # gripper_opening_length *= self.GRIP_REDUCTION
        gripper_opening_length = 0.02
        # Grasp and lift object
        z_offset = 0.#self.calc_z_offset(gripper_opening_length)
        self.env.moveEE([x, y, z + z_offset, orn])
        self.env.moveGripper(gripper_opening_length)
        # self.auto_close_gripper(check_contact=True)
        # self.move_gripper(0.02)
        for _ in range(50):
            self.env.stepSimulation()
        
        self.env.moveEE([x, y, self.GRIPPER_MOVING_HEIGHT, orn])

        # If the object has been grasped and lifted off the table
        grasped_id = self.env.checkGraspedID()
        if len(grasped_id) == 1:
            succes_grasp = True
            grasped_obj_id = grasped_id[0]
        else:
            return succes_target, succes_grasp

        # Move object to target zone
        y_drop = self.env.TARGET_ZONE_POS[2] + z_offset + obj_height + 0.15
        y_orn  = p.getQuaternionFromEuler([-np.pi*0.25, np.pi/2, 0.0])

        #self.env.move_arm_away()
        self.env.moveEE([self.env.TARGET_ZONE_POS[0],
                        self.env.TARGET_ZONE_POS[1], 1.25, y_orn])
        self.env.moveEE([self.env.TARGET_ZONE_POS[0],
                        self.env.TARGET_ZONE_POS[1], y_drop, y_orn])
        self.env.moveGripper(0.085)
        self.env.moveEE([self.env.TARGET_ZONE_POS[0], self.env.TARGET_ZONE_POS[1],
                        self.env.GRIPPER_MOVING_HEIGHT, y_orn])

        # Wait then check if object is in target zone
        for _ in range(80):
            self.env.stepSimulation()
        if self.env.IsTargetReached(grasped_obj_id):
            succes_target = True
            # env.remove_obj(grasped_obj_id)
        
        # succes_target = True
        return succes_grasp, succes_target
    

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
    
    gc = GraspControl(env)
    x, y, z, roll, opening_len, obj_height = grasps[0]
  
    # gc.grasp([x, y, z], yaw, opening_len, obj_height)
    gc.gState = "Stop"
    gc.graspStateMachine()
    env.dummySimulationSteps(100)   
    
         
    # gc.gPos = np.array([x, y, np.clip(z+env.finger_length, *env.ee_position_limit[2])])
    # gc.gOrn = np.array([roll,0,0])
    gc.gPos = [x, y, np.clip(z+env.finger_length, *env.ee_position_limit[2])]
    gc.gOrn = [roll,0,0]
    gc.gState  = "MoveOnTopofBox" 
    
    for _ in range(50000):
        gc.graspStateMachine()
    
    # gc.gPos = np.array([x, y, np.clip(z+env.finger_length, *env.ee_position_limit[2])])
    # gc.gOrn = np.array([roll,0,0])
    # gc.gState  = "Grasp"     
    # for _ in range(1000):
    #     gc.graspStateMachine()
  