import time

from cv2 import getTrackbarPos
from environment.basicEnv import BaiscEnvironment
from environment.camera.camera import Camera
from graspGenerator.grasp_generator import GraspGenerator

import pybullet as p
import numpy as np
import sys

class GraspControl():

    def __init__(self,env,network_model):

        self.env = env
        if self.env.robotType == "UR5":
            self.GRIPPER_MOVING_HEIGHT = 1.2
            self.GRIP_REDUCTION = 0.3
            self.GRIPPER_INIT_ORN = [-np.pi*0.25, np.pi/2, 0.0]
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

        self.gg = GraspGenerator(network_path, self.env.camera, self.depth_radius, self.env.camera.width, network_model)
        self.EnableGraspingProcess = False    

        self.TIMEOUT_MS = {
            "Stop": 100,
            "GoHome": 20000,
            "GraspDetection": 2000,
            "MoveOnTopofBox": 10000,
            "ReadyToGrasp": 10000,
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
        4- visualize gripper 

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

    def selectBestGraspPoint(self,grasps):
        distToCenter = 1000
        for g in grasps:
            x, y, z, roll, opening_len, obj_height = g
            d = np.linalg.norm(np.array([x,y])-np.array([0.05, -0.52]))
            if (d < distToCenter):
                distToCenter = d
                grasp = g 
        
        return grasp

    def getTimeMs(self):
        return round(time.time()*1000)
         
    def getTargetOrientation(self):
        
        if self.env.robotType == "UR5":
            return   p.getQuaternionFromEuler([self.gOrn[0], np.pi/2, 0.0]) 
        elif self.env.robotType == "Panda":
            return   p.getQuaternionFromEuler([-np.pi, 0, self.gOrn[0]]) 
            
        



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
                    self.updateState("GraspDetection")
                else:
                    print ("Accomplished...")
                    self.updateState("Stop")
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
                    self.updateState("GoHome")
            else:
                env.visualizePredictedGrasp(grasps,color=[1,1,0],visibleTime=0.1)
                grasp = self.selectBestGraspPoint(grasps)

                x, y, z, roll, opening_len, obj_height = grasp 
                self.gPos = [x, y, np.clip(z+self.env.finger_length, *self.env.ee_position_limit[2])]

                self.gOrn = [roll,0,0]
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
  
    env = BaiscEnvironment(GUI = True,robotType ="Panda",img_size= IMG_SIZE)
    env.createTempBox(0.35, 1)
    env.creatPileofTube(30)
    env.dummySimulationSteps(500)
    
    gc = GraspControl(env,network_model) 
    gc.updateState("GoHome")
    gc.EnableGraspingProcess = True
    for _ in range(50000):
        gc.graspStateMachine()
    
  