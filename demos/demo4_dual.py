from numpy.core.numeric import roll
from torch._C import device
from grasp_generator import GraspGenerator
from environment.utilities import Camera
from environment.envDual import Environment
from utils import YcbObjects, PackPileData, IsolatedObjData, summarize
import pybullet as p
import argparse
import os
import sys
import cv2
import math
import matplotlib.pyplot as plt
import time
import random
import numpy as np


class GrasppingScenarios():

    def __init__(self,networkName="GR_ConvNet"):
        
        if (networkName == "GGCNN"):
            ##### GGCNN #####
            self.network_model = "GGCNN"
            self.IMG_SIZE = 300
            self.network_path = 'trained_models/GGCNN/ggcnn_weights_cornell/ggcnn_epoch_23_cornell'
            sys.path.append('trained_models/GGCNN')
        elif (networkName == "GR_ConvNet"):
            ##### GR-ConvNet #####
            self.network_model = "GR_ConvNet"
            self.IMG_SIZE = 224
            self.network_path = 'trained_models/GR_ConvNet/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98'
            sys.path.append('trained_models/GR_ConvNet')
        
        
        self.CAM_Z = 1.9
        self.depth_radius = 2
        self.ATTEMPTS = 5
        self.fig = plt.figure(figsize=(10, 10))
       
            
    def is_there_any_object(self,camera):
        self.dummy_simulation_steps(10)
        rgb, depth, _ = camera.get_cam_img()
        #print ("min RGB = ", rgb.min(), "max RGB = ", rgb.max(), "rgb.avg() = ", np.average(rgb))
        print ("min depth = ", depth.min(), "max depth = ", depth.max())
        if (depth.max()- depth.min() < 0.005):
            return False
        else:
            return True
                    
    
    def draw_predicted_grasp(self,grasps,color = [0,0,1],lineIDs = []):
        x, y, z, yaw, opening_len, obj_height = grasps

        gripper_size = opening_len + 0.02 
        finger_size = 0.075
        # lineIDs = []
        lineIDs.append(p.addUserDebugLine([x, y, z], [x, y, z+0.15],color, lineWidth=3))
        
        return lineIDs
    
    def remove_drawing(self,lineIDs):
        for line in lineIDs:
            p.removeUserDebugItem(line)
    
    def dummy_simulation_steps(self,n):
        for _ in range(n):
            p.stepSimulation()


    def creatPileofTube(self,n):

        obj_init_pos = [0.05, -0.52]
        Z_TABLE_TOP  = 0.785
        radius   = 0.02
        length   = 0.15
        mass     = 0.025 #kg
        tubeObj  = []
        color    = [0.4,0.4,0.4,1]
        halfsize = [0.02, 0.02,0.075]
        self.tubeObj = []

        for i in range(n):
            r_x    = random.uniform(obj_init_pos[0] - 0.1, obj_init_pos[0] + 0.1)
            r_y    = random.uniform(obj_init_pos[1] - 0.1, obj_init_pos[1] + 0.1)
            roll   = random.uniform(0, np.pi)
            orn    = p.getQuaternionFromEuler([roll, 0, 0])
            pos    = [r_x, r_y, 1]
            
            obj_id = p.loadURDF("objects/ycb_objects/YcbTomatoSoupCan/model.urdf", pos, orn)
    
            # box    = p.createCollisionShape(p.GEOM_BOX, halfExtents=halfsize)
            # vis    = p.createVisualShape(p.GEOM_BOX, halfExtents=halfsize, rgbaColor=color, specularColor=[1,1,1])
            # obj_id = p.createMultiBody(mass, box, vis, pos, orn)
    
            # Tube  = p.createCollisionShape(p.GEOM_CYLINDER, radius = radius, height=length)
            # vis = p.createVisualShape(p.GEOM_CYLINDER, radius = radius,length=length, rgbaColor=color, specularColor=[1,1,1])
            # obj_id = p.createMultiBody(mass, Tube, vis, pos, orn, [0,0,0,1])
    
            for _ in range(100):
                p.stepSimulation()
    
            p.changeDynamics(obj_id, 
                            -1,
                            spinningFriction = 0.8,
                            rollingFriction  = 0.8,
                            linearDamping    = 0.1)
            self.tubeObj.append(obj_id)
        
        for _ in range(1000):
            p.stepSimulation()
        # time.sleep(0.1)
    

    def pileOfTubesScenario(self,runs, device, vis, output, debug):
                
        number_of_objects  = 10
        number_of_attempts = 5
        number_of_predict  = 3
        
        center_x, center_y, center_z = 0.05, -0.52, self.CAM_Z
        camera    = Camera((center_x, center_y, center_z), (center_x, center_y, 0.785), 0.2, 2.0, (self.IMG_SIZE, self.IMG_SIZE), 40)
        env       = Environment(camera, vis=vis, debug=debug, finger_length=0.11)
        generator = GraspGenerator(self.network_path, camera, self.depth_radius, self.fig, self.IMG_SIZE, self.network_model, device)

        bgBgrWithoutBox ,bgDepthWithoutBox, _ = camera.get_cam_img()
        

        timestr = time.strftime("%Y%m%d-%H%M%S")
        fName = "resultstxt/pileofTubes_" + timestr+".csv"
        env.create_temp_box(0.45, 1)

        bgBGR ,bgDepthBox, _ = camera.get_cam_img()
       

        for i in range(runs):
            env.remove_all_obj()
            env.reset_robot()              
            print(f"----------- run {i+1}-----------")
            
            if vis:
                debugID = p.addUserDebugText(f'Experiment {i+1}', [-0.0, -0.9, 0.8], [0,0,255], textSize=2)
                time.sleep(0.5)
                p.removeUserDebugItem(debugID)

            number_of_failures = 0
            no_grasp_point = 0
            self.creatPileofTube(number_of_objects)
            env.obj_ids = self.tubeObj
            robotNo = 1
            while self.is_there_any_object(camera) and number_of_failures < number_of_attempts and no_grasp_point<10:                
                #env.move_arm_away()
                # try:
                idx = 0 
                
                for _ in range(number_of_attempts):
                    bgr, depth, _ = camera.get_cam_img()
                    bgr = bgr-bgBGR+bgBgrWithoutBox
                    depth = depth-bgDepthBox+bgDepthWithoutBox

                    if robotNo == 1:
                        env.robot_id = env.robot1_id
                        env.controlGripper = env.controlGripper1
                    else:
                        env.robot_id = env.robot2_id
                        env.controlGripper = env.controlGripper2
                    


                   
                    ##convert BGR to RGB
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    grasps, save_name = generator.predict_grasp( rgb, depth, n_grasps=number_of_predict, show_output=output)
                    if (grasps == []):
                        for _ in range(100):
                            p.stepSimulation()
                        no_grasp_point+=1
                        continue
                    no_grasp_point = 0
                    
                    if (idx>len(grasps)-1):
                        idx = 0
                        
                    if vis:
                        LID =[]
                        for g in grasps:
                            LID = self.draw_predicted_grasp(g,color=[1,0,1],lineIDs=LID)
                        time.sleep(3)
                        self.remove_drawing(LID)
                        self.dummy_simulation_steps(10)
                        
                        lineIDs = self.draw_predicted_grasp(grasps[idx])
                    
                    x, y, z, yaw, opening_len, obj_height = grasps[idx]
                    succes_grasp, succes_target = env.grasp((x, y, z), yaw, opening_len, obj_height)
                    
                    # if succes_grasp:
                    #     data.add_succes_grasp(obj_name)
                    # if succes_target:
                    #     data.add_succes_target(obj_name)

                    ## remove visualized grasp configuration 
                    if vis:
                        self.remove_drawing(lineIDs)

                    with open(fName, "a") as fp:
                        # Append 'hello' at the end of file
                        if (succes_target):
                            fp.write(f"{i+1}\t success\t {1 if (succes_target) else 0}\t {1 if (succes_grasp) else 0}\t {number_of_failures}\n")
                        else:
                            fp.write(f"{i+1}\t failed\t {1 if (succes_target) else 0}\t {1 if (succes_grasp) else 0}\t {number_of_failures}\n")
                            



                    if succes_target:
                        print(f"success\t 1\t {1 if (succes_grasp) else 0}\t {number_of_failures}")
                         
                        number_of_failures = 0
                        if vis:
                            debugID = p.addUserDebugText("success", [-0.0, -0.9, 0.8], [0,0.50,0], textSize=2)
                            time.sleep(0.5)
                            p.removeUserDebugItem(debugID)
                        
                        if save_name is not None:
                            os.rename(save_name + '.png', save_name + f'_SUCCESS_grasp{i}.png')

                        env.go_home_pos(robotNo)
                        robotNo = 1 if robotNo == 2 else 2
                        break

                    else: 
                        number_of_failures += 1
                        idx = idx+1 if (idx<len(grasps)-1) else 0
                        print(f"failed\t 0\t {1 if (succes_grasp) else 0}\t {number_of_failures}")
                        
                        if vis:
                            debugID = p.addUserDebugText("failed", [-0.0, -0.9, 0.8], [0.5,0,0], textSize=2)
                            time.sleep(0.5)
                            p.removeUserDebugItem(debugID)
                            #print (x, y)
                        env.go_home_pos(robotNo)
                        robotNo = 1 if robotNo == 2 else 2




if __name__ == '__main__':
    
    runs        = 10
    device      = 'cpu'
    saveOutput  = False
    GUI         = True
    networkName ="GR_ConvNet" #GGCNN "GR_ConvNet"
    grasp  = GrasppingScenarios(networkName)
    grasp.pileOfTubesScenario(runs,device, vis=GUI, output=saveOutput, debug=False)
