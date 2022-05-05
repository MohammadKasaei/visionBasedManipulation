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

    def __init__(self,env,network_model):

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

        self.gg = GraspGenerator(network_path, self.env.camera, self.depth_radius, self.env.camera.width, network_model)
        self.EnableGraspingProcess = False    

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

    def selectBestGraspPoint(self,grasps):
        distToCenter = 1000
        zmax = 0
        for g in grasps:
            x, y, z, roll, opening_len, obj_height = g
            if (zmax < z):
                zmax = z
                grasp = g 
        
            # d = np.linalg.norm(np.array([x,y])-np.array([0.05, -0.52]))
            # if (d < distToCenter):
            #     distToCenter = d
            #     grasp = g 
        
        return grasp

    def getTimeMs(self):
        return round(time.time()*1000)
         
    def getTargetOrientation(self):
        
        if self.env.robotType == "UR5":
            # return   p.getQuaternionFromEuler([self.gOrn, np.pi/2, 0.0]) 
            return   p.getQuaternionFromEuler([0, np.pi/2, self.gOrn-np.pi/2]) 
        elif self.env.robotType == "Panda":
            return   p.getQuaternionFromEuler([-np.pi, 0, self.gOrn-np.pi/2]) 
            
        
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
                    self.updateState("WaitForGraspCandidates")
                else:
                    print ("Accomplished...")
                    self.env.creatPileofTube(1)
                    # self.updateState("Stop")
               else:
                self.cnt += 1
            else:
                self.cnt = 0

        elif self.gState  == "GraspDetection":
            
            rgb ,depth = self.env.captureImage(1)
            number_of_predict = 1
            output = False
            grasps, save_name = self.gg.predict_grasp( rgb, depth, n_grasps=number_of_predict, show_output=output)
            print(grasps)
            if (grasps == []):
                print ("can not predict any grasp point")
                self.cnt+=1
                if self.cnt > 3:
                    self.updateState("GoHome")
            else:
                env.visualizePredictedGrasp(grasps,color=[1,1,0],visibleTime=1)
                grasp = self.selectBestGraspPoint(grasps)

                x, y, z, yaw, opening_len, obj_height = grasp 
                self.gPos = [x, y, np.clip(z+self.env.finger_length, *self.env.ee_position_limit[2])]

                self.gOrn = yaw

            

def generate_grasp_candidates(rgb, seg, obj_info, prefix, save_results=False):
    data_dict = dict()
    for (obj_name, obj_idx) in obj_info:
        rgb_c = np.copy(rgb)

        mask = (seg == obj_idx).astype('uint8')

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        edges = cv2.Canny(mask*255, 100, 200)
        skeleton = (skeletonize(mask, method='lee')).astype('uint8')

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
                anlge_of_rect = np.arctan(norm) / np.pi * 180

                # Abandon those grasp rects with hugh width
                if width < 100:
                    rects.append([x, y, anlge_of_rect, width, 10])
        
        data_dict[obj_name] = rects

        if save_results:
            for rect in rects:
                center_x, center_y, theta, width, height = rect
                box = ((int(center_x), int(center_y)), (width, height), theta)
                box = cv2.boxPoints(box)
                box = np.int0(box)
                cv2.drawContours(rgb_c, [box], 0, [255,0,0], 2)

            cv2.imwrite("{}_{}.png".format(prefix, obj_name), rgb_c)

    
    return data_dict



def grasp_to_robot_frame(camera, grasp, depth, depth_radius, img_width):
    """
    return: x, y, z, roll, opening length gripper, object height
    """
    pixel_to_meter = 0.365 / img_width
    camera_pose = np.array([camera.x,camera.y,camera.z])
    image_rotation = -np.pi * 0.5
    MAX_GRASP = 100
    PIX_CONVERSION = 277 * img_width/544
    DIST_BACKGROUND = 1.115

    # Get x, y, z of center pixel
    x_p, y_p = grasp[0], grasp[1]
    grasp_angle = grasp[2]
    grasp_length = grasp[3]

    # Get area of depth values around center pixel
    x_min = np.clip(x_p-depth_radius, 0, img_width)
    x_max = np.clip(x_p+depth_radius, 0, img_width)
    y_min = np.clip(y_p-depth_radius, 0, img_width)
    y_max = np.clip(y_p+depth_radius, 0, img_width)
    depth_values = depth[x_min:x_max, y_min:y_max]

    # Get minimum depth value from selected area
    z_p = np.amin(depth_values)

    # Convert pixels to meters
    x_p *= pixel_to_meter
    y_p *= pixel_to_meter
    z_p = camera.far * camera.near / (camera.far - (camera.far - camera.near) * z_p)
    

    img_center = img_width / 2 - 0.5
    imgOriginRelative = np.array([-img_center*pixel_to_meter,img_center*pixel_to_meter,0])
    imgWorldOrigin  = p.multiplyTransforms(camera_pose, p.getQuaternionFromEuler([0,0,0]), imgOriginRelative, p.getQuaternionFromEuler([0,0,image_rotation]))
    robot_xyz = p.multiplyTransforms(imgWorldOrigin[0],imgWorldOrigin[1], np.array([x_p,y_p,-z_p]), p.getQuaternionFromEuler([0,0,grasp_angle]))

    yaw = p.getEulerFromQuaternion(robot_xyz[1])[2]

    # Covert pixel width to gripper width
    opening_length = (grasp_length / int(MAX_GRASP * PIX_CONVERSION)) * MAX_GRASP

    obj_height = DIST_BACKGROUND - z_p

    return robot_xyz[0][0], robot_xyz[0][1], robot_xyz[0][2], yaw, opening_length, obj_height



if __name__ == '__main__':
    from tqdm import tqdm

    networkName = "GR_ConvNet"
    if (networkName == "GGCNN"):
            ##### GGCNN #####
            network_model = "GGCNN"
            IMG_SIZE = 300
            network_path = 'trained_models/GGCNN/ggcnn_weights_cornell/ggcnn_epoch_23_cornell'
            sys.path.append('trainconda env export > environment.ymled_models/GGCNN')
    elif (networkName == "GR_ConvNet"):
            ##### GR-ConvNet #####
            network_model = "GR_ConvNet"
            IMG_SIZE = 224
            network_path = 'trained_models/GR_ConvNet/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98'
            sys.path.append('trained_models/GR_ConvNet')
  
    # env = BaiscEnvironment(GUI = True,robotType ="Panda",img_size= IMG_SIZE)
    env = DatasetEnvironment(GUI = True,robotType ="UR5",img_size= 544)
    # env.createTempBox(0.35, 1)
    # env.createTempBox(0.2, 1)
    env.updateBackgroundImage(1)
    gg = GraspGenerator(network_path, env.camera, 2, env.camera.width, network_model)

    env.dummySimulationSteps(500)
    
    pbar = tqdm(range(100))

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
        for num in range(1, 4):
            print("Current load {} objects".format(num))
            # info = env.objects.get_n_first_obj_info(num)
            info = [('objects/ycb_objects/YcbBanana/model.urdf', False, False), ('objects/ycb_objects/YcbScissors/model.urdf', False, False)]
            obj_names = [obj[0].split("/")[2] for obj in info]

            env.createPile(info)
            obj_info = zip(obj_names, env.obj_ids)

            rgb, depth, seg = env.captureImage(removeBackground=0, getSegment=True)

            data_dict = generate_grasp_candidates(rgb, seg, obj_info=obj_info, prefix="{}_{}".format(idx, num), save_results=True)


            for obj in data_dict.keys():
                print("Current attempt: {}, {} grasp candidates in total".format(obj, len(data_dict[obj])))
                grasp_rects = data_dict[obj]
                for rect in grasp_rects:
                    x, y, z, yaw, opening_len, obj_height = grasp_to_robot_frame(
                        env.camera, rect, depth, depth_radius=2, img_width=544
                    )
                    print(x, y, z, yaw, opening_len, obj_height)
                    env.visualizePredictedGrasp([[x, y, z, yaw, opening_len, obj_height]])

                    # Execute the predicted grasp and reset object
            
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
  

