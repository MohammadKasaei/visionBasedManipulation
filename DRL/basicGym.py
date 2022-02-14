import math
import numpy as np
from   numpy.core.defchararray import array
import pybullet as p
import pybullet_data

from   enum import Enum
from   operator import truth
import numpy as np
import gym
from   gym import spaces
from   numpy.core.function_base import linspace

import pybullet_data
from scipy.spatial import distance
from Visualization.InputParamMenu import InputParamMenu
from Visualization.GraphicalMenu import Menu 

import time

from   stable_baselines3.common.env_util import make_vec_env
from   stable_baselines3 import PPO
from   stable_baselines3.common.utils import set_random_seed
from   stable_baselines3.common.vec_env import SubprocVecEnv
from   stable_baselines3.common.vec_env.vec_normalize import VecNormalize
import os
import sys



from environment.basicEnv import BaiscEnvironment
from environment.camera.camera import Camera

from graspGenerator.grasp_generator import GraspGenerator



class CustomEnv(gym.Env):
    
    def __init__(self,id,params):

        super(CustomEnv, self).__init__()
        
        
        if (id==0):
            self.vis = True
        else:
            self.vis = False
        
        
        self.vis = True

        self.inputParams = params

        self.ctrlFreq          = 250.0
        self.NNFreq            = 25.0
        self.TimestepPerAction = int(self.ctrlFreq/self.NNFreq)
        
        self.simTimeStep = 1.0/self.ctrlFreq

        
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
    
        self.env = BaiscEnvironment(GUI = self.vis,robotType ="UR5",img_size= IMG_SIZE)
        self.env.dummySimulationSteps(100)

        
        self.env_id = id
        self.episode_number = 0

        # self.tableOrigin = np.array(p.getBasePositionAndOrientation(self.env.tableID)[0])
     
        action_bound_low = np.array([-1, -1,       0.01, # ee pos
                                      -np.pi, # ee yaw
                                      0 # gripper pos
                                    ])

        action_bound_high = np.array([1,1,0.3, # ee pos
                                      np.pi, # ee yaw
                                      0.2 # gripper pos
                                    ])

                                     
        observation_bound = np.array([np.inf,      np.inf,     np.inf,          # obj pos
                                      np.inf,      np.inf,     np.inf,  np.inf, #obj ori
                                      np.inf,      np.inf,     np.inf,          # ee pos
                                      np.inf,      np.inf,     np.inf,  np.inf, #ee ori                                   
                                      np.inf,      np.inf,     np.inf])         # distEEFromObj, f1contact, f2contact 


        
        self.action_space = spaces.Box(low=action_bound_low, high=action_bound_high, dtype="float32")
        self.observation_space = spaces.Box(low = -observation_bound, high = observation_bound, dtype="float32")
    


    def reset(self):

        self.act = None
        self.simTime = 0

        self.env.resetRobot(gripperType="140")
        self.env.moveGripper(0.4,100)
        self.env.removeAllObject()

        self.objID = self.env.loadIsolatedObj("objects/ycb_objects/YcbBanana/model.urdf")
        

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
  
        self.depth_radius = 2
        self.gg = GraspGenerator(network_path, self.env.camera, self.depth_radius, self.env.camera.width, network_model)
        rgb ,depth = self.env.captureImage(removeBackground=0)
        number_of_predict = 1
        output = False
        grasps, save_name = self.gg.predict_grasp( rgb, depth, n_grasps=number_of_predict, show_output=output)
        if (grasps == []):
                print ("can not predict any grasp point")
                self.cnt+=1
                if self.cnt > 3:
                    self.updateState("GoHome")
        else:
            self.env.visualizePredictedGrasp(grasps,color=[1,0,1],visibleTime=1)
            x, y, z, yaw, opening_len, obj_height = grasps[0]
            self.gPos = [x, y, np.clip(z+self.env.finger_length, *self.env.ee_position_limit[2])]
            self.gOrn = yaw
            self.opening_len = opening_len
            

        observation = self.observe()

       
        return observation   


    def observe(self):
        
        self.ee   = p.getLinkState(self.env.robot_id,self.env.eef_id)
        self.obj  = p.getBasePositionAndOrientation(self.objID)
        
        self.distEEFromObj = np.linalg.norm(np.array(self.ee[0])-np.array(self.obj[0]))
        self.f1contact = 1 if p.getContactPoints(self.objID,self.env.robot_id,-1,self.env.f1_id) != () else 0
        self.f2contact = 1 if p.getContactPoints(self.objID,self.env.robot_id,-1,self.env.f2_id) != () else 0
        
        ob      = np.concatenate([np.array(self.obj[0]),np.array(self.obj[1]),
                                  np.array(self.ee[0]),np.array(self.ee[1]),
                                  np.array([self.distEEFromObj,self.f1contact,self.f2contact])
                                  ])
        
        return ob



    def calcReward(self):
        
        k1 = self.inputParams["Param1"]     
        k2 = self.inputParams["Param2"]     
        
        k3 = self.inputParams["Param3"]     
        
        k4 = self.inputParams["Param4"]     
        k5 = self.inputParams["Param5"]     

        penalty1 = 10 if p.getContactPoints(self.env.tableID,self.env.robot_id,-1,self.env.eef_id) != () else 0
        penalty2 = 10 if p.getContactPoints(self.env.tableID,self.env.robot_id,-1,self.env.f1_id) != () else 0
        penalty3 = 10 if p.getContactPoints(self.env.tableID,self.env.robot_id,-1,self.env.f2_id) != () else 0
        
        penalty4 = 10 if p.getContactPoints(self.env.UR5Stand_id,self.env.robot_id,-1,self.env.eef_id) != () else 0
        penalty5 = 10 if p.getContactPoints(self.env.UR5Stand_id,self.env.robot_id,-1,self.env.f1_id) != () else 0
        penalty6 = 10 if p.getContactPoints(self.env.UR5Stand_id,self.env.robot_id,-1,self.env.f2_id) != () else 0


        self.hitTheGround = penalty1+penalty2+penalty3+penalty4+penalty5+penalty6
        
        if (self.hitTheGround>0):
            print("hit")

        
        return 2-self.distEEFromObj


    def step(self, action):

        if (np.isnan(action).any()):          
            action = self.act

        self.simTime += 1
        
        self.act = action
             
        # self.env.moveGripper(self.act[4],step = self.TimestepPerAction)
        self.env.moveGripper(self.opening_len,step = self.TimestepPerAction)
        self.env.dummySimulationSteps(100)
        # orn = p.getQuaternionFromEuler([-np.pi, np.pi/2,self.act[3]])
        # p.getQuaternionFromEuler([0, np.pi/2, self.gOrn-np.pi/2]) 
        # self.env.moveEE([self.act[0],self.act[1],self.act[2]],orn,max_step=self.TimestepPerAction)    
        orn = p.getQuaternionFromEuler([-0*np.pi, np.pi/2,self.gOrn])
        
        self.env.moveEE([self.gPos[0],self.gPos[1],self.gPos[2]],orn,max_step=self.TimestepPerAction)    
  
        observation = self.observe()

        
        self.reward = self.calcReward()  #rewDistObj + 0*rewDistEE + 100*self.targetCatch - (penalty1 + penalty2 + penalty3)
        terminal =  self.distEEFromObj < 0.005  or  (self.obj[0][2]<0.3) or self.hitTheGround>0 or self.simTime>500

        
        info = {"rew":self.reward}
        return observation, 0.01*self.reward, terminal, info


    def close (self):
        print (f"Environment({self.env_id}) is closing....")

   
   

def make_env(env_id,params, rank, seed=0):
        """
        Utility function for multiprocessed env.

        :param env_id: (str) the environment ID
        :param num_env: (int) the number of environments you wish to have in subprocesses
        :param seed: (int) the inital seed for RNG
        :param rank: (int) index of the subprocess
        """
        def _init():
            env = CustomEnv(env_id,params)
            #DummyVecEnv([lambda: CustomEnv()]) #gym.make(env_id)
            env.seed(seed + rank)
            return env
        # set_global_seeds(seed)
        set_random_seed(seed)


        return _init



if __name__ == "__main__":


    mymenu = Menu()
    CONFIG = mymenu.show_menu()
    myParams = InputParamMenu()

    # env = CustomEnv(1,myParams)

    num_cpu    = CONFIG["num_cpu"] # Number of processes to use
    max_epc    = CONFIG["max_epc"]

    if CONFIG["TestOrTrain"].upper() == "TRAIN":

      timestr = time.strftime("%Y%m%d-%H%M%S")

      modelName = "learnedPolicies/model_" + timestr
      logFname  = "learnedPolicies/log_"   + timestr
      paramFname  = "learnedPolicies/param_"   + timestr
      params = myParams.setParams(paramFname)
      

      ##==========================Create env===================================###
      if (num_cpu == 1):
        env = CustomEnv(1,params)
      else:
        env = SubprocVecEnv([make_env(i,params, i) for i in range(1, num_cpu)]) # Create the vectorized environment
       
      ###===========================Start Learing===============================###  
      model = PPO("MlpPolicy", env,verbose=0,tensorboard_log=logFname)
      model.learn(total_timesteps = max_epc)
      env.close()

      print ("saving the learned policy")
      model.save(modelName)
      del model

    ###===========================Enjoy trained agent==========================###
    else:
    
      mymenu.download_model()
      modelName     = mymenu.select_model()
      paramFname    = modelName.replace("model","param")
      paramFname    = paramFname.replace("zip","rpm")
      
      params = myParams.loadParms(paramFname) 
      
      maximized      = CONFIG["maximized"]#False
      AniPrevTime    = CONFIG["AniPrevTime"]#     #-1 -> disable 
      MasterCtrlGain = CONFIG["MasterCtrlGain"]

      print ("loading the learned policy")
      model  = PPO.load(modelName)
      
      env    = CustomEnv(0,params)
      env.controlAlgorithm = CONFIG["ctrlMethod"]
      
      obs    = env.reset()
      os.system("clear")

      act = None
      while (True):
            action, _states = model.predict(obs)
            if (act is not None):
                act = (1-MasterCtrlGain) * act + (MasterCtrlGain)*action
            else:
                act = action
            
            obs, rewards, dones, info = env.step(act)
            # print (f"{env.obj[0][0]:.3f} , {env.obj[0][1]:.3f},{env.obj[0][2]:.3f} , {env.ee[0][0]:.3f},{env.ee[0][1]:.3f},{env.ee[0][2]:.3f}")
            print (f"Rewards = {rewards:2.5f} \t action = {act[0]:2.3f}\t{act[1]:2.3f}\t{act[2]:2.3f}\t{act[3]:2.3f}")
            time.sleep(0.2)
            # env.drawCircle([-1,1], 0.3)
            if (dones):
                 obs=env.reset()

    