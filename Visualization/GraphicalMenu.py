import os, json
import platform
import glob
import time
import math
import sys, select


class Menu():
    
  def __init__(self) -> None:
      self.smallColumnSize  = 5
      self.mediumColumnSize = 15
      self.largeColumnSize  = 30
       

  class bcolors():
      HEADER    = '\033[95m'
      OKBLUE    = '\033[94m'
      OKCYAN    = '\033[96m'
      OKGREEN   = '\033[92m'
      WARNING   = '\033[93m'
      FAIL      = '\033[91m'
      ENDC      = '\033[0m'
      BOLD      = '\033[1m'
      UNDERLINE = '\033[4m'
      #comes from print_format_table()
      TEST      = '\x1b[1;30;47m' 
      ROW_ODD   = '\x1b[7;30;47m'
      ROW_EVEN  = '\x1b[7;30;46m'
      ROW_HEAD  = '\x1b[4;33;40m' #6;30;47m'
    

  def print_format_table(self):
      """
      prints table of formatted text format options
      """
      for style in range(8):
          for fg in range(30,38):
              s1 = ''
              for bg in range(40,48):
                  format = ';'.join([str(style), str(fg), str(bg)])
                  s1 += '\x1b[%sm %s \x1b[0m' % (format, format)
              print(s1)
          print('\n')


  def print_config(self,conf):
      idx=0
      print("\n")
      print(self.bcolors.ROW_HEAD + "{:5s} {:15s}: {:15s}".format("ID","Param.","Val.")+self.bcolors.ENDC)
      
      for key,val in conf.items():
        if (type(val)==int):
          v = f"{val:,}"
        else:
          v = str(val)

        if (idx%2):
          print(self.bcolors.ROW_EVEN + "{:5s} {:15s}: {:15s}".format(str(idx+1),key,v)+self.bcolors.ENDC)
        else:
          print(self.bcolors.ROW_ODD +  "{:5s} {:15s}: {:15s}".format(str(idx+1),key,v)+self.bcolors.ENDC)  
        idx+=1 

      print("\n")     

  def show_menu(self):
    # print_format_table() # to check the color table just uncomment this line
    os.system("clear")
    print(self.bcolors.OKGREEN + "Welcome to our control suite\n " + self.bcolors.ENDC)
    __cwd__ = os.path.realpath( os.path.join(os.getcwd(), os.path.dirname(__file__)))

    try:
        with open(os.path.join(__cwd__, 'config_' + platform.node() + '.txt'),'r') as fp:
            self.config = json.loads(fp.read()) 
    except:
        # we can generate a config file instead 
        print(self.bcolors.FAIL+ "Configurations file is not found!!!"+self.bcolors.ENDC)
        print(self.bcolors.WARNING+"Generating a new configuration"+self.bcolors.ENDC)
        self.config = {"ctrlMethod":1, "num_cpu": 16, "max_epc": 5000000, "TestOrTrain": "Test", "MasterCtrlGain": 0.8, "AniPrevTime": -1, "maximized": 0}
        with open(os.path.join(__cwd__, 'config_' + platform.node() + '.txt'), 'w') as file:
          file.write(json.dumps(self.config))
      
        
    print (self.bcolors.HEADER +"======================{:20s}======================".format("Current configurations")+self.bcolors.ENDC)
    self.print_config (self.config)
    
    inp = input("would you like to change the configurations?(y/n)")  

    if (inp.upper()=="Y"):
      for key,val in self.config.items():
        if (type(val)==int):
          v = f"{val:,}"
        else:
          v = str(val)
        
        print(self.bcolors.OKCYAN + "{:15s}: {:17s}".format(key, v)+ self.bcolors.ENDC,end='')
        v_in = input("New value: ")
        if v_in != '':
          self.config[key] = type(self.config[key])(v_in)
          
      self.print_config (self.config)

      save_cfg = input("save new configurations (y/n)?") 
      if save_cfg.upper() == "Y":
        with open(os.path.join(__cwd__, 'config_' + platform.node() + '.txt'), 'w') as file:
          file.write(json.dumps(self.config))

    return self.config  

  def download_model(self):
    # scp -rp -P 7020 guest@10.227.107.100:/home/guest/firsttest/*.zip .    

    dl_models = input("\nWould you like to download models from server?(y/n)?")

    if dl_models.upper() == "Y":
      os.system ("scp -rp -P 7020 guest@10.227.107.100:/home/guest/learnedPolicies/*.* ./learnedPolicies/")
      # os.system ("scp -rp -P 7020 guest@10.227.107.100:/home/guest/learnedPolicies/*.zip ./learnedPolicies/")
      # os.system ("scp -rp -P 7020 guest@10.227.107.100:/home/guest/learnedPolicies/*.rpm ./learnedPolicies/")

  def select_model(self):

    ModelNames = [f for f in glob.glob("learnedPolicies/model_*.zip")]
    ModelNames.sort(key=lambda x: os.path.getmtime(x))

    idx=0
    print("\n")
    print(self.bcolors.ROW_HEAD + "{:5s} {:32s}".format("ID","Model Name")+self.bcolors.ENDC)
    
    for model in ModelNames:
      if (idx%2):
        print(self.bcolors.ROW_EVEN + "{:5s} {:32s}".format(str(idx+1),model.split("/")[1])+self.bcolors.ENDC)
      else:
        print(self.bcolors.ROW_ODD +  "{:5s} {:32s}".format(str(idx+1),model.split("/")[1])+self.bcolors.ENDC)  
      idx+=1 

    print("\n") 
    inp = input("select model ID or press enter for selecting the last model:")
    if (inp == ''):
      return ModelNames[-1]       
    else: 
      return ModelNames[int(inp)-1]       

  def showOptions(self):

      print("\n (L)Load a new model \n (X)Exit \n Continue ...")
      inputs, outputs, errors = select.select([sys.stdin], [], [], 2)
      inpKey = sys.stdin.readline().strip() if inputs else ""
      if (inpKey.upper()=='L'):
        model_name     = self.select_model()
        print (model_name + "is loading!")
        return [1,model_name]

        #model  = PPO2.load(model_name)
      elif (inpKey.upper() == 'X'):
        return  [0,""]
        # print ("Exit")
        # exit()  
      else:
        return [1,""]  