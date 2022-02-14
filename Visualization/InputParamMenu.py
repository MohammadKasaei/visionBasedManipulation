import os, json
import platform
import glob
import time
import math
import sys, select


class InputParamMenu():
    
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


  def loadParms(self,fname):
    __cwd__ = os.path.realpath( os.getcwd())
    try:
        with open(os.path.join(__cwd__, fname),'r') as fp:
            self.params = json.loads(fp.read()) 
    except:
        print(self.bcolors.FAIL+ "Parameters are not found!!!"+self.bcolors.ENDC)
        print(self.bcolors.WARNING+"Generating a new one"+self.bcolors.ENDC)
        self.setParams(fname.replace(".rpm",""))
        
    return self.params


  def setParams(self,fname):
    __cwd__ = os.path.realpath( os.getcwd())

    self.params = {"MasterGain":1.0, "Param1":1.0,"Param2":1.0,"Param3":1.0,"Param4":1.0,"Param5":1.0,"Param6":1.0}
    with open(os.path.join(__cwd__, fname + '.rpm'), 'w') as file:
      file.write(json.dumps(self.params))
  
        
    print ("\n\n"+self.bcolors.HEADER +"======================{:20s}======================".format("Current parameteres")+self.bcolors.ENDC)
    self.print_config (self.params)
    
    inp = input("Would you like to change the parameteres?(y/n)")  

    if (inp.upper()=="Y"):
      for key,val in self.params.items():
        if (type(val)==int):
          v = f"{val:,}"
        else:
          v = str(val)
        print(self.bcolors.OKCYAN + "{:15s}: {:17s}".format(key, v)+ self.bcolors.ENDC,end='')
        v_in = input("New value: ")
        if v_in != '':
          self.params[key] = type(self.params[key])(v_in)
          
      self.print_config (self.params)

      save_cfg = input("Would you like to save the new parameteres (y/n)?") 
      if save_cfg.upper() == "Y" or save_cfg.upper() =='':
        with open(os.path.join(__cwd__, fname + '.rpm'), 'w') as file:
          file.write(json.dumps(self.params))

    return self.params  
