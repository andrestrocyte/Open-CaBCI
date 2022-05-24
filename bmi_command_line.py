import os
import numpy as np
import time
import nidaqmx
from nidaqmx.constants import (AcquisitionType)  # https://nidaqmx-python.readthedocs.io/en/latest/constants.html
import numpy as np
import matplotlib.pyplot as plt
from nidaqmx.constants import TerminalConfiguration
import math
from multiprocessing import Process

# 
from bmi import BMI, PlotROIs



# FOR LINUX
fname_root_path = '/home/cat/data/donato/bscope_tests/'
fname_fluorescence = os.path.join(fname_root_path, 
                                  'image_1000frames.raw')

#
fname_freq =  os.path.join(fname_root_path,
                           "freq.npy")

#
fname_rois = os.path.join(fname_root_path, 
                          "rois.txt")

# required for simulation mode
fname_ttl = os.path.join(fname_root_path,
                         "ttl_pulses.npy")


####################################################################### 			
################### DEFAULT PARAMTERS FOR BMI ######################### 			
####################################################################### 			
sampleRate_2P = 30
n_seconds_session = 30                          # number of seconds to run the BMI 
simulation_mode = True							# Run BMI in simulation mode (i.e. don't need Bscope input)

###############################################################
#################### INITIALIZE BMI ########################### 
###############################################################
bmi = BMI(simulation_mode,
          fname_root_path,
          fname_fluorescence,
          fname_rois,
          fname_freq,
          fname_ttl,
          sampleRate_2P,
          n_seconds_session)

# for simulation mode we sometimes want to slow down the processing;
# ... not as necessary 
bmi.sleep_time_sec = 0.033

# Flag to print out information from the proessing
bmi.verbose = False
bmi.verbose2 = False    # this displays the time it takes to copute ROI


###############################################################
#################### INITIALIZE BMI ########################### 
###############################################################
#print ("BMI BMI: ", bmi.n_ttl)

if True:
	print ("RUNNING Plotter in multiprocessing...")
	plotter_ = Process(target=PlotROIs, args=(
                    bmi.shmem_rois_traces.name,
                    bmi.shmem_n_ttl.name,
                    bmi.rois_traces.shape,))
else:
	print ("RUNNING Plotter in main process...")
	plotter_ = PlotROIs(
                    bmi.shmem_rois_traces.name,
                    bmi.shmem_n_ttl.name,
                    bmi.rois_traces.shape)

#print ("BMI BMI: ", bmi.n_ttl)
plotter_.start()

###############################################################
#################### INITIALIZE BMI ########################### 
###############################################################
#
# plotter_.join()

# 
# plotter_.terminate()

#
bmi.run_BMI()


