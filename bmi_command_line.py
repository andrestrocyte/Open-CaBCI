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
from bmi.bmi import BMI
from plotter.plotter import PlotROIs
from tone.tone import PlayTone


# FOR LINUX
fname_root_path = '/home/cat/data/donato/bscope_tests/'
#fname_root_path = '/media/cat/4TB/donato/BSCOPE_tests/'
#fname_root_path = '/media/cat/4TBSSD/donato/Bscope_tests/'

#
fname_fluorescence = os.path.join(fname_root_path, 
                                  'image_27000frames.raw')

#
fname_freq =  os.path.join(fname_root_path,
                           "freq.npy")
#
fname_rois = os.path.join(fname_root_path, 
                          "ensemble_rois_centres.txt")

# required for simulation mode
fname_ttl = os.path.join(fname_root_path,
                         "ttl_pulses.npy")
                         
#
fname_roi_pixels_and_thresholds = os.path.join(fname_root_path,
						'rois_pixels_and_thresholds.npz')


####################################################################### 			
################### DEFAULT PARAMTERS FOR BMI ######################### 			
####################################################################### 			
sampleRate_2P = 30
n_seconds_session = int(10000/30)                          # number of seconds to run the BMI 
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
bmi.sleep_time_sec = 0.033 # Delay in simulation mode

# Flag to print out information from the proessing
bmi.verbose = False
bmi.verbose2 = False    # this displays the time it takes to copute ROI


###############################################################
############## INITIALIZE PLOTTER ############################# 
###############################################################

if True:
	if True:
		print ("RUNNING Plotter in multiprocessing...")
		plotter_ = Process(target=PlotROIs, args=(
						fname_roi_pixels_and_thresholds,
						bmi.shmem_rois_traces.name,
						bmi.shmem_n_ttl.name,
						bmi.rois_traces_raw.shape,
						bmi.shmem_reward_times.name,
						bmi.shmem_tone_state.name
						))
		plotter_.start()

	# else:
		# print ("RUNNING Plotter in main process...")
		# plotter_ = PlotROIs(
						# bmi.shmem_rois_traces.name,
						# bmi.shmem_n_ttl.name,
						# bmi.rois_traces_raw.shape)


###############################################################
############## INITIALIZE TONE PLAYBACK #######################
###############################################################
'''  Here we pass only the ensemble state (i.e. E1-E2) to the 
	tone player. The tone player alone then computes the transfer function
	as this is not related to anything else in the BMI class
'''

if True:
	if True:
		print ("RUNNING Tone player in multiprocessing...")
		tone_player_ = Process(target=PlayTone, args=(fname_roi_pixels_and_thresholds,
													  bmi.shmem_ensemble_state.name,
													  bmi.shmem_tone_state.name,))
		tone_player_.start()

	else:
		print ("RUNNING Tone player in main process...")
		tone_player_ = PlayTone(bmi.shmem_ensemble_state.name)


###############################################################
#################### INITIALIZE BMI ########################### 
###############################################################


#
#time.sleep(5)
bmi.run_BMI()

quit()
