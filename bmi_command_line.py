import os
from multiprocessing import Process
import time

# 
from bmi.bmi import BMI
from plotter.plotter import PlotROIs
from tone.tone import PlayTone
from water.water import WaterReward

from tkinter import Tk #for Python 3.x
from tkinter.filedialog import askopenfilename, askdirectory

########################################################################
########################################################################
########################################################################
if __name__ ==  '__main__':
	
	####################################################################### 			
	################### DEFAULT PARAMTERS FOR BMI ######################### 			
	####################################################################### 			
	sampleRate_2P = 30    # # frames of recording   +  buffer frames, usually 10-15 sec
	n_seconds_session = int(20000/sampleRate_2P)                          # number of seconds to run the BMI
	n_frames_session = 10000
	simulation_flag_bmi = False         # Runs the BMI class in simulation mode (i.e. don't need Bscope input)
										#  - set to true unless we have a real mouse in the BScope to get
										#    real time data from; otherwise data is read from disk at some location
										# TODO: in non simulation mode - have slightly different panels for 
										#       reading directories of the data as Bscope does not make them until 
										#       it starts up
	simulation_flag_tone = False        # Runs the tone class in simulation mode
	simulation_flag_water = False       # Runs the water class in simulation mode                                   
	
	sleep_time_sec = 0.00001

	##########################################################################
	#################### LOAD FILE/DIRECTORY LOCATIONS ####################### 
	##########################################################################
	
	Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
	
	fname_root_path = askdirectory()
	
	#fname_fluorescence = askopenfilename() # show an "Open" dialog box and return the path to the selected file
	fname_fluorescence = os.path.join(fname_root_path,
									  'data_002',                   # this is the root directory of the .raw file saved by bscope
									  'Image_001_001.raw')
	print(" Fname fluorsecnce: ", fname_fluorescence)
	
	# 
	#fname_root_path = os.path.split(fname_fluorescence)[0]

	# required for simulation mode
	fname_ttl = os.path.join(fname_root_path,
							 "ttl_pulses.npy")
							 
	#
	fname_roi_pixels_and_thresholds = os.path.join(fname_root_path,
								'rois_pixels_and_thresholds.npz')

	###############################################################
	#################### INITIALIZE BMI ########################### 
	###############################################################
	bmi = BMI(simulation_flag_bmi,
			  fname_root_path,
			  fname_fluorescence,
			  #fname_rois,
			  #fname_freq,
			  fname_ttl,
			  sampleRate_2P,
			  fname_roi_pixels_and_thresholds,
			  n_seconds_session,
			  n_frames_session)

	# for simulation mode we sometimes want to slow down the processing;
	# ... not as necessary 
	bmi.sleep_time_sec = sleep_time_sec # Delay in simulation mode

	# Flag to print out information from the proessing
	bmi.verbose = False
	bmi.verbose2 = False    # this displays the time it takes to copute ROI

	###############################################################
	############## INITIALIZE TONE PLAYBACK #######################
	###############################################################
	'''  Here we pass only the ensemble state (i.e. E1-E2) to the 
		tone player. The tone player alone then computes the transfer function
		as this is not related to anything else in the BMI class
	'''
	if False:
		print ("RUNNING Tone player in multiprocessing...")
		tone_player_ = Process(target=PlayTone, args=(fname_roi_pixels_and_thresholds,
													  bmi.shmem_ensemble_state.name,
													  bmi.shmem_tone_state.name,
													  bmi.shmem_termination_flag.name,
													  simulation_flag_tone,))
		tone_player_.start()


	###############################################################
	############## INITIALIZE WATER REWARD ########################
	###############################################################
	'''  Here we pass only the ensemble state (i.e. E1-E2) to the 
		tone player. The tone player alone then computes the transfer function
		as this is not related to anything else in the BMI class
	'''
	
	#
	if True:
		print ("RUNNING water reward multiprocessing...")
		water_reward_ = Process(target=WaterReward, args=(bmi.shmem_water_reward.name,
														  bmi.shmem_termination_flag.name,
														  simulation_flag_water,
														  ))
		water_reward_.start()

	###############################################################
	############## INITIALIZE PLOTTER ############################# 
	###############################################################
	'''  This is the plotting functions that visualize ROI time sries
	'''
	#print ("RUNNING plotter in multiplrocessing ...")
	if True:
		plotter_ = Process(target=PlotROIs, args=(
												fname_roi_pixels_and_thresholds,
												bmi.shmem_rois_traces.name,
												bmi.shmem_n_ttl.name,
												bmi.rois_traces_raw.shape,
												bmi.shmem_reward_times.name,
												bmi.shmem_tone_state.name,
												bmi.shmem_termination_flag.name,
												))
		plotter_.start()

	###############################################################
	#################### INITIALIZE BMI ########################### 
	###############################################################

	# loop to wait 2 sec until plotting is initialized:
	# TODO: autod detect when plotting is initialized 
	time.sleep(2)
	
	#
	bmi.run_BMI()
	
	# close all classes
	bmi.close()

	#
	plotter_.close()
	tone_player_.close()
	water_reward_.close()

	quit()
