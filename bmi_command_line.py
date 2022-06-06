import os
from multiprocessing import Process

# 
from bmi.bmi import BMI
from plotter.plotter import PlotROIs
from tone.tone import PlayTone
from water.water import WaterReward
import tkinter as tk
from tkinter.filedialog import askopenfilename

#############################################################################
#############################################################################
#############################################################################
# bad practice!!! do not use global varaiables!!!
global fname_fluorescence
#
root = tk.Tk()
root.geometry('1250x200+1250+200')


#
def get_calibration_raw_data():
	global fname_fluorescence
	#
	fname_fluorescence = askopenfilename()
	print('Selected 3:', fname_fluorescence)
	root.destroy()


#
button1 = tk.Button(text='Load calibration .raw',
					command=get_calibration_raw_data,
					bg='brown',
					fg='white')
button1.pack(padx=2, pady=5)

#
root.mainloop()

########################################################################
########################################################################
########################################################################

# FOR LINUX
fname_root_path = os.path.split(fname_fluorescence)[0]
#'/home/cat/data/donato/bscope_tests/'
#fname_root_path = '/media/cat/4TB/donato/BSCOPE_tests/'
#fname_root_path = '/media/cat/4TBSSD/donato/Bscope_tests/'

#fname_root_path = r'D:\bmi'

#
#fname_fluorescence = os.path.join(fname_root_path,
#								  'mouse2_bmi5',
#                                  'Image_001_001.raw')

#
#fname_freq =  os.path.join(fname_root_path,
#                           "freq.npy")
#
#fname_rois = os.path.join(fname_root_path,
#                          "ensemble_rois_centres.txt")

# required for simulation mode
fname_ttl = os.path.join(fname_root_path,
                         "ttl_pulses.npy")
                         
#
fname_roi_pixels_and_thresholds = os.path.join(fname_root_path,
						'rois_pixels_and_thresholds.npz')

#
fname_roi_pixels_and_thresholds = os.path.join(fname_root_path,
											   'rois_pixels_and_thresholds.npz')

#############################################################
if __name__ ==  '__main__':
	
	####################################################################### 			
	################### DEFAULT PARAMTERS FOR BMI ######################### 			
	####################################################################### 			
	sampleRate_2P = 30    # # frames of recording   +  buffer frames, usually 10-15 sec
	n_seconds_session = int(20000/sampleRate_2P + 450/sampleRate_2P)                          # number of seconds to run the BMI
	simulation_flag = True							# Run BMI in simulation mode (i.e. don't need Bscope input)

	###############################################################
	#################### INITIALIZE BMI ########################### 
	###############################################################
	bmi = BMI(simulation_flag,
			  fname_root_path,
			  fname_fluorescence,
			  #fname_rois,
			  #fname_freq,
			  fname_ttl,
			  sampleRate_2P,
			  fname_roi_pixels_and_thresholds,
			  n_seconds_session)

	# for simulation mode we sometimes want to slow down the processing;
	# ... not as necessary 
	bmi.sleep_time_sec = 0.0001 # Delay in simulation mode

	# Flag to print out information from the proessing
	bmi.verbose = False
	bmi.verbose2 = False    # this displays the time it takes to copute ROI


	###############################################################
	############## INITIALIZE PLOTTER ############################# 
	###############################################################
	'''  This is the plotting functions that visualize ROI time sries
	'''
	#print ("RUNNING plotter in multiplrocessing ...")
	plotter_ = Process(target=PlotROIs, args=(
											fname_roi_pixels_and_thresholds,
											bmi.shmem_rois_traces.name,
											bmi.shmem_n_ttl.name,
											bmi.rois_traces_raw.shape,
											bmi.shmem_reward_times.name,
											bmi.shmem_tone_state.name,
											bmi.shmem_termination_flag.name,
											simulation_flag
											))
	plotter_.start()


	###############################################################
	############## INITIALIZE TONE PLAYBACK #######################
	###############################################################
	'''  Here we pass only the ensemble state (i.e. E1-E2) to the 
		tone player. The tone player alone then computes the transfer function
		as this is not related to anything else in the BMI class
	'''

	print ("RUNNING Tone player in multiprocessing...")
	tone_player_ = Process(target=PlayTone, args=(fname_roi_pixels_and_thresholds,
												  bmi.shmem_ensemble_state.name,
												  bmi.shmem_tone_state.name,
												  bmi.shmem_termination_flag.name,
												  simulation_flag))
	tone_player_.start()


	###############################################################
	############## INITIALIZE WATER REWARD ########################
	###############################################################
	'''  Here we pass only the ensemble state (i.e. E1-E2) to the 
		tone player. The tone player alone then computes the transfer function
		as this is not related to anything else in the BMI class
	'''

	print ("RUNNING water reward multiprocessing...")
	water_reward_ = Process(target=WaterReward, args=(bmi.shmem_water_reward.name,
													  simulation_flag,
													  bmi.shmem_termination_flag.name,))
	water_reward_.start()

	###############################################################
	#################### INITIALIZE BMI ########################### 
	###############################################################


	#
	#time.sleep(5)
	bmi.run_BMI()

	#
	plotter_.close()
	tone_player_.close()
	water_reward_.close()

	quit()
