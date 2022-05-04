'''
  
  C. Mitelut; github: "catubc"; mitelutco@gmail.com

'''

import numpy as np
import time
import nidaqmx
from nidaqmx.constants import (AcquisitionType)  # https://nidaqmx-python.readthedocs.io/en/latest/constants.html
import numpy as np
import matplotlib.pyplot as plt
from nidaqmx.constants import TerminalConfiguration
import math

#
class BMI():
	
	def __init__(self,
				 fname_fluroescence,
				 sampleRate_2P,
				 n_frames=3000):
		
		# 
		self.fname_fluorescence = fname_fluorescence

		#
		# Define variables
		self.sampleRate_NI = 1E3     # Sample rate of NI card
		
		#
		self.sampleRate_2P = sampleRate_2P	    # Sample rate of BScope 
		
		#
		self.n_frames_to_be_acquired = n_frames   # Number of frames from BScope
		
		#
		self.rois_smooth_window = 5   				# Number of frames to use to smooth the ROI traces
													# to be developed/changed further

		# initizlie ROIs using either a text file or more specific code
		#   for now simple version uses some random centres in the imaging file
		#   MUST CHANGE
		rois_fname = r"D:\User Training\rois.txt"
		self.initialize_ROIs(rois_fname)
		
		# initialize all arrays to be used:
		self.initialize_arrays()
		
		# locatoin of where to save the tone frequency
		# TODO: try to develop more dynamic option
		self.fname_freq =  r"F:\freq.npy"
	
	def initialize_arrays(self):
		
		# 
		self.data = []					# array to hold data being read
		self.n_ttl = 0					# ttl pulse counter
		self.ttl_n_computed = []	    # number of ttl pulses computed based on time elapsed
		self.ttl_n_detected = []        # number of ttl pulses detected based on TTL from NI board
		self.inter_ttl_time = []        # computed time between each detected TTL pluse
		self.abs_times = [0] 			# NOT SURE REQUIRED ANYMORE; keeps track of every TTLL read outside the main BMI
										# loop;   might be useful for debugging later on kernel interuptions etc.
		self.ttl_times = []				# ttl times to be saved
		self.previous_trigger=0			# time of the previous TTL trigger to be used to determine if next trigger etc
		self.prev_max = 0 				# TTL pulse previous read max value
		self.prev_min = 0				# TTL pulse previous read min value


	def initialize_ROIs(self, rois_fname=None):
		'''
			Initialize the ROIs and ensemble arrays to be used below
		'''
		
		# load ROI centres from disk;
		#  TODO run proper ROI detection with irregular shape et.
		self.rois = np.loadtxt(self.rois_fname, delimiter = ',', dtype=np.int32)
		print (" ROIS: , ", self.rois.shape)

		# 
		self.roi_width = 10 # number of pixels around ROI to grab
		print ("   using squire ROIs; TODO: use proper defined ROIs and cell masks ...") 
		
		# initialize the fluorescence time series for all the ROIs that are being tracked
		self.rois_traces = []
		for k in range(self.rois.shape[0]):
			self.rois_traces.append([])
			
		# 	
		self.smooth_function = np.arange(0,self.rois_smooth_window,1)/self.rois_smooth_window
		
		#
		self.ensemble_activity_realtime = np.zeros(self.rois.shape[0])

	#
	def run_BMI(self):

		# NOT SURE FUNCTION... TO DELETE
		# self.read_data_flag = True

		# 
		start = time.perf_counter_ns()
		
		#
		n_sec_to_stop_after_no_TTL_pluse = 5

		print('Running BMI (ctrl-c to stop)')
		#
		
		if self.test_mode == True:
			self.task_ttl = self.test_mode_functions
		else:
			self.task_ttl = nidaqmx.Task('bmi_online'):

			# set TTL pulse reader from 2p system
			self.task_ttl.ai_channels.add_ai_voltage_chan("Dev3/ai0", terminal_config = TerminalConfiguration.NRSE)


			#
			self.task_ttl.timing.cfg_samp_clk_timing(self.sampleRate_NI,
											#samps_per_chan=pointsToPlot*2, 
											sample_mode=AcquisitionType.CONTINUOUS)

			#
			self.task_ttl.start()
			
			# set time to auto finish the acquisition;
			#    TODO: automate acquisition end if no TTL pulses for X seconds
			t_end = (time.perf_counter() + 
			         self.n_frames_to_be_acquired/self.sampleRate_2P+
			         5) 
			         
			#
			pts = 1  # number of values to read from NI card
			self.now = time.perf_counter() #time.perf_counter_ns()/1E9
			self.frame_no = 0
			self.previous_trigger = time.perf_counter()-2 # set the previous tirgger a few sec prior to start
			
			# abssolute start time
			self.start = time.perf_counter()
			
			# start recording and acquisition
			while self.now < t_end:
				data = self.task_ttl.read(number_of_samples_per_channel=pts)
				
				#  leave these in just in case we end up reading at higher bit rates and multiple samples at a atime
				self.min_ = np.min(data)
				self.max_ = np.max(data)
				self.data.append(data)
				# 
				
				# get time of ttl pulse
				self.now = time.perf_counter() #perf_counter_ns()/1E9
				
				# not sure this is requiered; TO DELETE
				# while True:
				#	if (now - self.abs_times[-1])>0:
		        #				break
				# 	now = time.perf_counter() #perf_counter_ns()/1E9

				# 
				# self.now = now
				self.abs_times.append(self.now)
				
				# check of ttl pulse when from high ~5 to low ~0
				if self.min_<1 and self.prev_max>=1:
											
					#
					self.bmi_update()
									
					# update trigger time
					self.previous_trigger = self.now
				# 
				self.prev_min = self.min_
				self.prev_max = self.max_						
				
			task.stop()
		
		#
		self.save_data()


	#
	def bmi_update(self):
	
		# 
		# One workaround that may or may not be helpful in your specific 
		# code: create new mmap objects periodically (and get rid of old ones), 
		# at logical points in your workflow. Then the amount of RAM needed 
		# should be roughly proportional to the number of array items you touch 
		# between such steps. Against that, it takes time to create and destroy 
		# new mmap objects. So it's a balancing act.
	
	
		# Initialize ttl arrays; also make mmap of file to be read
		if len(self.ttl_n_computed)==0:
			self.ttl_computed=0
			
			#
			if self.read_data_flag:
				ss = time.time()
				self.newfp = np.memmap(self.fname, dtype='uint16', mode='r', 
									   shape=(self.n_frames_to_be_acquired,512,512))
				print (" duration to setup memmap: ", time.time()-ss)
				print ("     TODO: work with 1D flattened arrays")
			
			# reset start time: requird becaues we start the BMI a few seconds before the BScope 
			self.start=self.now
		else:
			
			# 
			self.ttl_computed = round((self.now-self.start)*self.sampleRate_2P)
				
		# load the [ca] imaging and compute activity in each ROI
		self.update_rois()

		# compute the ensemble activity from ROIs loaded
		self.update_ensembles()	
		
		#
		self.compute_ensemble_to_tone_state()	
		
		# update tone and check for task completion
		self.update_tone()				
		
		# check for reward condition:
		self.check_reward_condition()
		
		# save meta data 
		self.ttl_n_computed.append(self.ttl_computed)
		self.ttl_n_detected.append(self.n_ttl)
		self.ttl_times.append(self.now)

		# 
		self.n_ttl+=1
				
		#	
		
	def trigger_reward(self):
		
		# generate water reward
		
		pass
		
	def post_reward_state(self):
			
		# disable tone playback;
		self.tone_off()
		
		# run while loop until ensemble activit return to normal;
		# start recording and acquisition
		while self.now < t_end:
			
			# search for next TTL pulse
			data = self.task_ttl.read(number_of_samples_per_channel=pts)
			
			#  leave these in just in case we end up reading at higher bit rates and multiple samples at a atime
			self.min_ = np.min(data)
			self.max_ = np.max(data)
			self.data.append(data)
			
			# get time of ttl pulse
			self.now = time.perf_counter() #perf_counter_ns()/1E9
			self.abs_times.append(self.now)
			
			# check of ttl pulse when from high ~5 to low ~0
			if self.min_<1 and self.prev_max>=1:
										
				#
				self.update_rois()
								
				# 
				self.update_ensembles()
								
				# check to see if neural activity back to baseline
				if self.check_baseline_condition():
					break
				
				# update trigger time
				self.previous_trigger = self.now
			# 
			self.prev_min = self.min_
			self.prev_max = self.max_		
		
		
		pass
		
	def missed_reward_state(self):
		
		# run time out sequence for 10 sec
		#    play white noise!? to distinguish it from the post-reward state
		#
		
		pass
		
	def check_baseline_condition(self):
		
		# check if ensemble activity back to baseline; e.g. within 1 x of std
		# if abs(ensemble...[ensbmel_ID1] - ensbmel..[ensemble_ID2])< std x 1? 
		#     return True
		# else:
		#     return False
		
		pass
		
		
	def check_reward_condition(self):
		
		# check if esnemble difference reach threshold
		# if ensemble...[ensbmel_ID1] - ensbmel..[ensemble_ID2]> condition1  OR
		#        ...................''...............           < condition 2:
		 
		#       # trigger reward to mouse
		#       self.trigger_reward()
		#
		#       #
		#		self.post_reward_state()
		
		pass
		
	# 
	def update_ensembles(self):
		
		# wait for at min frames to be grabbed 
		if self.n_ttl<self.rois_smooth_window:
			self.ensemble_activity_realtime[:] = 0
		
		# update each ensemble based on some smoothing function
		else:
			for p in range(self.rois.shape[0]):
				
				# grab last 5 frames (e.g.)	
				temp = self.rois_traces[p][self.n_ttl-self.rois_smooth_window:self.n_ttl]
				
				# scale using a linear decay/trinagle function
				temp = temp*self.smooth_function
				
				# take largest value
				temp = np.max(temp)
			
				#
				self.ensemble_activity_realtime[p] = temp
		
		# 
		print (" ensembles realtime: ", self.ensemble_activity_realtime)
		
	
	def tone_off(self):
		
		# turn toneplayback off	
		#  freq = 0
		#  np.save(self.fname_freq,freq)

		pass
		
	def compute_ensemble_to_tone_state(self):
			
		# compute ensemble -> tone trasnfer function



		pass
		
	def update_tone(self):
		
		
		# update current tone use self.freq
		#  
		#  np.save(self.fname_freq,freq)
		
		pass
		
	#	
	def update_rois(self):
		
		if self.read_data_flag:
			
			#
			if self.ttl_computed>0 and self.ttl_computed<(self.n_frames_to_be_acquired-1):
				print(f"  detected frame #: ", self.n_ttl, 
					   " computed_frame : ", self.ttl_computed)
				
				# TODO: write this fucntion to loop until it finds a non-zero mean then exit: 
				for p in range(self.rois.shape[0]):
					
					roi_sum0 = self.newfp[self.n_ttl-1,
										  self.rois[p][0]-self.roi_width:self.rois[p][0]+self.roi_width,
										  self.rois[p][1]-self.roi_width:self.rois[p][1]+self.roi_width].mean()					
										  
					roi_sum1 = self.newfp[self.n_ttl,
										  self.rois[p][0]-self.roi_width:self.rois[p][0]+self.roi_width,
										  self.rois[p][1]-self.roi_width:self.rois[p][1]+self.roi_width].mean()
					
					roi_sum2 = self.newfp[self.n_ttl+1,
										  self.rois[p][0]-self.roi_width:self.rois[p][0]+self.roi_width,
										  self.rois[p][1]-self.roi_width:self.rois[p][1]+self.roi_width].mean()
									
					self.rois_traces[p].append(roi_sum0)
	  
					#print (" roi: ", p, roi_sum0, roi_sum1, roi_sum2)
					
				print ("")
				print ("")
				
		pass

		
	#	
	def save_data(self):
		
		#
		np.save(r"D:\User Training\data.npy",self.data)
		np.save(r"D:\User Training\ttl_n_computed.npy",self.ttl_n_computed)
		np.save(r"D:\User Training\ttl_n_detected.npy",self.ttl_n_detected)

		#
		np.save(r"D:\User Training\abs_times.npy",self.abs_times)
		np.save(r"D:\User Training\ttl_times.npy",self.ttl_times)
		np.save(r"D:\User Training\rois_traces.npy",self.rois_traces)

	
	
#######################################
fname = r"D:\User Training\Readtest1\Image_001_001.raw"
n_frames = 100
sampleRate_2P = 30


bmi = BMI()
bmi.run_BMI(fname,
			n_frames,
			sampleRate_2P,
)
