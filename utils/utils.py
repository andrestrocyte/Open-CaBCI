import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import signal
import scipy
import scipy.ndimage

import cv2

#
def smooth_ca_time_series(diff):
	#
	return np.mean(diff)

#
def convolve_parallel(idx, data_sparse):
	
	# 
	kernel = 11
	for k in idx:
		for p in range(data_sparse.shape[2]):
			data_sparse[:,k,p] = np.convolve(data_sparse[:,k,p], kernel, mode='same')

	return data_sparse


def get_octave_frequencies(low_freq,
						   high_freq,
						   octave_size=0.25):
	#
	octaves = []

	#
	octaves.append(low_freq)
	temp = low_freq
	while True:
		temp = temp * (1 + octave_size)
		if temp > high_freq:
			break
		octaves.append(temp)

	return np.array(octaves)

#
def ensemble_to_tone_transfer_function(ensemble_state,
									   low_freq,
									   high_freq,
									   low_threshold,
									   high_threshold):

	# for now do a linear projection between the neural states and the
	# TODO: calibrate the speaker so that it macthes the assumed playback states
	#
	high_threshold = 1500
	# scale from 0..1
	tone = (ensemble_state)/(high_threshold)
	# map onto frequencies selected
	tone = tone*(high_freq-low_freq)+low_freq

	# map onto octaves
	octave_freqs = get_octave_frequencies(low_freq,
										   high_freq,
										   0.25)

	#
	idx = np.argmin(np.abs(octave_freqs-tone))
	tone = octave_freqs[idx]

	return tone

##############################
class ComputeROIs(object):
	
	#
	def __init__(self, fname):
		
		#
		self.fname = fname

		# some of these paramters need to be exposed outside also
		# self.vmin = 500
		# self.vmax = 1500
		
		# 
		self.binarize_thresh =.05
		self.sigma = .5
		self.order = 0
		self.n_smooth_steps = 1
	
	
	def make_corr_map(self):
		''' Not yet working or tested etc.

		'''

		data = np.memmap(self.fname, dtype='uint16', mode='r')
		data = data.reshape(-1,512,512)
		print ("memmap : ", data.shape)

		data_sparse = data[::self.subsample]
		print ("data into analysis: ", data_sparse.shape)

		# 
		img = scipy.signal.correlate2d(data_sparse[0], 
									   data_sparse[1], 
									   mode='same')
		
		# 
		plt.figure()
		plt.imshow(img,
				   #vmin=vmin,
				   #vmax=vmax
				  )
		plt.show()
		
	
		return img
	
	
	# 
	def make_std_map(self):

		data = np.memmap(self.fname, dtype='uint16', mode='r')
		data = data.reshape(-1,512,512)
		print ("memmap : ", data.shape)

		data_sparse = data[::self.subsample]
		print ("data into analysis: ", data_sparse.shape)

		# filter once to remove much of the white noise
		if True:
			sigma = 1
			order = 0
			print (" gaussian filter width: ", sigma, ", order: ", order)
			data_sparse = scipy.ndimage.gaussian_filter(data_sparse, 
														sigma, 
														order)
			print ("done filtering... (TO CHECK which axis are we filtering!!)")
        
        #
		if False:
			kernel = [7,0,0]   # filter only across time
			print (" median filter width: ", kernel)
			data_sparse = signal.medfilt(data_sparse, kernel)
			print ("done median filtering... ")
		
		#
		if False:

			#scipy.ndimage.gaussian_filter1d
			#import scipy.ndimage # import gaussian_filter1d
			#kernel = [1,1,7]
			kernel = 30
			print (" filter1d: ", kernel)
			data_sparse = scipy.ndimage.gaussian_filter1d(data_sparse, kernel)
			print ("done filter1d... ", data_sparse.shape)
		
		# 
		if False:

			#
			if False:
				import parmap
				n_cores = 8
				idx = np.array_split(np.arange(data_sparse.shape[1]),n_cores)
				#print ("data split idx: ", idx)
				
				res = parmap.map(convolve_parallel, 
								 idx,
								 data_sparse,
								 pm_processes = n_cores,
								 pm_pbar = True)
				
				#
				print (" len res: ", len(res), res[0].shape)
				
				# 
				data_sparse = np.sum(data_sparse,axis=0)
				print ("recombined data sparse", data_sparse.shape)
			
			#
			else:
				data_out = np.zeros(data_sparse.shape)
				for k in trange(data_sparse.shape[1]):
					for p in range(data_sparse.shape[2]):
						data_out[:,k,p] = np.convolve(data_sparse[:,k,p], kernel, mode='same')
			
			print ("done window smoothing...")
				
		# 
		print ("staring computing std...")
		std = np.std(data_sparse,axis=0)
		print ("done computing std...")
		#
		std = (std-self.vmin)/(self.vmax-self.vmin)
		idx = np.where(std<0)
		std[idx]=0
		idx = np.where(std>1)
		std[idx]=1

		# 
		plt.figure()
		plt.imshow(std,
				   #vmin=vmin,
				   #vmax=vmax
				  )
		plt.show()
		
		self.std_map = std
		
		return std

	def area_inside_convex_hull(self, pts):
		lines = np.hstack([pts,np.roll(pts,-1,axis=0)])
		area = 0.5*abs(sum(x1*y2-x2*y1 for x1,y1,x2,y2 in lines))
		return area

	def binarize_data(self, img, thresh):
		
		#thresh = .15
		idx = np.where(img>thresh)
		img[idx]=1
		idx = np.where(img<=thresh)
		img[idx]=0
			
		return img

	#
	def find_roi_boundaries(self, image):

		for k in range(self.n_smooth_steps):
			image = scipy.ndimage.gaussian_filter(image, 
												  self.sigma, 
												  self.order)

			image = self.binarize_data(image, self.binarize_thresh)
		
		#
		image = image.astype('int32')
						
		# run watershed segmentation
		distance = ndi.distance_transform_edt(image)
		coords = peak_local_max(distance, 
								footprint=np.ones((17, 17)), 
								labels=image)

		# 
		mask = np.zeros(distance.shape, dtype=bool)
		mask[tuple(coords.T)] = True
		markers, _ = ndi.label(mask)
		labels = watershed(-distance, 
						   markers, 
						   mask=image)
		#
		labels = labels.astype('float32')

		# remove very small and very large ROIs
		min_size = 15
		max_size = 250 #???
		roi_centres = []
		indexes = []
		for k in np.unique(labels):
			idx = np.where(labels==k)
			
			
			if idx[0].shape[0]<min_size or idx[0].shape[0]>max_size:
				labels[idx]=np.nan
			else:
				
				roi_centres.append([np.median(idx[0]),
									 np.median(idx[1])])
				indexes.append(idx)

		self.rois = np.vstack(roi_centres)		
		self.indexes = indexes
		
	# 
	def compute_contour_map(self, std_map, cell_ids):
		''' Compute contours and save them to disk also
		
		'''
		
		# 
		contour_array = []
		for cell_id in cell_ids:
			temp = np.zeros(std_map.shape, dtype='uint8')
			temp[self.indexes[cell_id]]=1
			#temp = temp.astype('uint8')
			
			#
			contour, _ = cv2.findContours(temp, 
											cv2.RETR_TREE, 
											cv2.CHAIN_APPROX_SIMPLE)
			contour = contour[0].squeeze()
			contour = np.vstack((contour, contour[0]))

			# 
			contour_array.append(contour)
	

		return contour_array
		

	#
	def show_contour_map(self, std_map, indexes, fig=None):
		
		if fig is None:
			plt.figure()
			
		plt.imshow(std_map)
		for p in range(len(indexes)):
			temp = np.zeros(std_map.shape)
			temp[indexes[p]]=1
			temp = temp.astype('uint8')
			contour, _ = cv2.findContours(temp, 
											cv2.RETR_TREE, 
											cv2.CHAIN_APPROX_SIMPLE)
			contour = contour[0].squeeze()
			contour = np.vstack((contour, contour[0]))

			# 
			for k in range(len(contour)-1):
				plt.plot([contour[k][0], contour[k+1][0]],
						 [contour[k][1], contour[k+1][1]],
						c='white')
			# 
			z = np.vstack(indexes[p]).T
			plt.text(np.median(z[:,1]), np.median(z[:,0]), str(p),c='red')

		plt.show()
    
	#
	def compute_traces2(self):
		''' Same as below but visualize every single frame
		'''
		
		data = np.memmap(self.fname, dtype='uint16', mode='r')
		data = data.reshape(-1,512,512)
		print ("memmap : ", data.shape)
			
		#  
		plt.figure()
		traces = []
		ctr=0
		ax=plt.subplot(121)
		roi_traces = []
		for k in range(len(self.rois)):
			roi_traces.append([])
		
		# loop over each frame
		#skip = 1
		for p in trange(0, data.shape[0], self.trace_subsample):

			# grab frame
			frame = data[p]

			# loop over ROIS
			for k in range(0,len(self.indexes)):
				#loc = np.int32(np.array(self.rois[k])/1.5)  # why are we dividing by 1/5?  Is this due to smoothign!?
				#loc = np.int32(np.array(self.rois[k]))  # why are we dividing by 1/5?  Is this due to smoothign!?

				# grab roi
				temp = frame[self.indexes[k]]

				# normalize by surface area so that cells don't look way different because of footprint size
				if True:
					temp = temp/self.indexes[k][0].shape[0]

				# add pixel values inside roi
				temp = np.nansum(temp)

				# save
				roi_traces[k].append(temp)
				
		#
		roi_traces = np.array(roi_traces)
		self.roi_traces = roi_traces

		#	
		t = np.arange(0, data.shape[0], self.trace_subsample)/30.
		ctr = 0

		# save the baselin of the cells in order to be able to offset it in the BMI
		# TODO: this is important; it functions as a rough DFF method
		#    TODO: we may wish to implement a more complex version of this
		self.roi_f0s = np.zeros(len(roi_traces),dtype=np.float32)
		for k in range(len(roi_traces)):

			temp = roi_traces[k]
			self.roi_f0s[k] = np.median(temp)
			temp = temp - self.roi_f0s[k]
			plt.plot(t, temp+ctr*self.scale)
		
			ctr+=1

		#
		labels = np.arange(len(self.rois))
		labels_old = np.arange(0,ctr*self.scale,self.scale)
		
		#
		plt.yticks(labels_old, labels)
		plt.xlabel("Time (sec)")
        
        # 
		ax=plt.subplot(122)
		new_plot = False
		self.show_contour_map(self.std_map,self.indexes, new_plot)

		plt.show()
		
	def show_traces_ids(self, ids):
		
		#
		fig=plt.figure()
		
		#
		plt.title("Cell Ids: "+str(ids))
		#	
		t = np.arange(0, self.roi_traces[0].shape[0],1)/30.*self.trace_subsample
		ctr = 0
		for k in ids:

			temp = self.roi_traces[k]
			temp = temp- np.median(temp)
			plt.plot(t, temp+ctr*self.scale)
		
			ctr+=1
			
		labels = np.arange(len(ids))
		labels_old = np.arange(0,ctr*self.scale,self.scale)
		
		#
		plt.yticks(labels_old, labels)
		plt.xlabel("Time (sec)")
        
		plt.show()
		
		
		
	def compute_traces(self):
		
		data = np.memmap(self.fname, dtype='uint16', mode='r')
		data = data.reshape(-1,512,512)
		print ("memmap : ", data.shape)
			
		#  
		plt.figure()
		traces = []
		ctr=0
		ax=plt.subplot(121)
		roi_traces = []
		
		
		for k in trange(0,len(self.rois)):
			loc = np.int32(np.array(self.rois[k])/1.5)
			
			# check every .3 secs
			t = np.arange(0, data.shape[0], 10)/30.
			traces = []
			# step in time
			for p in range(0, data.shape[0], 10):
				
				# grab frame
				temp = data[p]
				
				# grab roi
				temp = temp[self.indexes[k]]
				
				# normalize by surface area
				if True:
					temp = temp/self.indexes[k][0].shape[0]
				
				# add data inside roi
				temp = np.nansum(temp)
				
				# save
				traces.append(temp)

			#
			traces = np.array(traces)
			traces = traces- np.median(traces)

			#
			roi_traces.append(traces)
			
			plt.plot(t, traces+ctr*self.scale)
			ctr+=1
			
		labels = np.arange(len(self.rois))
		labels_old = np.arange(0,ctr*self.scale,self.scale)
		plt.yticks(labels_old, labels)
		plt.xlabel("Time (sec)")
        
		#
		ax=plt.subplot(122)
		new_plot = False
		self.show_contour_map(self.std_map,self.indexes, new_plot)

		plt.show()
		
		self.roi_traces = roi_traces
		
		return roi_traces

	#
	def find_reward_thresholds_high(self):

		# TODO: Make sure that these functions are identical to those being used inside the BMI!!!


		# run smoothing on each ensemble
		if self.smooth_diff_function_flag:

			# ensemble #1
			for p in range(2):
				smooth = np.zeros(self.roi_traces[self.ensemble1[p]].shape)
				for k in trange(self.rois_smooth_window, self.roi_traces[self.ensemble1[p]].shape[0], 1):
					smooth[k] = smooth_ca_time_series(self.roi_traces[self.ensemble1[p]][k - self.rois_smooth_window:k])
				#
				self.roi_traces[self.ensemble1[p]] = smooth

			# ensemble #2
			for p in range(2):
				smooth = np.zeros(self.roi_traces[self.ensemble2[p]].shape)
				for k in trange(self.rois_smooth_window, self.roi_traces[self.ensemble2[p]].shape[0], 1):
					smooth[k] = smooth_ca_time_series(self.roi_traces[self.ensemble2[p]][k - self.rois_smooth_window:k])
				#
				self.roi_traces[self.ensemble2[p]] = smooth

		# get baseline f0 after smoothing
		self.roi_f0s = []
		self.roi_f0s.append(np.median(self.roi_traces[self.ensemble1[0]], axis=0))
		self.roi_f0s.append(np.median(self.roi_traces[self.ensemble1[1]], axis=0))
		self.roi_f0s.append(np.median(self.roi_traces[self.ensemble2[0]], axis=0))
		self.roi_f0s.append(np.median(self.roi_traces[self.ensemble2[1]], axis=0))

		# detrend traces and make ensembles
		temp0 = self.roi_traces[self.ensemble1[0]] - self.roi_f0s[0]
		temp1 = self.roi_traces[self.ensemble1[1]] - self.roi_f0s[1]
		E1 = temp0 + temp1

		#
		temp2 = self.roi_traces[self.ensemble2[0]] - self.roi_f0s[2]
		temp3 = self.roi_traces[self.ensemble2[1]] - self.roi_f0s[3]
		E2 = temp2 + temp3

		# initialize the max and min values
		max_E1 = np.max(E1)
		max_E2 = np.max(E2)
		low = -max_E1
		high = max_E2

		print("low, high: ", low, high)
		# difference between ensemble
		diff = E1 - E2

		#
		n_sec_recording = int(diff.shape[0] / self.sample_rate)
		n_rewards_random = n_sec_recording // self.sample_rate
		print("nsec recording: ", n_sec_recording,
			  "max # of random rewards (i.e. every 30sec) ", n_rewards_random)

		# loop over time series decreasing the rewards until we hit the random #
		n_rewards = 0
		stepper = 0.95
		while n_rewards < n_rewards_random:

			# run inside while loop for eveyr setting of low and high until we hit
			#   exact number of random rewards
			k = 0
			n_rewards = 0
			reward_times = []
			while k < diff.shape[0]:

				temp_diff = diff[k]

				# #
				# if temp_diff <= low:
				# 	# low reward state reached
				# 	n_rewards += 1
				# 	reward_times.append([k, 0])
				# 	k += int(self.post_reward_lockout * self.sample_rate)
				# elif
				if temp_diff >= high:
					# high reward state reached
					n_rewards += 1
					reward_times.append([k, 1])
					k += int(self.post_reward_lockout * self.sample_rate)
				else:
					k += 1

			# print ("Reard times: ", reward_times)
			# check exit condition otherwise decrase thresholds
			if len(reward_times) > 1:
				rewarded_times = np.vstack(reward_times)
				high *= stepper
			else:
				high *= stepper

		print("updated rwards #: ", n_rewards, low, high)

		self.reward_times = np.vstack(reward_times)

		self.low = np.nan
		self.high = high
		self.E1 = E1
		self.E2 = E2
		self.diff = diff

	#
	def find_reward_thresholds_low_and_high(self):

		# run smoothing on each ensemble
		if self.smooth_diff_function_flag:

			for p in range(2):
				smooth = np.zeros(self.roi_traces[self.ensemble1[p]].shape)
				for k in trange(self.rois_smooth_window, self.roi_traces[self.ensemble1[p]].shape[0],1):
					smooth[k] = smooth_ca_time_series(self.roi_traces[self.ensemble1[p]][k-self.rois_smooth_window:k])
				#
				self.roi_traces[self.ensemble1[p]] = smooth

			for p in range(2):
				smooth = np.zeros(self.roi_traces[self.ensemble2[p]].shape)
				for k in trange(self.rois_smooth_window, self.roi_traces[self.ensemble2[p]].shape[0],1):
					smooth[k] = smooth_ca_time_series(self.roi_traces[self.ensemble2[p]][k-self.rois_smooth_window:k])
				#
				self.roi_traces[self.ensemble2[p]] = smooth


		# detrend traces and make ensembles
		temp0 = self.roi_traces[self.ensemble1[0]]-np.median(self.roi_traces[self.ensemble1[0]],axis=0)
		temp1 = self.roi_traces[self.ensemble1[1]]-np.median(self.roi_traces[self.ensemble1[1]],axis=0)
		E1 = temp0+temp1

		#
		temp2 = self.roi_traces[self.ensemble2[0]]-np.median(self.roi_traces[self.ensemble2[0]],axis=0)
		temp3 = self.roi_traces[self.ensemble2[1]]-np.median(self.roi_traces[self.ensemble2[1]],axis=0)
		E2 = temp2+temp3

		# initialize the max and min values
		max_E1 = np.max(E1)
		max_E2 = np.max(E2)
		low = -max_E1
		high = max_E2

		print ("low, high: ", low, high)
		# difference between ensemble
		diff = E1-E2

		#
		n_sec_recording = int(diff.shape[0]/self.sample_rate)
		n_rewards_random = n_sec_recording//self.sample_rate
		print ("nsec recording: ", n_sec_recording,
			   "max # of random rewards (i.e. every 30sec) ", n_rewards_random)

		# loop over time series decreasing the rewards until we hit the random #
		n_rewards = 0
		stepper = 0.95
		while n_rewards<n_rewards_random:

			# run inside while loop for eveyr setting of low and high until we hit 
			#   exact number of random rewards
			k=0
			n_rewards = 0
			reward_times = []
			while k<diff.shape[0]:
				
				temp_diff = diff[k]
				
				#
				if temp_diff<=low:
					# low reward state reached
					n_rewards+=1
					reward_times.append([k,0])
					k+= int(self.post_reward_lockout*self.sample_rate)
				elif temp_diff>=high:
					# high reward state reached
					n_rewards+=1
					reward_times.append([k,1])
					k+= int(self.post_reward_lockout*self.sample_rate)
				else:
					k+=1

			#print ("Reard times: ", reward_times)
			# check exit condition otherwise decrase thresholds
			if len(reward_times)>1:
				rewarded_times = np.vstack(reward_times)
				if self.balance_ensemble_rewards_flag:
					idx_E1 = np.where(rewarded_times[:,1]==0)[0].shape[0]
					idx_E2 = np.where(rewarded_times[:,1]==1)[0].shape[0]
					if idx_E1 <= idx_E2:
						low*=stepper
					else:
						high*=stepper
				else:
					low*=stepper
					high*=stepper
			else:
				low*=stepper
				high*=stepper

		print ("updated rwards #: ", n_rewards, low, high)

		self.reward_times = np.vstack(reward_times)
		
		self.low = low
		self.high = high
		self.E1 = E1
		self.E2 = E2
		self.diff = diff
		
	#
	def plot_rewarded_ensembles(self):
		
		#
		plt.figure()
		
		t = np.arange(self.diff.shape[0])/self.sample_rate
		plt.plot([t[0],t[-1]], [self.low, self.low], '--', c='grey')
		plt.plot([t[0],t[-1]], [self.high, self.high], '--', c='grey')
		plt.plot(t,self.E1,c='blue',alpha=.1,label='E1')
		plt.plot(t,self.E2,c='red',alpha=.1,label='E2')
		plt.plot(t, self.diff,c='black', alpha=.8, label='Difference')
		plt.plot([t[0],t[-1]], [0, 0], c='black', linewidth=3)

		#
		for k in range(len(self.reward_times)):
			temp = self.reward_times[k]

			if temp[1]==0:
				plt.plot([t[temp[0]], t[temp[0]]], [-5000,5000], '--', c='red')
			else:
				plt.plot([t[temp[0]], t[temp[0]]], [-5000,5000], '--', c='blue')

		# replot two random rewards just to make nice legend
		idx1 = np.where(self.reward_times[:,1]==0)[0].shape[0]
		idx2 = np.where(self.reward_times[:,1]==1)[0].shape[0]

		plt.plot([t[temp[0]], t[temp[0]]], [-5000,5000], '--', c='red', label='E1 rewarded # '+str(idx1),)
		plt.plot([t[temp[0]], t[temp[0]]], [-5000,5000], '--', c='blue', label='E2 rewarded # '+str(idx2),)
		plt.legend()

		plt.title("Rec duration: " + str(int(t[-1])) + " sec; expected # of random rewards: "+str(int(t[-1]/30)))
		plt.show()
