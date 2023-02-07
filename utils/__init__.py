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
def smooth_ca_time_series5(diff):
	#
	''' This is a more aggressive

	'''

	temp = (diff[-1] * 0.7+
			diff[-2] * 0.3 )

	return temp

def smooth_ca_time_series4(diff):
	#
	''' This returns the last value. i.e. no filter

	'''

	temp = (diff[-1]*0.4+
			diff[-2] * 0.25 +
			diff[-3] * 0.15 +
			diff[-4] * 0.10 +
			diff[-5] * 0.10)

	return temp


def get_mode(F_raw_clipped):
    
    modes = []
    bin_widths = np.arange(0.1,5,0.1)
    for bin_width in bin_widths:
        
        # binarize the fluorescence data to a selected resolution from 0.1 to 5 fluorecence values
        F_c = np.round(bin_width*np.floor(np.round(F_raw_clipped / bin_width,2)),1)

		# compute the mode of the distribution
        modes.append(scipy.stats.mode(F_c, keepdims=True, nan_policy='omit')[0][0])
        
    #
    mode = np.mean(modes)
    
    return mode



#
def smooth_ca_time_series3(diff):
	#
	''' This returns the last value. i.e. no filter

	'''


	return np.mean(diff[-1])

#
def smooth_ca_time_series2(diff):
	#
	''' This returns the median of the last 5 time steps

	'''


	return np.mean(diff[-5:])


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

# get baseline f0 after smoothing
def compute_dff0(data):
	
	f0 = np.median(data, axis=0)
	
	#
	dff0 = (data-f0)/f0
	
	return f0, dff0

# get baseline f0 after smoothing
def compute_dff0_with_reference(data, reference_data):
	
	''' same as above function but uses the reerence data to compute baseline not the fed in time series
	'''
	
	f0 = np.median(reference_data, axis=0)
	
	#
	dff0 = (data-f0)/f0
	
	return f0, dff0

	
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
	"""
	low_freq = 1000
	high_freq = 16000
	octaves = np.arange(int(low_freq/1000), 1+int(high_freq/1000))
    
	octaves = 2**(octave_size*x)
    
	#
	return np.array(1000*octaves)
	"""
	x = np.arange(int(low_freq/1000), 1+int(high_freq/1000))
	octaves = 2**(octave_size*x)
	return np.array(1000*octaves)

	#


#
def ensemble_to_tone_transfer_function_high_and_low(ensemble_state,
												    low_freq,
												    high_freq,
												    low_threshold,
												    high_threshold,
													octave_freqs):

	# for now do a linear projection between the neural states and the
	# TODO: calibrate the speaker so that it macthes the assumed playback states
	#

	# scale from 0..1
	#
	#
	n_octaves = len(octave_freqs)
	baseline_freq = octave_freqs[n_octaves//2]

	#
	if ensemble_state>=0:
		#tone_raw = baseline_freq + (ensemble_state/high_threshold)*(high_freq-baseline_freq)
		tone_raw = octave_freqs[min(len(octave_freqs)-1,len(octave_freqs)//2+int(len(octave_freqs)//2*ensemble_state/high_threshold))]
	else:
		#tone_raw = baseline_freq - (ensemble_state/low_threshold)*(baseline_freq-low_freq)
		tone_raw = octave_freqs[max(0,len(octave_freqs)//2-int(len(octave_freqs)//2*ensemble_state/low_threshold))]
	# map onto frequencies selected
	#tone = tone*(high_freq-low_freq)+low_freq

	# find closest octave frequency to the tone
	idx = np.argmin(np.abs(octave_freqs-tone_raw))
	tone = octave_freqs[idx]

	return tone

#
def ensemble_to_tone_transfer_function_absolute(ensemble_state,
											   low_freq,
											   high_freq,
											   low_threshold,
											   high_threshold):

	# for now do a linear projection between the neural states and the
	# TODO: calibrate the speaker so that it macthes the assumed playback states
	#

	# scale from 0..1
	tone = (ensemble_state)/(high_threshold)

	# map onto frequencies selected
	tone = tone*(high_freq-low_freq)+low_freq

	# map onto octaves
	octave_freqs = get_octave_frequencies(low_freq,
										  high_freq,
										  0.25)

	# find closest octave frequency to the tone
	idx = np.argmin(np.abs(octave_freqs-tone))
	tone = octave_freqs[idx]

	return tone

#
def make_white_noise(ensemble_state,
			    low_freq,
			    high_freq,
			    low_threshold,
			    high_threshold):

	# for now do a linear projection between the neural states and the
	# TODO: calibrate the speaker so that it macthes the assumed playback states
	#
	#high_threshold = 1500
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
