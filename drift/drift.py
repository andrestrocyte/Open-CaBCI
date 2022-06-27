'''
  
  Catalin Mitelut; github: "catubc"; mitelutco@gmail.com

'''

import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import scipy
import os
import time

def phase_correlation(a, b):
    G_a = np.fft.fft2(a)
    G_b = np.fft.fft2(b)
    conj_b = np.ma.conjugate(G_b)
    R = G_a*conj_b
    R /= np.absolute(R)
    r = np.fft.ifft2(R).real
    return r

#
def pad_data(img):
    img[:50] = 0
    img[-50:] = 0
    img[:,:50] = 0
    img[:,-50:] = 0

    return img

#
def get_drift_xy(data, 
                 x_drift_max, 
                 y_drift_max):

    # make random shift array
    x_shifts = np.random.randint(low=-x_drift_max,
                                 high = x_drift_max,
                                 size = data.shape[0])
    y_shifts = np.random.randint(low=-x_drift_max,
                                 high = x_drift_max,
                                 size = data.shape[0])

    return x_shifts, y_shifts
    
#
def make_template(data,
                  n_imgs_to_sample = 500,
                  n_best_imgs = 100):

    # find best correlation map first
    idx_imgs = np.random.choice(np.arange(data.shape[0]),
                                n_imgs_to_sample,
                                replace=False)

    # make temporary template to match to
    template = np.mean(data[idx_imgs],axis=0)

    #
    corr_maxs = np.zeros(idx_imgs.shape[0])
    ctr=0
    for k in tqdm(idx_imgs, desc="computing phase correlations"):

        #
        temp = phase_correlation(data[k], template)

        r,c = np.unravel_index(temp.argmax(), temp.shape)

        maxcorr = temp[r,c]
        corr_maxs[ctr] = maxcorr
        ctr+=1
    #
    idx = np.argsort(corr_maxs)[::-1]

    # take the n best images and compute template
    idx_best = idx[:n_best_imgs]
    template = data[idx_imgs[idx_best]].mean(0)

    #
    return corr_maxs, template, idx_imgs

def compute_drift(template, image):
    # compute phase correlation to the template.
    img_corr = phase_correlation(template, image)

    # find peak
    r,c = np.unravel_index(img_corr.argmax(), img_corr.shape)


    # convert to roll function which has negative and positive values
    if r > 512/2:
        r = r-512

    if c > 512/2:
        c = r-512
            #
    return r,c

######################################################################################
class DriftCorrection():

    ''' Class that implements drift correction of imaging data using phase correlation

        Input: ....
        Output: ...

    '''

    def __init__(self, 
				 fname_roi_pixels_and_thresholds,
				 shmem_live_frame,
                 shmem_drift_xy_values):

        #
        self.fname_roi_pixels_and_thresholds = fname_roi_pixels_and_thresholds

        #
	    self.shmem_live_frame = shmem_live_frame

        #
	    self.shmem_drift_xy_values = shmem_drift_xy_values

        #
        self.load_template()

        #
        self.initialize_drift_xy_state()

        #
        while True:

            #
            self.detect_drift()

            #


    #
    def initialize_drift_xy_state(self):

        #
        aa = np.zeros(2, dtype=np.float32)

        # get the rois_traces from the shared memory name
        self.existing_shmem_drift_xy_values = shared_memory.SharedMemory(name=
                                                                         self.shmem_drift_xy_values)

        #
        self.drift_xy_values = np.ndarray(aa.shape,
                                          dtype=aa.dtype,
                                          buffer=self.existing_shmem_drift_xy_values.buf)

    #
    def initialize_live_frame_shared_memory(self):

        ''' shared variable that keeps current image in memeory for plotter to visualize

        '''

        # make a numpy array to hold the rois_traces
        aa = np.zeros((1,512,512), dtype=np.uint16)

        # get the rois_traces from the shared memory name
        self.existing_shm_live_frame = shared_memory.SharedMemory(name=self.shmem_live_frame)

        #
        self.live_frame = np.ndarray(aa.shape,
                                 dtype=aa.dtype,
                                 buffer=self.existing_shm_live_frame.buf)
    #
    def load_template(self):
        ''' Load template computed in the calibration step

        '''

        #####################################################
        #
        data = np.load(self.fname_rois_pixels_and_thresholds,
                       allow_pickle=True)
        #
        self.template = data['calibration_template']

        #


	def detect_drift(self):

        # take live image
        r, c = compute_drift(self.template, self.live_frame)

        #
        self.drift_xy_values[0] = r.copy()
        self.drift_xy_values[0] = c.copy()






































					 

