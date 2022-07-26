'''
  
  Catalin Mitelut; github: "catubc"; mitelutco@gmail.com

'''

import numpy as np
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import parmap
from multiprocessing import shared_memory
import time

#from scipy import ndimage as ndi
#from skimage.segmentation import watershed
#from skimage.feature import peak_local_max
#import scipy
#import os
#import time

#
def apply_shifts(img, x, y):

    #
	img = np.roll(img, x, axis=0)
	img = np.roll(img, y, axis=1)

	return img

def correct_drift_single_frame(img, shift):

    #
    x = shift[0]
    y = shift[1]

    #
    img = apply_shifts(img,
                           x,
                           y)
    #
    return img


def correct_drift(iter_number,
				  bmi_c, 
				  shifts):

	#
	for k in trange(bmi_c.data.shape[0], desc='fixing drift on calibration data'):

		#
		x = shifts[k][0]
		y = shifts[k][1]

		#
		temp = bmi_c.data[k].copy()

		bmi_c.data[k] = apply_shifts(temp,
						       x,
							   y)
							   
							   
	bmi_c.data.flatten().tofile(bmi_c.fname.replace('.raw','_'+str(iter_number)+'.raw'))

	return bmi_c

######################################################################################
class DriftCorrection():

    ''' Class that implements ONLINE drift correction of imaging data using phase correlation

        Input: [ca] raw image using shared-memory array from bmi
        Output: r, c: 2 integers which indicate the shift in the x-y directions
    '''

    def __init__(self,
				 fname_rois_pixels_and_thresholds,
				 shmem_live_frame,
                 shmem_drift_xy_values,
                 shmem_termination_flag,
                 ):

        #
        self.n_frames_smooth = 5
        
        # flag that computes average motion over last n frames for recomputing r,c values to bmi
        self.smooth_motion_flag = True
        
        # initialize a small array that will hold the latest n images in memory for smoothing pre-motion detection
        self.motion_history = np.zeros((self.n_frames_smooth,2), dtype=np.int32)

        #
        self.fname_rois_pixels_and_thresholds = fname_rois_pixels_and_thresholds

        #
        self.shmem_live_frame = shmem_live_frame

        #
        self.shmem_termination_flag = shmem_termination_flag

        # max # of pixels x or y direction that can move in a single frame
        self.max_motion_per_frame = 20

        #
        self.shmem_drift_xy_values = shmem_drift_xy_values

        #
        self.load_template()

        #
        self.initialize_drift_xy_state()

        #
        self.initialize_live_frame_shared_memory()

        #
        self.initialize_termination_flag()

        #
        while True:

            # grab latest image and shift the previous memory frames 1 over
            self.read_frame_shift_array()

            #
            self.detect_drift()

            #
            if self.termination_flag[0]:
                break

    #
    def read_frame_shift_array(self):

        # shift image array 1 over
        self.image_array[:-1] = self.image_array[1:]

        # insert the latest image
        self.image_array[-1] = self.live_frame[0]
        
    #
    def initialize_termination_flag(self):
        #
        aa = np.zeros((1,), dtype=np.int64)

        # get the rois_traces from the shared memory name
        self.existing_shm_termination_flag = shared_memory.SharedMemory(name=self.shmem_termination_flag)

        #
        self.termination_flag = np.ndarray(aa.shape,
                                           dtype=aa.dtype,
                                           buffer=self.existing_shm_termination_flag.buf)

    #
    def initialize_drift_xy_state(self):

        #
        aa = np.zeros(2, dtype=np.int32)

        # get the rois_traces from the shared memory name
        self.existing_shmem_drift_xy_values = shared_memory.SharedMemory(name=
                                                                         self.shmem_drift_xy_values)

        #
        self.drift_xy_values = np.ndarray(aa.shape,
                                          dtype=aa.dtype,
                                          buffer=self.existing_shmem_drift_xy_values.buf)

        # also initialize a memory array
        self.drift_xy_values_previous = np.zeros(2, dtype=np.int32)


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
                                 
        ########################################
        # initialize a small array that will hold the latest n images in memory for smoothing pre-motion detection
        self.image_array = np.zeros((self.n_frames_smooth, 512, 512),
                                    dtype=np.uint16)

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

        #
        #r, c = compute_drift_single_frame(self.template,
        #                                  self.live_frame)

        #
        r, c = compute_drift_multiple_frames(self.template,
                                             self.image_array)

        #
        # print ("DRIFT CLASS motion detection (r,c): ", r, c)

        # also shift the motion history over 1 frame:
        if self.smooth_motion_flag:
            self.motion_history[:-1] = self.motion_history[1:]
            self.motion_history[-1] = [r,c]
            
            #print ("Computed presmooth SHIFTS rc, : ", r,c)
            r = np.median(self.motion_history[:,0])
            c = np.median(self.motion_history[:,1])
            #print ("Self.motion history: ", self.motion_history)
            #print ("Computed Postsmootoh SHIFTS rc, : ", r,c)

        # check if delta_r and delta_c are too large, and limit huge jumps which could be errors
        # if jumps are too high then just levae the previous drift in place
        if (self.drift_xy_values_previous[0]-r)<=self.max_motion_per_frame:
            self.drift_xy_values[0] = int(r)
        if (self.drift_xy_values_previous[1] - c) <= self.max_motion_per_frame:
            self.drift_xy_values[1] = int(c)

#
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
    y_shifts = np.random.randint(low=-y_drift_max,
                                 high = y_drift_max,
                                 size = data.shape[0])

    return x_shifts, y_shifts


def phase_correlation_parallel(idx_parmap,
                               template,
                               fname):
    # load the data as mmap;
    # - this should avoid memmory crash issues
    # TODO: can even have this mmap reset every 1000 frames so that
    #   any size data can be processed!!
    data = np.memmap(fname, dtype='uint16', mode='r')
    data = data.reshape(-1, 512, 512)

    #
    corr_maxs = np.zeros(data.shape[0])
    shifts = np.zeros((data.shape[0],2))

    #
    subtract_flag = True

    #
    a = template.copy()
    if idx_parmap[0]==0:
        for idx in tqdm(idx_parmap, desc="phase corr computation"):

            # seelct an image
            #print ('idx: ', idx, data.shape)
            b = data[idx]

            #
            G_a = np.fft.fft2(a)
            G_b = np.fft.fft2(b)
            conj_b = np.ma.conjugate(G_b)
            R = G_a * conj_b
            R /= np.absolute(R)
            surface = np.fft.ifft2(R).real

            # compute peak location for row and column
            r, c = np.unravel_index(surface.argmax(), surface.shape)

            #
            corr_maxs[idx] = surface[r, c]

            # convert to roll function which has negative and positive values
            if subtract_flag:
                if r > 512 / 2:
                    r = r - 512

                if c > 512 / 2:
                    c = c - 512

            #
            shifts[idx] = [r, c]

    else:
        for idx in idx_parmap:

            # seelct an image
            # print ('idx: ', idx, data.shape)
            b = data[idx]

            #
            G_a = np.fft.fft2(a)
            G_b = np.fft.fft2(b)
            conj_b = np.ma.conjugate(G_b)
            R = G_a * conj_b
            R /= np.absolute(R)
            surface = np.fft.ifft2(R).real

            # compute peak location for row and column
            r, c = np.unravel_index(surface.argmax(), surface.shape)

            #
            corr_maxs[idx] = surface[r, c]

            # convert to roll function which has negative and positive values
            if subtract_flag:
                if r > 512 / 2:
                    r = r - 512

                if c > 512 / 2:
                    c = c - 512

            #
            shifts[idx] = [r, c]


    #
    return np.int32(shifts), corr_maxs

#
def make_template(data,
                  fname_mmap,
                  n_imgs_to_sample = 500,
                  n_best_imgs = 100,
                  template = None,
                  idx_imgs = None,
                  random_img_sample_flag = True,
                  plotting=False,
                  n_cores=1):

    # find best correlation map first
    # don't pick random frames, much harder to find matchin frames

    if idx_imgs is None:
        if random_img_sample_flag:
            idx_imgs = np.random.choice(np.arange(data.shape[0]),
                                    n_imgs_to_sample,
                                    replace=False)
        else:
            idx_start = np.random.choice(np.arange(data.shape[0]-n_imgs_to_sample))
            idx_imgs = np.arange(idx_start,
                                 idx_start+n_imgs_to_sample,
                                 1)

    #
    #print ("idx imgs; ", idx_imgs)

    # make temporary template to match to
    if template is None:
        template = np.mean(data[idx_imgs],axis=0)

    # parallelize
    if n_cores==1:
        corr_maxs = np.zeros(idx_imgs.shape[0])
        ctr=0
        for k in tqdm(idx_imgs, desc="computing phase correlations"):

            #
            temp = phase_correlation(data[k], template)

            r,c = np.unravel_index(temp.argmax(), temp.shape)

            maxcorr = temp[r,c]
            corr_maxs[ctr] = maxcorr
            ctr+=1
    else:
        # split the image indexes into gropus
        imgs_split = np.array_split(idx_imgs,
                                    40)

        #
        res = parmap.map(phase_correlation_parallel,
                         imgs_split,   # indexes of each image to process
                         template,     # defatul template
                         fname_mmap,   # place where to load data from
                         pm_pbar = True,
                         pm_processes = n_cores
                         )

        # initialize arrays
        shifts = np.zeros((data.shape[0], 2))
        corr_maxs = np.zeros(data.shape[0])

        # merge the shifts:
        for k in range(len(res)):
            shifts = shifts + res[k][0]
            corr_maxs = corr_maxs + res[k][1]

        # select only the values chose
        #shifts = shifts[idx_imgs]
        corr_maxs = corr_maxs[idx_imgs]

    #
    idx = np.argsort(corr_maxs)[::-1]

    # take the n best images and recompute template
    idx_best = idx[:n_best_imgs]
    template = data[idx_imgs[idx_best]].mean(0)

    #
    return corr_maxs, template, idx_imgs

#
def compute_drift_multi_frames(iter_number,
                               bmi_c,
                               subsample=1,
                               n_cores=1):
    
    #                              
    template = bmi_c.template
    data = bmi_c.data
    fname = bmi_c.fname
    
    # make sure to save different files each time a step is processed
    # TODO: no need to work with the entire dataset here, could just do a subset
    if iter_number >0:
        fname = fname.replace('.raw','_'+str(iter_number-1)+".raw")
    
    #
    print ("computing motion on: ", fname)

    #
    if n_cores>1:
        idx_all = np.arange(data.shape[0])

        # subsample data
        idx_all = idx_all[::subsample]

        #
        idx_parmap = np.array_split(idx_all,
                                    40)
        #
        res = parmap.map(phase_correlation_parallel,
                         idx_parmap,  # indexes of each image to process
                         template,  # defatul template
                         fname,  # place where to load data from
                         pm_pbar=True,
                         pm_processes=n_cores
                         )

        # initialize arrays
        shifts = np.zeros((data.shape[0], 2))
        corr_maxs = np.zeros(data.shape[0])

        # merge the shifts:
        for k in range(len(res)):
            shifts = shifts + res[k][0]
            corr_maxs = corr_maxs + res[k][1]

        # fix any subsmapled frame by taking previous:
        # TODO: this looks dangerous... it replaces 0,0 detected drift by previous val
        print ("TODO: undo interpolation for drift with better function")
        for k in range(shifts.shape[0]):
            if shifts[k][0]==0 and shifts[k][1]==0:
                shifts[k] = shifts[k-1]
                corr_maxs[k] = corr_maxs[k-1]

    else:
        print ("non parallel version not implemented")

    #
    return np.int32(shifts), corr_maxs


#
def compute_drift_multiple_frames(template, images):

    #
    # take sample of data
    image = np.mean(images, axis=0)

    #
    img_corr = phase_correlation(template, image)

    r, c = np.unravel_index(img_corr.argmax(), img_corr.shape)

    #
    if r > 512 / 2:
        r = r - 512

    if c > 512 / 2:
        c = c - 512

    #
    return int(r), int(c)


#
def compute_drift_single_frame(template, image):

    # compute phase correlation to the template.
    img_corr = phase_correlation(template, image[0])

    # find peak
    r,c = np.unravel_index(img_corr.argmax(), img_corr.shape)


    # convert to roll function which has negative and positive values
    if r > 512/2:
        r = r-512

    if c > 512/2:
        c = c-512
            #

    #
    return int(r), int(c)

#
def template_generation(bmi_c,
                        ):
    n_iter = 1
    bmi_c.n_imgs_to_sample = 1000
    random_img_sample_flag = False
    bmi_c.n_best_imgs = 100
    plotting = True
    n_cores = 8
    template=None
    idx_imgs = None
    for k in range(n_iter):
        #idx_imgs = None
        corr_maxs, template, idx_imgs = make_template(
                                                  bmi_c.data,
                                                  bmi_c.fname,
                                                  bmi_c.n_imgs_to_sample,
                                                  bmi_c.n_best_imgs,
                                                  template,
                                                  idx_imgs,
                                                  random_img_sample_flag,
                                                  plotting,
                                                  n_cores)
    bmi_c.template = template
    bmi_c.idx_imgs = idx_imgs

    #
    return bmi_c

#

#
def plot_mean_vs_template(bmi_c):

    plt.figure()
    print (bmi_c.data.shape, bmi_c.idx_imgs.shape)
    temp = bmi_c.data[bmi_c.idx_imgs].mean(0)
    plt.title("Average map over " + str(bmi_c.idx_imgs.shape[0]) + " images")
    plt.imshow(temp,
               #vmin=0,
               #vmax=1500
               #
               )

    #
    plt.figure()
    plt.title("Average map over highest correlated " + str(bmi_c.n_best_imgs) + " images")
    plt.imshow(bmi_c.template,
               #vmin=0,
               #vmax=1500
               )

    #
    plt.show()




#
def make_motion_template_and_correct_data(bmi_c):

    n_iterations = 1  # number of times to iterate over the template
    n_cores = 12
    subsample = 3  # subsample every N'th frame for drift to speed up;
                    # TODO: would be good to use all frames!!

    for k in range(n_iterations):
        # make template
        bmi_c = template_generation(bmi_c)

        # detection of motion
        shifts, _ = compute_drift_multi_frames(k,
                                               bmi_c,
                                               subsample,
                                               n_cores,)

        # correct motion (+ save to disk)
        bmi_c = correct_drift(k,
                              bmi_c,
                              shifts)

        #
        plt.figure()
        s=np.array(shifts)
        t= np.arange(s.shape[0])
        plt.plot(t,s[:,0])
        plt.show()

    # remake template
    bmi_c = template_generation(bmi_c)

    #
    return bmi_c































                     

