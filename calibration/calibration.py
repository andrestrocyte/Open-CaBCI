import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange, tqdm

from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import signal
import scipy
import scipy.ndimage
import cv2
from matplotlib.widgets import Slider, Button, RadioButtons

from stardist.models import StarDist2D
from stardist.data import test_image_nuclei_2d
from stardist.plot import render_label
from csbdeep.utils import normalize
plt.ion()


##############################


import matplotlib.pyplot as plt
import nidaqmx
from nidaqmx.constants import (AcquisitionType)
from nidaqmx.constants import TerminalConfiguration
import tqdm
import os
import time
import numpy as np
from multiprocessing import shared_memory
from utils.utils import smooth_ca_time_series, compute_dff0, compute_dff0_with_reference
from drift.drift import apply_shifts


#################################################
############# SIMULATION CLASS ##################
#################################################
class Simulation():
    ''' This class simulates the TTL pulses coming out of the Bscope
        by reading a ttl pulse file and returning values as requested
    '''

    def __init__(self,
                 fname_ttl):

        # set location of reading index to 0 at beginning
        self.index = 0

        # read ttl pulses from file
        self.ttl = np.load(fname_ttl)

    def read(self, number_of_samples_per_channel):

        # added a 2nd value for the lick detector
        ttl_val_bscope = self.ttl[self.index:self.index+number_of_samples_per_channel]
        ttl_val_lick_detector = 0
        ttl_val_rotary_encoder_1 = 0
        ttl_val_rotary_encoder_2 = 0

        self.index += number_of_samples_per_channel

        out = np.hstack((ttl_val_bscope[0],
                         ttl_val_lick_detector,
                         ttl_val_rotary_encoder_1,
                         ttl_val_rotary_encoder_2))

        #
        return out

    def stop(self):
        # not required in simulation mode
        pass


    def close(self):
        # not required in simulation mode
        pass


#################################################
################## BMI CLASS ####################
#################################################
class BMICalibration():

    ''' BMI class
        Inputs:
            - path of Thorimage memmap file where [ca] data is to be saved
            - ROI values for ROIs to be tracked during BMI
            - path of speaker file where tones are saved
            - ...

        Outputs:
            -

        TODO: we may want to fix the names of all the shared variables
              i.e., don't use random variables, just fix them to something that doesn't change
              - this way, we don't even need to share them between the modules
        TODO:  this way we dont' even have to pass them to the other modules
    '''

    #
    def __init__(self,
                 simulation_mode_bmi,
                 simulation_flag_licking,
                 fname_root_path,
                 fname_fluorescence,
                 fname_ttl,
                 sampleRate_2P,
                 fname_roi_pixels_and_thresholds,
                 max_n_seconds_session,
                 n_frames_session,
                 video_width,
                 video_length):

        #
        print ("... initializing BMI parameters...")
        print ("    TODO: consider saving all imaging data to RAM disk (or faster SSD) for improved speeds")

        #
        self.video_width = video_width
        self.video_length = video_length

        #
        self.simulation_mode_bmi = simulation_mode_bmi

        #
        self.simulation_mode_lick_detector = simulation_flag_licking

        #
        self.apply_drift_flag = True

        #
        self.fname_root_path = fname_root_path
        self.fname_fluorescence = fname_fluorescence
        self.fname_ttl = fname_ttl

        #
        self.fname_save_data = os.path.split(fname_fluorescence)[0]+"bmi_results.npz"

        #
        self.fname_rois_pixels_thresholds = fname_roi_pixels_and_thresholds

        #
        self.shared_memory_variables_names_list = []

        # NOT SURE IF REQUIRED... TO DELETE
        # TODO flag was probably used during development toskip the reading step;
        self.read_data_flag = True

        # Define variables
        self.sampleRate_NI = 1E3     # Sample rate of NI card

        #
        self.ttl_pts = 1             # number of values to read from NI card - usually we read a single value to avoid buffering issues

        #
        self.sampleRate_2P = sampleRate_2P      # Sample rate of BScope

        # TODO: externalize these parameters
        self.image_width = 512
        self.image_length = 512

        #
        self.max_n_seconds_session = max_n_seconds_session

        # number of frames to run BMI for
        self.n_frames = n_frames_session # OLD WAY OF COMPUTING max_n_seconds_session*sampleRate_2P

        # TODO: why do we have 2 of these variables?
        self.n_frames_to_be_acquired = self.n_frames   # Number of frames from BScope

        #

        # start the ttl frame counter at 0
        self.ttl_computed = 0

        # initialize all arrays to be used, mostly to save data after BMI run
        self.initialize_data_arrays()

        # initalize reward contidions based on ~15mins of pre BMI recorded data
        self.initialize_reward_conditions_and_parameters()

        # initialize rewards counter
        self.initialize_reward_times()

        # intiatlie n_ttl
        self.initialize_n_ttl()

        # initialize tone state
        self.initialize_tone_state()

        # initialize the water reward memory variable
        self.initialize_water_reward()

        #
        self.initialize_termination_flag()

        #
        #self.initialize_live_frame_shared_memory()

        # start reading the ttl pulses from the 2p scope
        self.initialize_bscope_ttl_pulse_reader()

        # this gets read simultaneously with all other TTL/BNC channels now
        #self.initalize_lick_detector_reader()

        # keeps track of lick values
        self.lick_detector_abstime = [] #np.zeros((self.n_frames,2),dtype=np.float32)

        #
        self.initialize_rotary_encoder()

        #
        self.initialize_video_frame()

    #
    def initialize_video_frame(self):
        ''' shared variable that keeps current video camera frame in memeory for
        '''

        # make a numpy array to hold the rois_traces
        print ("self.video width: ", self.video_width, " length: ", self.video_length)
        aa = np.zeros((1,self.video_width,self.video_length), dtype=np.uint8)
        self.shmem_live_video_frame = shared_memory.SharedMemory(create=True,
                                                             size=aa.nbytes)

        #
        self.live_video_frame = np.ndarray(aa.shape,
                                        dtype=aa.dtype,
                                        buffer=self.shmem_live_video_frame.buf)

        #
        self.live_video_frame[:] = aa[:]

    #
    def initialize_rotary_encoder(self):

        # this keeps track of the rotary encoder wheel rotations
        self.rotary_encoder1_abstime = []
        self.rotary_encoder2_abstime = []
    #
    def initialize_termination_flag(self):

        '''
            Signal that is shared with all cores to indicate termination of BMI
            - 0: keep running
            - 1: end all processing
        '''

        # make a numpy array to hold the rois_traces
        aa = np.zeros(1, dtype=np.int64)
        self.shmem_termination_flag = shared_memory.SharedMemory(create=True,
                                                                 size=aa.nbytes)

        #
        self.termination_flag = np.ndarray(aa.shape,
                                     dtype=aa.dtype,
                                     buffer=self.shmem_termination_flag.buf)

        #
        self.termination_flag[:] = aa[:]

    #
    def initialize_water_reward(self):

        '''
            This variable keeps track of the value of the water spout
            - 0: no water reward
            - 1: water reward
            Note: the duration and timing and lockouts of water rewards are controlled by the
            waterreward class

        '''

        # make a numpy array to hold the rois_traces
        aa = np.zeros(1,dtype=np.float32)
        self.shmem_water_reward = shared_memory.SharedMemory(create=True,
                                                              size=aa.nbytes)

        #
        self.water_reward = np.ndarray(aa.shape,
                                         dtype=aa.dtype,
                                         buffer=self.shmem_water_reward.buf)

        #
        self.water_reward[:] = aa[:]

    #
    def initialize_tone_state(self):

        '''
            This variable keeps track of the tone value computed by the TONE class
            - technically it doesn't have to be initialized here, but we do it for simplicity to easier
              share it with the plotter class (BMI class doesn't need it for now)

        '''

        # make a numpy array to hold the rois_traces
        aa = np.zeros(1,dtype=np.float32)
        self.shmem_tone_state = shared_memory.SharedMemory(create=True,
                                                       size=aa.nbytes)

        #
        self.tone_state = np.ndarray(aa.shape,
                                         dtype=aa.dtype,
                                         buffer=self.shmem_tone_state.buf)

        #
        self.tone_state [:] = aa[:]

    #
    def initialize_reward_conditions_and_parameters(self):

        ''' This function should load the parameters computed by another script
            Input: filenames of several parameters:
                - minimum frequency (e.g. 1000Hz)
                - maximum frequency (e.g. 180000Hz)
                - threshold_low (e.g. -1.0)
                - threshold_high (e.g. +1.0)
                ... others to add
        '''

        data = np.load(self.fname_rois_pixels_thresholds, allow_pickle=True)

        #
        self.low_threshold = data['low_threshold']
        self.high_threshold = data['high_threshold']

        #
        self.post_reward_lockout = data['post_reward_lockout']

        #
        self.rois_smooth_window = data['rois_smooth_window']

        #
        self.smooth_diff_function_flag = data['smooth_diff_function_flag']

        # set the last reward time in ttl pulses (might need something better here)
        self.initialize_last_reward_ttl()

        # reward lockout time after a positive reward - in seconds
        self.received_random_reward_lockout = 5
        print (">>>>>>>>>>>> POST-RANDOM-REWARD LOCKOUT: ", self.received_random_reward_lockout, "sec")

        # counter that track time after last reward
        self.initialize_reward_lockout_counter()

        # the amount of time the mouse has to try and receive a reward - in seconds
        self.max_reward_window = 30

        #
        self.template = data['calibration_template']

    #
    def initialize_pbar(self):
        self.pbar = tqdm.tqdm(total=self.n_frames_to_be_acquired,
                              desc='% complete',
                              position=0,
                              leave=True,
                              ascii=True)  # Init pbar

    #
    def initialize_data_arrays(self):
        ''' TODO: check to make sure all the possible data being recorded is being saved

        '''
        #
        self.ttl_values = []            # array to hold ttl data being read
        self.ttl_n_computed = []        # number of ttl pulses computed based on time elapsed
        self.ttl_n_detected = []        # number of ttl pulses detected based on TTL from NI board
        self.inter_ttl_time = []        # computed time between each detected TTL pluse
        self.abs_times = []             # Keep of every time TTL is read... important!
                                        # loop;   might be useful for debugging later on kernel interuptions etc.
        self.ttl_times = []             # ttl times to be saved
        self.previous_trigger=0         # time of the previous TTL trigger to be used to determine if next trigger etc
        self.prev_max = 0               # TTL pulse previous read max value
        self.prev_min = 0               # TTL pulse previous read min value
        self.ttl_voltages = []          # ttl_voltages

        #self.initialize_n_ttl()
        self.rewarded_times_abs = []

    #
    def initialize_last_reward_ttl(self):
        ''' This variable keeps track of the last received reward or missed reward time
            - it is used to reset certain conditions
            TODO: may wish to have separate clocks for received reward vs. missed reward time.
        '''

        # make a numpy array to hold the rois_traces
        aa = np.zeros(1, dtype=np.int64)
        self.shmem_last_reward_ttl = shared_memory.SharedMemory(create=True,
                                                          size=aa.nbytes)

        #
        self.last_reward_ttl = np.ndarray(aa.shape,
                                    dtype=aa.dtype,
                                    buffer=self.shmem_last_reward_ttl.buf)

        #
        self.last_reward_ttl[0] = -1

    #
    def initialize_reward_lockout_counter(self):

        '''  This value keeps track of a counter that resets every time there's a reward
            - or a missed reward to prevent rewards during the period

        '''

        # make a numpy array to hold the rois_traces
        aa = np.zeros(1, dtype=np.int64)
        self.shmem_reward_lockout = shared_memory.SharedMemory(create=True,
                                                          size=aa.nbytes)

        #
        self.random_reward_lockout_counter = np.ndarray(aa.shape,
                                    dtype=aa.dtype,
                                    buffer=self.shmem_reward_lockout.buf)

        #
        self.random_reward_lockout_counter[:] = aa[:]

    #
    def initialize_reward_times(self):

        ''' shared variable that tracks # of rewards

        '''

        # an array to hold the reward times
        aa = np.zeros((2,1000), dtype=np.int64)-1
        self.shmem_reward_times = shared_memory.SharedMemory(create=True,
                                                             size=aa.nbytes)

        #
        self.reward_times = np.ndarray(aa.shape,
                                dtype=aa.dtype,
                                buffer=self.shmem_reward_times.buf)

        #
        self.reward_times[:] = aa[:]

    #
    def initialize_live_frame_shared_memory(self):

        ''' shared variable that keeps current image in memeory for plotter to visualize
            NOTE: We actually need 2 independent ones (for now) to send to plotter
            and motion detection algorithm independently.
        '''

        # make a numpy array to hold the rois_traces
        aa = np.zeros((1,512,512), dtype=np.uint16)
        self.shmem_live_frame_plotter = shared_memory.SharedMemory(create=True,
                                                             size=aa.nbytes)

        #
        self.live_frame_plotter = np.ndarray(aa.shape,
                                        dtype=aa.dtype,
                                        buffer=self.shmem_live_frame_plotter.buf)

        #
        self.live_frame_plotter[:] = aa[:]

        # Also initialize a live frame for the
        # make a numpy array to hold the rois_traces
        aa = np.zeros((1,512,512), dtype=np.uint16)
        self.shmem_live_frame_motion_detector = shared_memory.SharedMemory(create=True,
                                                             size=aa.nbytes)

        #
        self.live_frame_motion_detector = np.ndarray(aa.shape,
                                        dtype=aa.dtype,
                                        buffer=self.shmem_live_frame_motion_detector.buf)

        #
        self.live_frame_motion_detector[:] = aa[:]
        #

    #
    def initialize_n_ttl(self):

        ''' This variable keeps track of how many frames the BMI has detected
            - it is used to trigger the search for the next imaging frame
            - it is also shared with the plotting algorithm

            TODO:
            - we may actually be able to run the BMI without TTL signals from the microscope
            - that is, we can just actively search (e.g. every 10ms) the raw imaging data to see if any
              new data has been written and take the latest image as proof of this
            - this "nuclear" option could be implemented in systems that are more complex to work with
              or that dont' have easily accessible TTL pulses - but are very good at writing to disk

            - NOTE: this option should probably be implemented using a RAM-drive where the imaging data
              is saved to a ram disk to avoid brekaing spindisks/SSDs

        '''

        # make a numpy array to hold the rois_traces
        aa = np.zeros(1, dtype=np.int64)
        self.shmem_n_ttl = shared_memory.SharedMemory(create=True,
                                                      size=aa.nbytes)

        #
        self.n_ttl = np.ndarray(aa.shape,
                                dtype=aa.dtype,
                                buffer=self.shmem_n_ttl.buf)
        self.n_ttl[:] = aa[:]

        #
        #print(" ttl counter initialized: ", self.n_ttl, self.shmem_n_ttl.name)

    #
    def run_BMI(self):

        #
        print('Running BMI - Calibrtion (ctrl-c to stop)')

        #
        self.now = time.perf_counter() #time.perf_counter_ns()/1E9
        self.previous_trigger = time.perf_counter()-2 # set the previous tirgger 2 sec prior to start

        #
        self.initialize_pbar()

        # abssolute start time
        self.start_time_acquisition = time.time()

        # start recording and acquisition
        # count number of frames; but probably safer to just count time;
        # TODO: merge ttl pulse counting and time tracking into a single while statement
        while self.ttl_computed < self.n_frames_to_be_acquired - 1:

            # read next bscope ttl pulse
            self.read_bscope_ttl()

            # check of ttl pulse when from high ~5 to low ~0
            if self.min_<1 and self.prev_max>=1:

                # runs the bmi code whenever imaging frame is completed
                self.bmi_update()

                # update trigger time
                self.previous_trigger = self.now

                #
                self.pbar.update(n=1)

            #
            self.prev_min = self.min_
            self.prev_max = self.max_

            # exit OPTION 2 check if estimated recording time + 2mins have been completed
            if (time.time() - self.start_time_acquisition)>self.max_n_seconds_session:
                print ("Duration of BMI loop: ", time.time() - self.start_time_acquisition, 'sec',
                       "  , total requested: ", self.max_n_seconds_session)
                break

        # save all data acquried during recording
        # TODO: try to save this on the fly if possible to avoid loosing data during crashes
        self.save_data()

        #
        self.bscope_ttl_task.stop()
        self.bscope_ttl_task.close()

        #

    #
    # def save_lick_detector_ttl(self):
    #
    #     # get current lick detector state from NI card output port
    #     #self.lick_detector_ttl_value = self.lick_detector_ttl_task.read(number_of_samples_per_channel=self.ttl_pts)
    #
    #     # this saves a triple: [lick_detector_ttl_value,
    #     #                       self.now,
    #     #                       n_ttl_counter]
    #     #print ("lick detector val: ", self.lick_detector_ttl_value," self.now: ", self.now)
    #
    #     # saves updated values
    #     self.lick_detector_value_abstime_nttl[self.n_ttl[0]] = self.lick_detector_ttl_value, self.now

    #
    def read_bscope_ttl(self):

        # get current bscope ttl pulse value from NI card output port
        #read_values = np.array(self.bscope_ttl_task.read(number_of_samples_per_channel=self.ttl_pts)).squeeze()
        read_values = self.bscope_ttl_task.read(number_of_samples_per_channel=self.ttl_pts)

        #print ("READ VALUES: ", read_values)

        # ttl bscope value
        ttl_value = read_values[0]#.copy()

        # lick detector value
        # TODO: IMPORTANT to read out both encoder and lick-detector in high res rate as 30Hz may miss
        #       important information
        self.lick_detector_abstime.append(read_values[1])

        # rotary encoder
        self.rotary_encoder1_abstime.append(read_values[2])
        self.rotary_encoder2_abstime.append(read_values[3])

        #  leave these in just in case we end up reading at higher bit rates and multiple samples at a atime
        # TODO: these might be redundant, not clear
        self.min_ = np.min(ttl_value)
        self.max_ = np.max(ttl_value)
        self.ttl_voltages.append(ttl_value)
        #

        # get time of ttl pulse
        self.now = time.perf_counter() #perf_counter_ns()/1E9

        # this helps us figure out how fast this loop runs
        # TODO: we may want to introduce a delay of 5ms or so so we don't constanly read TTL pulses
        #     but this is probably not necessary as the NIDAQMX package was made to be pinged a lot
        self.abs_times.append(self.now)

    #
    def close(self):

        #
        print (" ... closing BMI, # of ttl pulses processed: ", self.n_ttl)
        #
        print (" ... SENDING TERMIANTION FLAG SIGNAL TO ALL PROCESSES ...")
        self.termination_flag[0] = 1

        #
        print("... EXITING BMI CLASS...")

        # give the rest of the modules a few sec to complete
        time.sleep(2)

    #
    # def initalize_lick_detector_reader(self):
    #   #
    #     if self.simulation_mode_lick_detector == True:
    #         self.lick_detector_ttl_task = Simulation(self.fname_ttl)
    #     else:
    #
    #         #
    #         self.lick_detector_ttl_task = nidaqmx.Task('lick_detector')
    #         #print ("iniitlied bmi online")
    #         # set TTL pulse reader from 2p system
    #         self.lick_detector_ttl_task.ai_channels.add_ai_voltage_chan("Dev3/ai1",
    #                                                       terminal_config=TerminalConfiguration.NRSE)
    #
    #         #
    #         self.lick_detector_ttl_task.timing.cfg_samp_clk_timing(self.sampleRate_NI,
    #                                                  # samps_per_chan=pointsToPlot*2,
    #                                                  sample_mode=AcquisitionType.CONTINUOUS )
    #
    #         # start the TTL reader (not required in simulation mode)
    #         self.lick_detector_ttl_task.start()
    #
    #     # list that holds all the lick detector infor + meta data
    #     self.lick_detector_value_abstime_nttl = np.zeros((self.n_frames,2),dtype=np.float32)

    #
    def initialize_bscope_ttl_pulse_reader(self):

        #
        if self.simulation_mode_bmi == True:
            self.bscope_ttl_task = Simulation(self.fname_ttl)
        else:


            #
            self.bscope_ttl_task = nidaqmx.Task('ttl_reader')
            # set TTL pulse reader from 2p system
            # add ttl pulse channel from bscope
            self.bscope_ttl_task.ai_channels.add_ai_voltage_chan("Dev3/ai0:4",
                                                          terminal_config=TerminalConfiguration.NRSE)

            # add lick detector channel
            #self.bscope_ttl_task.ai_channels.add_ai_voltage_chan("Dev3/ai1",
            #                                              terminal_config=TerminalConfiguration.NRSE)
            #c
            self.bscope_ttl_task.timing.cfg_samp_clk_timing(self.sampleRate_NI,
                                                     # samps_per_chan=pointsToPlot*2,
                                                     sample_mode=AcquisitionType.CONTINUOUS)

            # start the TTL reader (not required in simulation mode)
            self.bscope_ttl_task.start()

        # list to add to when reading values
        self.bscope_read_values = np.zeros(2,dtype=np.float32)

    #
    def bmi_update(self):

        #
        self.compute_frame_number()

        #
        self.load_current_frame_and_apply_drift_correction()

        # check for reward condition:
        self.check_reward_condition_random()

        # decrease any potential reward lockout counter
        self.received_random_reward_lockout[0] -= 1

        # save meta data
        self.ttl_n_computed.append(self.ttl_computed)
        self.ttl_n_detected.append(self.n_ttl)
        self.ttl_times.append(self.now)

        #
        self.n_ttl+=1
    #

    def smooth_rois(self):

        ''' Function that smooths the raw roi traces;
            - this is required for both visualization but also ensemble computations as
              we do not run algorithms on noisy raw data directly

        '''

        # if we made threhsods using smoothing, then need to run them on data also
        # TODO:  IMPORTANT: implement the identical algorithm used in the calibration step to compute
        #        this step; currently only the smoothing step is shared; need to share DFF0 computation also
        # wait a few seconds until get enough data to smooth out
        if self.smooth_diff_function_flag and self.n_ttl[0]>self.rois_smooth_window:

            # loop over each cell
            for p in range(self.rois_traces_raw.shape[0]):
                #
                temp = self.rois_traces_raw[p,self.n_ttl[0]-self.rois_smooth_window:self.n_ttl[0]]

                # There are two options for deterneding and computing a DFF0
                # option 1: use the calibration time roi_f0s
                #  Note: this is risky to do:
                #          - sometimes there is signficant drift which we don't correct for (yet!)
                # WE FIXED THIS NOW
                #if False or self.n_ttl[0]<self.n_ttl_to_start_applying_DFF0_computation:
                if True:
                    temp = (temp - self.roi_f0s[p])/self.roi_f0s[p]

                # Recompute baseline dynamically to ensure alignemtn of data
                # Note: this is also risky as this means the thresholds computed in the calibration step
                #        might not be completely accurate any longer
                #
                else:
                    # so here we feed current chunk of data going back n frames
                    #  plus the refrenc trace which should be the last n frames of raw data; usually take at least 30 seconds
                    _, temp = compute_dff0_with_reference(temp,
                                self.rois_traces_raw[p,self.n_ttl[0]-self.n_ttl_to_start_applying_DFF0_computation:
                                                        self.n_ttl[0]]
                                                        )

                #
                self.rois_traces_smooth[p,self.n_ttl[0]] = smooth_ca_time_series(temp)

        else:
            #
            self.rois_traces_smooth[:,self.n_ttl[0]] = self.rois_traces_raw[:,self.n_ttl[0]]

    #
    def compute_frame_number(self):

        ''' Function that computes which frame to read from [Ca] file based on how many TTL pulses
            were generated via passage of time (using start time, current time and sample rate.  The
            goal is to figure out which imaging frame we should load next from the memmory map of
            the [Ca] data
            - alternative is to just count TTL pulses and index into those (this variable is already
            available in this lise: self.ttl_n_computed)
            - however, just counting this list may be incorrect due to possible operating system lockoups
            or kernel issues;
            - TODO: still should test which if these methods gives more accurate location in the imaging
            stack

            NOTE #1:
            In simulation mode, the TTL pulses computed will be incorrect because we will
            be reading the TTL pulses from a file and this will be superfast compared to waiting for
            them to be generated by the 2P system;
            - so for simulation mode we have to bypass the "computed ttl" method and just assume that
            everytime we enter into this function we are ok to read the next frame

            NOTE #2:
            Function also creates a memory map file which is used for storying the imaging frames
            as they are read from the hard disk. This method is great to limit memory use, but as we
            read more frames from the disk, they do get stored in memory up to and until the entire
            file is loaded into memory
            - beause some the of the imaging datasets can be 30GB or more we might run out of
            memory, unless we have 128GB or much more ram
            - TODO: we will implement a method that destroys/deletes the memory map and starts over
            perhaps every 10 minutes or so.  Restarting the memmap seems to take Order (1ms) so it
            should not be an issue, but we do have to test it to ensure we are freeing up memory

            Input:
            - self.ttl_n_computed = contains the number of ttl pulses computed based on passage of time
            - self.now = contains the realtime of the last read ttl pulse (usually in millsec; to check)
            - self.fname_fluorescence = path of the fluorescence file
            - self.n_frames_to_be_acquired = total number of imaging frames for the session
            - self.sampleRate_2P = samplerate of the 2P microscope, usually 30FPS

            Output:
            - self.newfp = this is the memory map that holds all our calcium data
            - self.ttl_computed = the frame location based on passage of time; we use this to reach into
            the memory map [Ca] file and grab a specific frame

        '''

        # initialize raw data arrays
        if len(self.ttl_n_computed)==0:

            #
            if self.read_data_flag:
                #import mmap
                #ss = time.time()
                print ("  setting up memory map: shape: ", (self.n_frames_to_be_acquired,512,512))

                if True:
                    self.newfp = np.memmap(self.fname_fluorescence,
                                           dtype='uint16',
                                           mode='r',
                                           shape=self.n_frames_to_be_acquired*512*512)

                # if False:
                #     fp = open(self.fname_fluorescence, "r")
                #     byts = self.n_frames_to_be_acquired*self.image_width*self.image_length
                #
                #     self.newfp = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
                #     # print ("OPTION @@@@@@@@@@@@@@@@: ", self.newfp)
                #
                #     #
                #     mview = memoryview(self.newfp)
                #     print (" memroy view: ", mview)
                #     self.newfp = np.asarray(mview).reshape(self.n_frames_to_be_acquired,512,512)
                #     print (" final array view: ", self.newfp.shape)


                #
                self.newfp = self.newfp.reshape(self.n_frames_to_be_acquired,512,512)

            # reset start time: requird becaues we start the BMI a few seconds before the BScope
            self.start=self.now

        # after arrays initialized
        else:

            # in simulation mode we just assume that we have correctly dected a TTL pulse and add 1 extra
            #   ttl pulse to the stack
            if self.simulation_mode_bmi==True:
                self.ttl_computed = self.n_ttl+1  # move to next ttl.
                time.sleep(self.sleep_time_sec)

            else:
                time_passed = self.now-self.start
                self.ttl_computed = round((self.now-self.start)*self.sampleRate_2P)

                #
                if self.verbose:
                    print (" time passed: ", time_passed, "   bmi_update self.ttl_computed: ", self.ttl_computed)

    #
    def trigger_random_reward(self):

        # generate water reward only if we are not in a reward lockout state
        print (" ****giving random reward at time: ", self.n_ttl)

        #
        self.water_reward[0] = 1

        # start a counter that prevents another reward for some time
        self.random_reward_lockout_counter[0] = self.received_random_reward_lockout * self.sampleRate_2P

    #
    def post_reward_state(self):

        # disable tone playback;
        self.tone_off()

    #
    def check_baseline_condition(self):

        # check if ensemble activity back to baseline; e.g. within 1 x of std
        # if abs(ensemble...[ensbmel_ID1] - ensbmel..[ensemble_ID2])< std x 1?
        #     return True
        # else:
        #     return False

        pass

    #
    def check_reward_condition_random(self):

        ''' We check if reward condition was reached
        '''

        # draw random rewards so that the mouse is rewarded about once every minute
		if np.random.rand()< (1/(30*60)) and self.random_reward_lockout_counter[0]<1:
            #
            print (" reached RANDOM reward conition: ")

            # search for the first empty slot in the reward times list
            # TODO: this is a poor way to save these values;
            # TODO: have a shared memory variable separate from one that keeps track of the times
            for k in range(self.reward_times.shape[1]):
                if self.reward_times[1, k] == -1:
                    self.reward_times[1, k] = self.n_ttl[0]  # save current reward time
                    break
            #
            self.rewarded_times_abs.append([1, self.n_ttl[0], self.abs_times])

            # reset last reward time to current time
            self.last_reward_ttl[0] = self.n_ttl[0]

            #
            self.trigger_random_reward()

            # decouple tone etc. from feedback
            self.post_reward_state()

    #
    def load_current_frame_and_apply_drift_correction(self):

        ''' This is an overkill function which cheks whether the n_ttl detected from TTL pulses
            doe sindeed have values in it, if so it

            TODO: either drop this first check - or implement it in full - which means going forward in time
                  until we get zeros in the data

        '''

        #
        if self.verbose:
            print ("self.ttl_computed: ", self.ttl_computed)
            print("  detected frame #: ", self.n_ttl,
               " computed_frame : ", self.ttl_computed)

        # search the very first ROI in time from previous frame to future frames until we get a non-zero pixel values;
        #  then we set the time i.e. n_ttl
        for z in range(-1,self.n_frames_search_forward,1):

            # TODO ;could just check any part of the FOV to see if there is non zero values
            roi_sum0 = self.newfp[self.n_ttl[0]+z][self.rois_pixels[0]].sum()

            #
            if roi_sum0 != 0:

                # TODO: reset the n_ttl value here - check that this is safe!!!
                # self.n_ttl[0] = self.n_ttl[0]+z

                break


        #
        self.live_frame_local = self.newfp[self.n_ttl[0]+z].copy()

        # there is no motion correction during calibration step
        self.live_frame_local_drift_corrected = self.live_frame_local.copy()

        # this is the frame that the plotting function sees
        self.live_frame_plotter[0] = self.live_frame_local_drift_corrected.copy()

    #
    def tone_off(self):

        # turn toneplayback off
        #  freq = 0
        #  np.save(self.fname_freq,freq)

        # need to also pass the time out counter
        #
        # pass a zero neural state vector??!?!


        pass

    #
    def save_data(self):

        '''  TO FILL OUT
             need better description of variables
             TODO:  other variables we might want to save including
             - tone frequencies, or tone state of the speaker
             - the camera frames/informatin
             - IR light info
             - EMG data
             - lick detector information
             - treadmill/ball walking distance

        '''
        print("...Saving BMI meta/data...")

        for k in range(self.reward_times.shape[1]):
            if self.reward_times[1,k]==-1:
                break
        print (" .. # of rewards: ", k-1, ", water volume dispensed (@ 20uL per reward): ",(k-1)*0.020, "mL")

        #
        np.savez(self.fname_save_data,
                 ttl_voltages = self.ttl_voltages,
                 ttl_n_computed = self.ttl_n_computed,
                 ttl_n_detected = self.ttl_n_detected,
                 abs_times = self.abs_times,
                 ttl_times = self.ttl_times,
                 rois_pixels = np.hstack(self.rois_pixels),
                 rois_traces_raw = np.array(self.rois_traces_raw,dtype='object'),
                 rois_traces_smooth = np.array(self.rois_traces_smooth,dtype='object'),
                 reward_times = self.reward_times,
                 rewarded_times_abs = self.rewarded_times_abs,
                 ensemble_activity = self.ensemble_activity,
                 ensemble_diff_array = self.ensemble_diff_array,
                 received_reward_lockout = self.received_reward_lockout,
                 max_reward_window = self.max_reward_window,
                 missed_reward_lockout = self.missed_reward_lockout,

                 sampleRate_NI = self.sampleRate_NI,
                 ttl_pts = self.ttl_pts,
                 sampleRate_2P = self.sampleRate_2P,
                 image_width = self.image_width,
                 image_length = self.image_length ,
                 max_n_seconds_session = self.max_n_seconds_session,

                 n_frames = self.n_frames,
                 n_frames_to_be_acquired = self.n_frames_to_be_acquired,                #
                 rois_smooth_window = self.rois_smooth_window,
                 n_ttl_to_start_applying_DFF0_computation = self.n_ttl_to_start_applying_DFF0_computation,
                 n_frames_search_forward = self.n_frames_search_forward,
                 drift_array = self.drift_array,
                 template = self.template,
                 lick_detector_abstime = self.lick_detector_abstime,
                 rotary_encoder1_abstime = self.rotary_encoder1_abstime,
                 rotary_encoder2_abstime = self.rotary_encoder2_abstime,
                 )




##############################
class CalibrationTools(object):
	
    #
	def __init__(self, fname):

		#
		self.fname = fname

		# 
		self.binarize_thresh =.05
		self.sigma = .5
		self.order = 0
		self.n_smooth_steps = 1

		#
		#data = np.memmap(self.fname, dtype='uint16', mode='r+')
		#data = np.memmap(self.fname, dtype='uint16', mode='r')
		data = np.fromfile(self.fname, dtype='uint16')
		self.data = data.reshape(-1,512,512)
		print ("memmap : ", self.data.shape)
		
	#	
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
				  )
		plt.show()
		
	
		return img

		#
	def make_max_proj_map(self):

		#
		data_sparse = self.data[::self.subsample]
		print("data into analysis: ", data_sparse.shape)

		# filter once to remove much of the white noise
		if False:
			sigma = 1
			order = 0
			print(" gaussian filter width: ", sigma, ", order: ", order)
			data_sparse = scipy.ndimage.gaussian_filter(data_sparse,
														sigma,
														order)
			print("done filtering... (TO CHECK which axis are we filtering!!)")

		maxproj = np.max(data_sparse, axis=0)

		#std = np.std(data_sparse, axis=0)

		return maxproj

	#
	def make_std_map(self):

		#
		data_sparse = self.data[::self.subsample]
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
			kernel = [7,1,1]   # filter only across time
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
				
		std = np.std(data_sparse,axis=0)

		return std

	def plot_std_map(self, std):
		# 
		temp = std.copy()
		print ("staring computing std...")
		print ("done computing std...")
		#
		idx = np.where(temp<self.vmin)
		temp[idx]=0
		idx = np.where(temp>self.vmax)
		temp[idx]=self.vmax

		# 
		plt.figure()
		plt.imshow(temp,
				  )
		plt.show()
		
		return temp

	def area_inside_convex_hull(self, pts):
		lines = np.hstack([pts,np.roll(pts,-1,axis=0)])
		area = 0.5*abs(sum(x1*y2-x2*y1 for x1,y1,x2,y2 in lines))
		return area

	def binarize_data(self, img, thresh):
		
		#thresh = .15
		idx1 = np.where(img>thresh)
		idx2 = np.where(img<=thresh)
		img[idx1]=1
		img[idx2]=0
			
		return img

	#
	def find_roi_boundaries(self, data):

		#
		image = data.copy()

		for k in trange(self.n_smooth_steps, desc='gaussian filtering data'):
			image = scipy.ndimage.gaussian_filter(image, 
												  self.sigma, 
												  self.order)

		image = image.astype('int32')
        
		#
		image = self.binarize_data(image, self.vmin)

		#
		image = image.astype('int32')
						
		# run watershed segmentation
		distance = ndi.distance_transform_edt(image)
		coords = peak_local_max(distance, 
								footprint=np.ones((1, 1)), 
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
		min_size = self.min_size_roi
		max_size = self.max_size_roi
		roi_centres = []
		indexes = []
		for k in tqdm(np.unique(labels), desc='looping over cells'):
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
	def show_contour_map(self, std_map, indexes, cell_ids, fig=False):
		
		if fig is True:
			plt.figure()
			
		plt.imshow(std_map,
				   vmin = self.vmin*0.7,
				   vmax = self.vmax*1.3)
		#
		for p in cell_ids:
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

	#	def show_contour_map(self, std_map, indexes, cell_ids, fig=None):

	def compute_traces2(self, std_map, cell_ids=None, fig=None):
		''' Same as below but visualize every single frame
		'''
		
		if cell_ids is None:
			cell_ids = np.arange(len(self.indexes))
		print ("plotting cells: ", cell_ids)
           
		data = np.memmap(self.fname, dtype='uint16', mode='r')
		data = data.reshape(-1,512,512)
		print ("memmap : ", data.shape)
			
		#####################################################
		plt.figure()
		ax = plt.subplot(121)
		ax.tick_params(axis='both', which='both', labelsize=20)
		plt.ylabel("Neuron ID ", fontsize=20)

		#
		roi_traces = []
		for k in range(len(cell_ids)):
			roi_traces.append([])
		
		# loop over each frame
		for p in trange(0, data.shape[0], self.trace_subsample):

			# grab frame
			frame = data[p]

			# loop over ROIS
			ctr = 0
			for k in cell_ids:
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
				roi_traces[ctr].append(temp)
				ctr+=1
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
		labels = cell_ids
		labels_old = np.arange(0,ctr*self.scale,self.scale)

		#
		plt.yticks(labels_old, labels, fontsize=10)
		plt.xlabel("Time (sec)",fontsize=20)
        
        # 
		ax2=plt.subplot(122)
		new_plot = False
		print (cell_ids)
		self.show_contour_map(std_map,
							  self.indexes,
							  cell_ids, new_plot)

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

	#
	def find_reward_thresholds_absolute(self, normalize_peaks=True):
		'''  Computes the aboslute |E1-E2|
		     and rewards anytime the ensembel goes above this value
		     Note that self.roi_traces contains only the 4 neurons from the ensembes selected
		     now
		     - TODO: change this for the high and high_low functions also
		
		'''
		
		# TODO: refactor this part and send it to the BMI session code
		
		# run smoothing on each ensemble
		if self.smooth_diff_function_flag:

			# ensemble #1
			for p in range(2):
				smooth = np.zeros(self.roi_traces[p].shape)
				for k in trange(self.rois_smooth_window, self.roi_traces[p].shape[0], 1):
					smooth[k] = self.smooth_ca_time_series(self.roi_traces[p][k - self.rois_smooth_window:k])
				#
				self.roi_traces[p] = smooth

			# ensemble #2
			for p in range(2,4,1):
				smooth = np.zeros(self.roi_traces[p].shape)
				for k in trange(self.rois_smooth_window, self.roi_traces[p].shape[0], 1):
					smooth[k] = self.smooth_ca_time_series(self.roi_traces[p][k - self.rois_smooth_window:k])
				#
				self.roi_traces[p] = smooth
			
		#
		self.roi_f0s = []
		self.dff0 = []
		for k in range(len(self.roi_traces)):
			f0, dff0 = self.compute_dff0(self.roi_traces[k])
			self.roi_f0s.append(f0)
			self.dff0.append(dff0)

		# compute ensembles using the smooth + baseline removed values
		E1 = self.dff0[0] - self.dff0[1]
		E2 = self.dff0[2] - self.dff0[3]

		# initialize the max and min values
		max_E1 = np.max(E1)
		max_E2 = np.max(E2)

		print ("TODO: Normalize the peaks of the 2 Ensembles so the mice don't learn to use one esnemble against the other!!!!")
		low = np.nan
		high = min(max_E1, max_E2)*3

		print("low, high: ", low, high)
		# difference between ensemble
		diff = np.abs(E1 - E2)

		#
		self.n_sec_recording = int(diff.shape[0] / self.sample_rate)
		self.n_rewards_random = self.n_sec_recording // self.sample_rate
		self.n_rewards_default = int(self.n_rewards_random*0.3)
		print("nsec recording: ", self.n_sec_recording,
			  "max # of random rewards (i.e. every 30sec) ", self.n_rewards_random,
			  "# of rewards for 30% of the time: ", self.n_rewards_default)

		# loop over time series decreasing the rewards until we hit the random #
		n_rewards = 0
		stepper = 0.95
		while n_rewards < self.n_rewards_default:

			# run inside while loop for eveyr setting of low and high until we hit
			#   exact number of random rewards
			k = 0
			n_rewards = 0
			reward_times = []
			while k < diff.shape[0]:

				temp_diff = diff[k]

				if temp_diff >= high:
					# high reward state reached
					n_rewards += 1
					reward_times.append([k, 1])
					k += int(self.post_reward_lockout * self.sample_rate)
				else:
					k += 1

			# print ("Reard times: ", reward_times)
			# check exit condition otherwise decrase thresholds
			#if len(reward_times) > 1:
			# 	rewarded_times = np.vstack(reward_times)
			#	high *= stepper
			#else:
			high *= stepper

		print("updated rwards #: ", n_rewards, low, high)

		self.reward_times = np.vstack(reward_times)

		self.low = np.nan
		self.high = high
		self.E1 = E1
		self.E2 = E2
		self.diff = diff

		

	#
	def find_reward_thresholds_high(self):

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

		ymaxes = np.max(np.abs(self.diff))
		#
		for k in range(len(self.reward_times)):
			temp = self.reward_times[k]

			if temp[1]==0:
				plt.plot([t[temp[0]], t[temp[0]]], [-ymaxes,ymaxes], '--', c='red')
			else:
				plt.plot([t[temp[0]], t[temp[0]]], [-ymaxes,ymaxes], '--', c='blue')

		# replot two random rewards just to make nice legend
		idx1 = np.where(self.reward_times[:,1]==0)[0].shape[0]
		idx2 = np.where(self.reward_times[:,1]==1)[0].shape[0]

		plt.plot([t[temp[0]], t[temp[0]]], [-ymaxes,ymaxes], '--', c='red', label='E1 rewarded # '+str(idx1),)
		plt.plot([t[temp[0]], t[temp[0]]], [-ymaxes,ymaxes], '--', c='blue', label='E2 rewarded # '+str(idx2),)
		plt.legend()

		plt.title("Rec duration: " + str(int(t[-1])) + " sec "+
				  "\n expected # of random rewards: "+str(int(t[-1]/30))+
				  "\n actual # of provided rewards: "+str(self.n_rewards_default))
		plt.show()




def get_binary_std_map(std,
					   vmax=1500):

    #
    fig = plt.figure()

    sigma = 1.5

    #
    #ax=plt.subplot(111)
    plt.title("std map")
    live_image_vmin = 0
    live_image_vmax = vmax

    #
    image_obj = plt.imshow(std,
              vmin=live_image_vmin,
              vmax=live_image_vmax,
                           interpolation='none'
              )

    axmin = fig.add_axes([0.05, 0.90, 0.1, 0.03])
    axmax  = fig.add_axes([0.05, 0.93, 0.1, 0.03])

    #
    smin = Slider(axmin, 'Min', 0, live_image_vmax, valinit=live_image_vmin)
    smax = Slider(axmax, 'Max', 0, live_image_vmax, valinit=live_image_vmax)

    #
    def update_clim1(val):
        if smin.val<smax.val:
            image_obj.set_clim([smin.val,
                                smax.val])
            res = scipy.ndimage.gaussian_filter(std, sigma=sigma)
            image_obj.set_data(res)
        else:
            smin.val = smax.val-1

    #
    smin.on_changed(update_clim1)
    smax.on_changed(update_clim1)

    #
    #plt.show(block=True)

    return smin, smax


def get_img_std(smin, smax, std_map,bmi_c):
    #
    print ("max proj values (vmin, vmax): ", smin.val, smax.val)

    img_std = std_map.copy()
    idx = np.where(img_std<smin.val)
    idx2 = np.where(img_std>=smin.val)

    img_std[idx] = 0
    img_std[idx2] = 1
    sigma = 1.5
    img_std = scipy.ndimage.gaussian_filter(img_std, sigma=sigma)

    bmi_c.vmin = smin.val; bmi_c.vmax = smax.val
    
    return bmi_c, img_std
    
    
#
def get_rois_stardist2d(img):
    # prints a list of available models
    print (StarDist2D.from_pretrained())

    # creates a pretrained model
    model = StarDist2D.from_pretrained('2D_versatile_fluo')


    #img = normalize(img[16], 1,99.8, axis=axis_norm)
    labels, details = model.predict_instances(img)

    plt.figure(figsize=(8,8))
    plt.imshow(img if img.ndim==2 else img[...,0], clim=(0,1), cmap='gray')
    plt.imshow(labels, cmap='viridis', alpha=0.5)
    plt.axis('off');

    plt.show()

    #######################################
    min_size_roi = 15
    max_size_roi = 700
    #bmi_c.sigma = 0.1

    labels = labels.astype('float32')

    # remove very small and very large ROIs
    min_size = min_size_roi
    max_size = max_size_roi
    roi_centres = []
    indexes = []
    for k in tqdm(np.unique(labels), desc='looping over cells'):
        idx = np.where(labels==k)


        if idx[0].shape[0]<min_size or idx[0].shape[0]>max_size:
            labels[idx]=np.nan
        else:

            roi_centres.append([np.median(idx[0]),
                                 np.median(idx[1])])
            indexes.append(idx)

    rois = np.vstack(roi_centres)
    indexes = indexes

    return rois, indexes
