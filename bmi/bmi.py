'''
  
  Catalin Mitelut; github: "catubc"; mitelutco@gmail.com

'''

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
        #
        ttl_val = self.ttl[self.index:self.index+number_of_samples_per_channel]
        self.index += number_of_samples_per_channel

        #
        return ttl_val

#################################################
################## BMI CLASS ####################
#################################################
class BMI():

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
                 simulation_mode,
                 fname_root_path,
                 fname_fluorescence,
                 fname_ttl,
                 sampleRate_2P,
                 fname_roi_pixels_and_thresholds,
                 max_n_seconds_session,
                 n_frames_session):

        #
        print ("... initializing BMI parameters...")
        print ("    TODO: consider saving all imaging data to RAM disk (or faster SSD) for improved speeds")

        #
        self.simulation_mode = simulation_mode

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

        # NOT SURE IF REQUIRED... TO DELETE
        # TODO flag was probably used during development toskip the reading step;
        self.read_data_flag = True

        # Define variables
        self.sampleRate_NI = 1E3     # Sample rate of NI card

        #
        self.ttl_pts = 1  			 # number of values to read from NI card - usually we read a single value to avoid buffering issues

        #
        self.sampleRate_2P = sampleRate_2P	    # Sample rate of BScope

		#
        self.image_width = 512
        self.image_length = 512

		#
        self.max_n_seconds_session = max_n_seconds_session

        # number of frames to run BMI for
        self.n_frames = n_frames_session # OLD WAY OF COMPUTING max_n_seconds_session*sampleRate_2P

        # TODO: why do we have 2 of these variables?
        self.n_frames_to_be_acquired = self.n_frames   # Number of frames from BScope

        #
        self.rois_smooth_window = 5   				# Number of frames to use to smooth the ROI traces
                                                    # to be developed/changed further

		# complicated paramter which turns on realitime DFF0 computation only after a certain period of time
        # TODO: determine if online DFF0 is required:
        #  things to evaluate: bleaching type of slow baseline drift...
        #     but for this slow drift we can use very long windows (like 2mins or more)
        # - for faster update not sure this is correct
        self.n_ttl_to_start_applying_DFF0_computation = 30 *self.sampleRate_2P

        # start the ttl frame counter at 0
        self.ttl_computed = 0

        # number of frames to search forward in time to see if there is any neural data saved
        #   this is for the ROI reading step
        self.n_frames_search_forward = 5

        # initizlie ROIs
        self.initialize_ROIs()

        # initizlie the realtime value of the ensembel states (i.e. no history)
        # TODO: may wish to hold history somewhere also
        self.ensemble_activity = np.zeros((2, self.n_frames_to_be_acquired))
        
        # this is the differences of the 2 ensemble
        self.ensemble_diff_array = np.zeros(self.n_frames_to_be_acquired)

        # initailize the realtime roi states; these hold the smooth/processed version of the realtime roi
        self.rois_activity_realtime = np.zeros(len(self.rois_pixels),dtype=np.float32)

        # initialize all arrays to be used, mostly to save data after BMI run
        self.initialize_data_arrays()

        # initialize tone state
        self.initialize_ensemble_state()

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
        self.initialize_live_frame_shared_memory()

        #
        self.initialize_drift_correction()

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
        print (" ensemble states initialized: ",
               self.tone_state,
               self.shmem_tone_state.name)

    #
    def initialize_drift_correction(self):

        ''' These 2 variables keep track of the x and y drift
            - template is computed in the calibration step
            - drift gets computed in the Drift class using phase correlation
              between the template and the live image
            - here we just initialize the variables that can be shared with rest of code
        '''

        # make a numpy array to hold the rois_traces
        aa = np.zeros(2, dtype=np.int32)
        self.shmem_drift_xy_values = shared_memory.SharedMemory(create=True,
                                                       size=aa.nbytes)

        #
        self.drift_xy_values = np.ndarray(aa.shape,
                                         dtype=aa.dtype,
                                         buffer=self.shmem_drift_xy_values.buf)

        #
        self.drift_xy_values[:] = aa[:]

        # a list to save all the realtime applied drift
        self.drift_array = []

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
        self.received_reward_lockout = 10

        # counter that track time after last reward
        self.initialize_reward_lockout_counter()

        # the amount of time the mouse has to try and receive a reward - in seconds
        self.max_reward_window = 30

        # similar to post-reward lockout
        self.missed_reward_lockout = 0
        
        #
        self.template = data['calibration_template']

    #
    def initialize_ensemble_state(self):

        '''
            This variable keeps track of the locally computed E1-E2
            - it is shared with a different process which plays tones
            - TODO: perhaps want a better name like neural_state - to disambugate from ensembel states
        '''

        # make a numpy array to hold the rois_traces
        aa = np.zeros(1,dtype=np.float32)
        self.shmem_ensemble_state = shared_memory.SharedMemory(create=True,
                                                       size=aa.nbytes)

        #
        self.ensemble_state = np.ndarray(aa.shape,
                                         dtype=aa.dtype,
                                         buffer=self.shmem_ensemble_state.buf)

        #
        self.ensemble_state [:] = aa[:]

        #
        #print (" ensemble states initialized: ",
        #       self.ensemble_state,
        #       self.shmem_ensemble_state.name)

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
        self.ttl_values = []			# array to hold ttl data being read
        self.ttl_n_computed = []	    # number of ttl pulses computed based on time elapsed
        self.ttl_n_detected = []        # number of ttl pulses detected based on TTL from NI board
        self.inter_ttl_time = []        # computed time between each detected TTL pluse
        self.abs_times = [0] 			# NOT SURE REQUIRED ANYMORE; keeps track of every TTLL read outside the main BMI
                                        # loop;   might be useful for debugging later on kernel interuptions etc.
        self.ttl_times = []				# ttl times to be saved
        self.previous_trigger=0			# time of the previous TTL trigger to be used to determine if next trigger etc
        self.prev_max = 0 				# TTL pulse previous read max value
        self.prev_min = 0				# TTL pulse previous read min value
        self.ttl_voltages = []          # ttl_voltages

        #self.initialize_n_ttl()
        self.rewarded_times = []

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
        self.reward_lockout_counter = np.ndarray(aa.shape,
                                    dtype=aa.dtype,
                                    buffer=self.shmem_reward_lockout.buf)

        #
        self.reward_lockout_counter[:] = aa[:]

    #
    def initialize_reward_times(self):

        ''' shared variable that tracks # of rewards

        '''

        # make a numpy array to hold the rois_traces
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
        print(" n_rewards initialized: ", self.reward_times, self.shmem_reward_times.name)

    #
    def initialize_live_frame_shared_memory(self):

        ''' shared variable that keeps current image in memeory for plotter to visualize

        '''

        # make a numpy array to hold the rois_traces
        aa = np.zeros((1,512,512), dtype=np.uint16)
        self.shmem_live_frame = shared_memory.SharedMemory(create=True,
                                                             size=aa.nbytes)

        #
        self.live_frame = np.ndarray(aa.shape,
                                dtype=aa.dtype,
                                buffer=self.shmem_live_frame.buf)

        #
        self.live_frame[:] = aa[:]

        #
        #print(" n_rewards initialized: ", self.reward_times, self.shmem_reward_times.name)

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
        print(" ttl counter initialized: ", self.n_ttl, self.shmem_n_ttl.name)

    #
    def initialize_ROIs(self):

        '''
            Initialize the ROIs and ensemble arrays to be used below

            TODO: Must properly transfer ROIs to this function not just use a box aroudn a point of interest
        '''

        #####################################################
        # load individual cell ROIs as saved by the calibration step
        # TODO: generalize some of this code to allow different #s of cells; - not a priority
        data = np.load(self.fname_rois_pixels_thresholds,
                       allow_pickle=True)
        self.rois_pixels = []
        self.rois_pixels.append(data['cell0_footprint'])
        self.rois_pixels.append(data['cell1_footprint'])
        self.rois_pixels.append(data['cell2_footprint'])
        self.rois_pixels.append(data['cell3_footprint'])

        # make a default size matrix that will hold [n_rois, n_frames]
        a = np.zeros((len(self.rois_pixels),self.n_frames),
                      dtype=np.float32)+1E-8

        # rois traces raw: contains the raw ROIs (i.e. summed pixels etc in each ROI)
        # these are NOT shared with external classes, unless needed in future
        self.rois_traces_raw = np.zeros(a.shape, dtype=np.float32)
        
        # these are the (raw - F0)/F0 traces
        # note the calibration class also shares code
        # do not change their computation without changing it also in the calibration class
        self.rois_traces_dff0 = np.zeros(a.shape, dtype=np.float32)
        

        # also load the f0 for each cell computed by calibration step (hopefully within a few mins of BMI step)
        # TODO: this calculation is simplified, make sure you don't change power, settings etc.
        #       between calibartion seession and BMI session
        #      - to use an identical function between calibration and bmi whenever possible/required
        # NOTE June 9 - NOT SURE WE WANT TO RESUSE CALIBRATION TIME F0s anylonger
        self.roi_f0s = data['cell_f0s']

        #####################################
        # initialize the fluorescence time series called rois_traces which keeps
        #  track of time series for all the ROIs
        # Note: we have to share this with the plotting function so we use sharedmemory
        # NOTE we want to share the smooth traces with the plotting function as that is what the
        #      reward condition computation is based on

        self.shmem_rois_traces = shared_memory.SharedMemory(create=True,
                                                            size=a.nbytes)

        #
        self.rois_traces_smooth = np.ndarray(a.shape,
                                      dtype=a.dtype,
                                      buffer=self.shmem_rois_traces.buf)

        #
        self.rois_traces_smooth[:] = a[:]

    #
    def run_BMI(self):

        #
        print('Running BMI (ctrl-c to stop)')

        #
        self.initialize_ttl_reader()

        # # set time to auto finish the acquisition;
        # #    TODO: automate acquisition end if no TTL pulses for X seconds
        # t_end = (time.perf_counter() +
        #          self.n_frames_to_be_acquired/self.sampleRate_2P+
        #          5)

      
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

            ttl_value = self.task_ttl.read(number_of_samples_per_channel=self.ttl_pts)

            #  leave these in just in case we end up reading at higher bit rates and multiple samples at a atime
            #print ("READ: TTL: ", ttl_value)
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

            # check of ttl pulse when from high ~5 to low ~0
            if self.min_<1 and self.prev_max>=1:

                # runs the bmi code whenever imaging frame is completed
                self.bmi_update()

                # update trigger time
                self.previous_trigger = self.now

                #
                self.pbar.update(n=1)
            #else:
            #    print (" no pulse...")

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
    def initialize_ttl_reader(self):

        #
        if self.simulation_mode == True:
            self.task_ttl = Simulation(self.fname_ttl)
        else:
			
            #print ("  RESETTING DEV3 ")
            #dev_name = "Dev3"
            #dev_name=dev_name.strip("/")
            #dev=nidaqmx.system.Device(dev_name)
            #dev.reset_device()
            
            #
            # time.sleep(3)
            			
			#			
            self.task_ttl = nidaqmx.Task('bmi_online')
            print ("iniitlied bmi online")
            # set TTL pulse reader from 2p system
            self.task_ttl.ai_channels.add_ai_voltage_chan("Dev3/ai0",
                                                          terminal_config=TerminalConfiguration.NRSE)

            #
            self.task_ttl.timing.cfg_samp_clk_timing(self.sampleRate_NI,
                                                     # samps_per_chan=pointsToPlot*2,
                                                     sample_mode=AcquisitionType.CONTINUOUS)

            # start the TTL reader (not required in simulation mode)
            self.task_ttl.start()

    #
    def bmi_update(self):

        #
        self.compute_frame_number()

        #
        self.load_current_frame_and_apply_drift_correction()

        # load the [ca] imaging and compute activity in each ROI
        self.update_rois()

        # smooth the ROIs using the external function
        self.smooth_rois()

        # compute the ensemble activity from ROIs loaded
        self.update_ensembles()

        # check for reward condition:
        self.check_reward_condition()

        # check if > 30 sec has passed since last reward
        self.check_missed_reward_state()

        # decrease any potential reward lockout counter
        self.reward_lockout_counter[0] -= 1

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
		#
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

        # first time point
        if len(self.ttl_n_computed)==0:

            #
            if self.read_data_flag:
                import mmap
                ss = time.time()
                print ("  setting up memory map: shape: ", (self.n_frames_to_be_acquired,512,512))
                
                if True:
                    self.newfp = np.memmap(self.fname_fluorescence,
                                           dtype='uint16',
                                           mode='r',
										   shape=self.n_frames_to_be_acquired*512*512)
                
                # TODO: THIS IS RQUIRD BY WINDOWS.
                #    FOR SOME REASON IT DOESN"T LIKE NUMPY MEMMAP
                if False:
                    fp = open(self.fname_fluorescence, "r")
                    byts = self.n_frames_to_be_acquired*self.image_width*self.image_length
                    
                    self.newfp = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
                    # print ("OPTION @@@@@@@@@@@@@@@@: ", self.newfp)
					
					#
                    mview = memoryview(self.newfp)
                    print (" memroy view: ", mview)
                    self.newfp = np.asarray(mview).reshape(self.n_frames_to_be_acquired,512,512)
                    print (" final array view: ", self.newfp.shape)

					
                # 
                print ("sefl newfp: ", self.newfp.shape)
				
                self.newfp = self.newfp.reshape(self.n_frames_to_be_acquired,512,512)
				#m.write("Hello world!")                       
                                       
                print (" duration to setup memmap: ", time.time()-ss, " sec.")
                print ("     TODO: work with 1D flattened arrays")

            # reset start time: requird becaues we start the BMI a few seconds before the BScope
            self.start=self.now

        else:

            # in simulation mode we just assume that we have correctly dected a TTL pulse and add 1 extra
            #   ttl pulse to the stack
            if self.simulation_mode==True:
                self.ttl_computed = self.n_ttl+1  # move to next ttl.
                time.sleep(self.sleep_time_sec)
                
            else:
                time_passed = self.now-self.start
                self.ttl_computed = round((self.now-self.start)*self.sampleRate_2P)

                # 
                if self.verbose:
                    print (" time passed: ", time_passed, "   bmi_update self.ttl_computed: ", self.ttl_computed)

    #
    def trigger_reward(self):

        # generate water reward only if we are not in a reward lockout state
        print (" ****giving reward at time: ", self.n_ttl)

        #
        self.water_reward[0] = 1

        # start a counter that
        self.reward_lockout_counter[0] = self.received_reward_lockout * self.sampleRate_2P

    #
    def post_reward_state(self):

        # disable tone playback;
        self.tone_off()

    #
    def check_missed_reward_state(self):

        # if mouse does not perform in e.g. 30 sec, do a lockout
        # TODO: May wish to play white noise to distinguish it from post-reward state
        if (self.n_ttl[0]-self.last_reward_ttl[0])>(self.max_reward_window*self.sampleRate_2P):
            print ("  triggering missed reward lockout ")

            # reset counter
            self.reward_lockout_counter[0] = self.missed_reward_lockout * self.sampleRate_2P

            # turn tone off; but better might be to play white noise!?
            print(".... may wish to play white noise to dsituish between post-reward state")
            self.tone_off()

            # reset the last rewarded time to
            self.last_reward_ttl[0] = self.n_ttl[0]

    #
    def check_baseline_condition(self):

        # check if ensemble activity back to baseline; e.g. within 1 x of std
        # if abs(ensemble...[ensbmel_ID1] - ensbmel..[ensemble_ID2])< std x 1?
        #     return True
        # else:
        #     return False

        pass

    #
    def check_reward_condition(self):

        ''' We check if reward condition was reached

        '''

        # IF E1 reward state reached
        # if (self.ensemble_state >= self.low_threshold) and (self.reward_lockout_counter[0]<=0):

            # # low reward state reached
            # # search for the first empty slot in the reward times list
            # for k in range(self.reward_times.shape[1]):
                # if self.reward_times[0,k]==-1:
                    # self.reward_times[0,k] = self.n_ttl[0]     # save current reward time
                    # break

            # # same variable as above; probably need to reduce osme of this redundancy at some point
            # # - this is a list though, more useful for other things also
            # self.rewarded_times.append([0, self.n_ttl[0], self.abs_times])

            # # reset last reward time to current time
            # self.last_reward_ttl[0] = self.n_ttl[0]

            # #
            # self.trigger_reward()

            # # decouple tone etc. from feedback
            # self.post_reward_state()
            
        #
        if (self.ensemble_state[0] >= self.high_threshold) and (self.reward_lockout_counter[0]<=0):
			
			#
            print (" reached high reward conition: ")
            print (" ensemble state: ", self.ensemble_state)
            print (" high_threshold: ", self.high_threshold)

            # search for the first empty slot in the reward times list
            for k in range(self.reward_times.shape[1]):
                if self.reward_times[1, k] == -1:
                    self.reward_times[1, k] = self.n_ttl[0]  # save current reward time
                    break
            #
            self.rewarded_times.append([1, self.n_ttl[0], self.abs_times])

            # reset last reward time to current time
            self.last_reward_ttl[0] = self.n_ttl[0]

            #
            self.trigger_reward()

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

        # Before updating ROIS - must find the correct frame number and load the frame
        # for the first ROI: we loop over the data from -1 frames back to up to n_frames_search_forward in the future
        #  - we are looking for the last frame that has data in it;
        #    we then exit and keep the counter in memroy

        # IMPORTANT #1
        # TODO: 2 options for computing activity in a cell ROI: sum vs. mean (there are others 
        # IMPORTANT #2
        # TODO: this algorithm essentially uses empirical data to check how far our imaging system has gone
        # - it is probably the best way to ensure that we are up todate with real time (at least realtime with the 2p + writing times
        # - more to think about whether this can go wrong
        # - but for now, this next loop is quasi-guarantee that we are in real time

        # search the very first ROI in time from previous frame to future frames until we get a non-zero pixel values;
        #  then we set the time i.e. n_ttl
        for z in range(-1,self.n_frames_search_forward,1):

            # check
            #roi_sum0 = self.newfp[self.n_ttl[0]+z,
            #                      self.rois[0][0]-self.roi_width:self.rois[0][0]+self.roi_width,
            #                      self.rois[0][1]-self.roi_width:self.rois[0][1]+self.roi_width].sum()

            # TODO ;could just check any part of the FOV to see if there is non zero values
            roi_sum0 = self.newfp[self.n_ttl[0]+z][self.rois_pixels[0]].sum()

            #
            if roi_sum0 != 0:
				
				# TODO: reset the n_ttl value here - check that this is safe!!!
                # self.n_ttl[0] = self.n_ttl[0]+z
				
                break

        # TODO: we should reset the n_ttl here
        # - if we find that we needed to search x steps forward,
        #   we should then add x to n_ttl - and vice versa

        # TODO: update latest image for imaging purposese
        # this raw frame is fed to the drift correction algorithm (anywhere else!?)

        # this is the same raw frame but now it is fixed for purpose of computing ROIs!!
        #  - this is the latest frame extracted
        self.live_frame_local = self.newfp[self.n_ttl[0]+z].copy()

		# motion detector gets this frame; and returns drift_xy_values
		self.live_frame_motion_detector[0] = self.live_frame_local.copy()
        
        #
        if self.apply_drift_flag:
            #print ("LIVE IMAGE BMI*  motion detection self.drift_xy_values: ", self.drift_xy_values)
        
			# save most recent drift values from drift module
            self.drift_array.append([self.drift_xy_values[0],
									 self.drift_xy_values[1]])
			
			# NOTE: the drift_xy values could be the previously saved ones
            self.live_frame_local_drift_corrected = apply_shifts(self.live_frame_local.copy(),
                                                                 self.drift_xy_values[0],
                                                                 self.drift_xy_values[1])

        else:
            self.live_frame_local_drift_corrected = self.live_frame_local.copy()

        # this is the frame that the plotting function sees
        self.live_frame[0] = self.live_frame_local_drift_corrected.copy()


    #
    def update_rois(self):

        # loop over the remaning cells on the last frame 'z'
        for p in range(0,len(self.rois_pixels),1):

            # new way use exact pixel location
            temp = self.live_frame_local_drift_corrected[self.rois_pixels[p].T[:, 0],        # broadcast/index into the frame as per ROI pixels
                                                         self.rois_pixels[p].T[:, 1]]

            # divide by the number of pixels in the ROI - NOT SURE IF THIS IS CORRECT?!
            # TODO: these algorithms must match the default water disposal algorithms
            # TODO: USE A FUNCTION OVER THIS AND FOLLOWING STEP THAT IS SHARED WITH CALIBRATION CODE
            roi_sum0 = temp / self.rois_pixels[p][0].shape[0]

            # sum
            # TODO: not sure this is the correct function; to check literature
            # TODO: also this part shoudl be refactored to a callabale function by both calibration and BMI classes
            roi_sum0 = np.nansum(roi_sum0)
            		
			# Note: Do not remove baseline yet; this is done in the smoothing step;
            # TODO: make sure that this approach is correct
            self.rois_traces_raw[p,self.n_ttl[0]] = roi_sum0

        #
        if self.verbose:
            print ("")
            print ("")

    #
    def update_ensembles(self):

        # wait for at least some frames to go by first
        if self.n_ttl[0]>self.rois_smooth_window:

            # compute ensemble 1
            self.ensemble_activity[0,self.n_ttl[0]] = (self.rois_traces_smooth[0, self.n_ttl[0]]+
                                                       self.rois_traces_smooth[1, self.n_ttl[0]])

            # compute ensemble 1
            self.ensemble_activity[1,self.n_ttl[0]] = (self.rois_traces_smooth[2, self.n_ttl[0]]+
                                                       self.rois_traces_smooth[3, self.n_ttl[0]])

        # Compute the E1-E2 for current time point
        # this value goes to the tone package which converts it into a tone
        self.ensemble_state[0] = abs(self.ensemble_activity[0, self.n_ttl[0]] -
                                     self.ensemble_activity[1, self.n_ttl[0]])

        #
        self.ensemble_diff_array[self.n_ttl[0]] = self.ensemble_state[0]
        #print ("time: ", self.n_ttl[0]/self.sampleRate_2P, " updated ensembel state: ", self.ensemble_state, "**********************")

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
        print("...Saving BMI metadata...")
        print ("DRIFT ARRAY: ", self.drift_array)

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
				 n_frames_to_be_acquired = self.n_frames_to_be_acquired,				#
				 rois_smooth_window = self.rois_smooth_window,
				 n_ttl_to_start_applying_DFF0_computation = self.n_ttl_to_start_applying_DFF0_computation,
				 n_frames_search_forward = self.n_frames_search_forward,
                 drift_array = self.drift_array,
                 template = self.template,
                 )

