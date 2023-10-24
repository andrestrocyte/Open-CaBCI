import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange, tqdm

plt.ion()

#
import nidaqmx
from nidaqmx.constants import (AcquisitionType)
from nidaqmx.constants import TerminalConfiguration
import os
import time
from multiprocessing import shared_memory
from utils.utils import smooth_ca_time_series, compute_dff0, compute_dff0_with_reference
from simulation.simulation import Simulation

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
                 n_frames,
                 video_width,
                 video_length,
                 motion_flag,
                 align_flag):
                     

        #
        print("... initializing BMI parameters...")
        print("    TODO: consider saving all imaging data to RAM disk (or faster SSD) for improved speeds")

        #
        self.align_flag = align_flag

        #
        self.motion_flag = motion_flag

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
        # self.fname_save_data = os.path.split(fname_fluorescence)[0]+"bmi_results.npz"
        self.fname_save_data = os.path.join(os.path.split(fname_fluorescence)[0], "results.npz")

        #
        self.fname_rois_pixels_thresholds = fname_roi_pixels_and_thresholds

        #
        self.shared_memory_variables_names_list = []

        # NOT SURE IF REQUIRED... TO DELETE
        # TODO flag was probably used during development toskip the reading step;
        self.read_data_flag = True

        # Define variables
        self.sampleRate_NI = 1E3  # Sample rate of NI card

        #
        self.ttl_pts = 1  # number of values to read from NI card - usually we read a single value to avoid buffering issues

        #
        self.sampleRate_2P = sampleRate_2P  # Sample rate of BScope

        # TODO: externalize these parameters
        self.image_width = 512
        self.image_length = 512

        # not currently used
        self.n_rewards_per_minute = 0
        self.random_reward_probability = 0
        if False:
            self.random_reward_probability = (self.n_rewards_per_minute / (30 * 60))
            print(" RANDOM REWARD PROBABILITY (rewards per minute): ", self.n_rewards_per_minute,
                  "; reward prob per TTL frame: ", self.random_reward_probability)
            #
            if self.n_rewards_per_minute > 0.5:
                for k in range(10):
                    print(" >>>>>>>>>>>> RANDOM REWARD PROBABILITY IS HIGH!!! (rewards per minute): ",
                          self.n_rewards_per_minute, "; reward prob per TTL frame: ", self.random_reward_probability)


        #
        self.max_n_seconds_session = max_n_seconds_session

        # number of frames to run BMI for
        self.n_frames = n_frames  # OLD WAY OF COMPUTING max_n_seconds_session*sampleRate_2P

        # TODO: why do we have 2 of these variables?
        self.n_frames_to_be_acquired = self.n_frames  # Number of frames from BScope

        #
        self.n_frames_search_forward = 5

        # start the ttl frame counter at 0
        self.ttl_computed = 0
        
        #
        self.trials = []


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
        self.initialize_live_frame_shared_memory()

        # start reading the ttl pulses from the 2p scope
        self.initialize_bscope_ttl_pulse_reader()

        # this gets read simultaneously with all other TTL/BNC channels now
        # self.initalize_lick_detector_reader()

        # keeps track of lick values
        self.lick_detector_abstime = []  # np.zeros((self.n_frames,2),dtype=np.float32)

        #
        self.initialize_rotary_encoder()

        #
        self.initialize_video_frame()

        #
        self.initialize_ensemble_state_and_rois()

        #
        self.initialize_reward_conditions_and_parameters()

        #
        self.initialize_motion_correction_variable()

        #
        self.initialize_dynamic_f0_variable()
        
        #
        self.initialize_white_noise_state()
        
        #
        self.initialize_alignment_flag()
        
        #
        self.initialize_contingency_degradation_flag()
        
        #
        self.initialize_threshold_shared_memory()
        
        #
        self.initialize_dynamic_reward_lockout_state()

        #
        self.initialize_manual_motion_correction_array()

    #
    def initialize_alignment_flag(self):

        '''
            Signal that is shared with all cores to indicate termination of BMI
            - 0: keep running
            - 1: end all processing
        '''

        # make a numpy array to hold the rois_traces
        aa = np.zeros(1, dtype=np.int32)
        self.shmem_alignment_flag = shared_memory.SharedMemory(create=True,
                                                                 size=aa.nbytes)

        #
        self.alignment_flag = np.ndarray(aa.shape,
                                     dtype=aa.dtype,
                                     buffer=self.shmem_alignment_flag.buf)

        #
        self.alignment_flag[0] = self.align_flag
     
         #
    def initialize_threshold_shared_memory(self):

        '''
            This variable keeps track of the locally computed E1-E2
            - it is shared with a different process which plays tones
            - TODO: perhaps want a better name like neural_state - to disambugate from ensembel states
        '''

        # make a numpy array to hold the rois_traces
        aa = np.zeros(1,dtype=np.float32)
        self.shmem_high_threshold_state = shared_memory.SharedMemory(create=True,
                                                              size=aa.nbytes)

        #
        self.high_threshold = np.ndarray(aa.shape,
                                         dtype=aa.dtype,
                                         buffer=self.shmem_high_threshold_state.buf)

        #
        self.high_threshold[0] = 1
    
    def initialize_contingency_degradation_flag(self):
        '''
            Signal that is shared with all cores to indicate termination of BMI
            - 0: keep running
            - 1: end all processing
        '''

        # make a numpy array to hold the rois_traces
        aa = np.zeros(1, dtype=np.int32)
        self.shmem_contingency_degradation = shared_memory.SharedMemory(create=True,
                                                                size=aa.nbytes)

        #
        self.contingency_degradation = np.ndarray(aa.shape,
                                          dtype=aa.dtype,
                                          buffer=self.shmem_contingency_degradation.buf)

        #
        self.contingency_degradation[0] = 0
        
           
    #
    def initialize_white_noise_state(self):

        '''
            This variable keeps track of the tone value computed by the TONE class
            - technically it doesn't have to be initialized here, but we do it for simplicity to easier
              share it with the plotter class (BMI class doesn't need it for now)

        '''

        # make a numpy array to hold the rois_traces
        aa = np.zeros(1,dtype=np.float32)
        self.shmem_white_noise_state = shared_memory.SharedMemory(create=True,
                                                                  size=aa.nbytes)

        #
        self.white_noise_state = np.ndarray(aa.shape,
                                         dtype=aa.dtype,
                                         buffer=self.shmem_white_noise_state.buf)

        #
        self.white_noise_state [:] = aa[:]
        
    #
    def initialize_dynamic_reward_lockout_state(self):

        '''
            shared variable indicating whether we are in a reward-lockout state or not
            - required by tone class (possibly others)
        '''

        # make a numpy array to hold the rois_traces
        aa = np.zeros((1), dtype=np.int32)
        self.shmem_dynamic_reward_lockout_state = shared_memory.SharedMemory(create=True,
                                                                 size=aa.nbytes)

        #
        self.dynamic_reward_lockout_state = np.ndarray(aa.shape,
                                             dtype=aa.dtype,
                                             buffer=self.shmem_dynamic_reward_lockout_state.buf)

        #
        self.dynamic_reward_lockout_state[0] = 0

        #
        # ## flag which indicates whether we are in the period post-reward that we want to lockout
        # self.dynamic_reward_lockout_state = False
        
    #
    def initialize_manual_motion_correction_array(self):

        '''
            Left-Right and Up-Down motion correction
            Array index 0 controls left-right shifts
            Array index 1 controls up-down shifts
        '''

        # make a numpy array to hold the rois_traces
        aa = np.zeros((2), dtype=np.int32)
        self.shmem_manual_motion_correction_array = shared_memory.SharedMemory(create=True,
                                                                 size=aa.nbytes)

        #
        self.manual_motion_correction_array = np.ndarray(aa.shape,
                                     dtype=aa.dtype,
                                     buffer=self.shmem_manual_motion_correction_array.buf)

        #
        self.manual_motion_correction_array[:] = aa[:]
        
        
    #
    def initialize_dynamic_f0_variable(self):
        '''
            Signal that is shared with all cores to indicate termination of BMI
            - 0: keep running
            - 1: end all processing
        '''

        # make a numpy array to hold the rois_traces
        aa = np.zeros(1, dtype=np.int32)
        self.shmem_dynamic_f0_flag = shared_memory.SharedMemory(create=True,
                                                                       size=aa.nbytes)

        #
        self.dynamic_f0_flag = np.ndarray(aa.shape,
                                                 dtype=aa.dtype,
                                                 buffer=self.shmem_dynamic_f0_flag.buf)

        #
        self.dynamic_f0_flag[0] = 0

    #
    def initialize_motion_correction_variable(self):

        '''
            Signal that is shared with all cores to indicate termination of BMI
            - 0: keep running
            - 1: end all processing
        '''

        # make a numpy array to hold the rois_traces
        aa = np.zeros(1, dtype=np.int32)
        self.shmem_motion_correction_flag = shared_memory.SharedMemory(create=True,
                                                                     size=aa.nbytes)

        #
        self.motion_correction_flag = np.ndarray(aa.shape,
                                             dtype=aa.dtype,
                                             buffer=self.shmem_motion_correction_flag.buf)

        #
        self.motion_correction_flag[0] = self.motion_flag



    def initialize_ensemble_state_and_rois(self):

        '''
            Initialize the ROIs and ensemble arrays to be used below

            THESE VARIABLES ARE NOT USED FOR CALIBARTION STEP
        '''

        # TODO: generalize some of this code to allow different #s of cells; - not a priority
        #data = np.load(self.fname_rois_pixels_thresholds,
        #               allow_pickle=True)

        #############################################################
        #################### LOAD ENSEMBLE 1 DATA ###################
        #############################################################

        # make a default size matrix that will hold [n_rois, n_frames]
        a = np.zeros((10,self.n_frames),
                      dtype=np.float32)+1E-8

        # rois traces raw: contains the raw ROIs (i.e. summed pixels etc in each ROI)
        self.rois_traces_raw_ensemble1 = np.zeros(a.shape, dtype=np.float32)

        #
        self.shmem_rois_traces_ensemble1 = shared_memory.SharedMemory(create=True,
                                                                   size=a.nbytes)

        #
        self.rois_traces_smooth_ensemble1 = np.ndarray(a.shape,
                                              dtype=a.dtype,
                                              buffer=self.shmem_rois_traces_ensemble1.buf)

        #
        self.rois_traces_smooth_ensemble1[:] = a[:]

        #############################################################
        #################### LOAD ENSEMBLE 2 DATA ###################
        #############################################################

        # make a default size matrix that will hold [n_rois, n_frames]
        a = np.zeros((10,self.n_frames),
                      dtype=np.float32)+1E-8

        #
        self.rois_traces_raw_ensemble2 = np.zeros(a.shape, dtype=np.float32)

        #
        self.shmem_rois_traces_ensemble2 = shared_memory.SharedMemory(create=True,
                                                                   size=a.nbytes)

        #
        self.rois_traces_smooth_ensemble2 = np.ndarray(a.shape,
                                              dtype=a.dtype,
                                              buffer=self.shmem_rois_traces_ensemble2.buf)

        #
        self.rois_traces_smooth_ensemble2[:] = a[:]


        #############################################################
        #################### LOAD ENSEMBLE 2 DATA ###################
        #############################################################
        # make a numpy array to hold the rois_traces
        aa = np.zeros(1, dtype=np.float32)
        self.shmem_ensemble_state = shared_memory.SharedMemory(create=True,
                                                               size=aa.nbytes)

        #
        self.ensemble_state = np.ndarray(aa.shape,
                                         dtype=aa.dtype,
                                         buffer=self.shmem_ensemble_state.buf)

        #
        self.ensemble_state[:] = aa[:]

        # NOTE: this is set to negative only during calibration so there's no feedback
        self.ensemble_state[0] = -3

        #############################################
        self.high_threshold = 10

    #
    def initialize_video_frame(self):
        ''' shared variable that keeps current video camera frame in memeory for
        '''

        # make a numpy array to hold the rois_traces
        print("self.video width: ", self.video_width, " length: ", self.video_length)
        aa = np.zeros((1, self.video_width, self.video_length), dtype=np.uint8)
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
        aa = np.zeros(1, dtype=np.float32)
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
        aa = np.zeros(1, dtype=np.float32)
        self.shmem_tone_state = shared_memory.SharedMemory(create=True,
                                                           size=aa.nbytes)

        #
        self.tone_state = np.ndarray(aa.shape,
                                     dtype=aa.dtype,
                                     buffer=self.shmem_tone_state.buf)

        #
        self.tone_state[:] = aa[:]

    #
    def initialize_reward_conditions_and_parameters(self):

        # set the last reward time in ttl pulses (might need something better here)
        self.initialize_last_reward_ttl()

        # reward lockout time after a positive reward - in seconds
        self.received_random_reward_lockout = 5
        print(">>>>>>>>>>>> POST-RANDOM REWARD LOCKOUT: ", self.received_random_reward_lockout, "sec")

        # counter that track time after last reward
        self.initialize_reward_lockout_counter()

        #

    def initialize_pbar(self):
        self.pbar = tqdm(total=self.n_frames_to_be_acquired,
                         desc='% complete',
                         position=0,
                         leave=True,
                         ascii=True)  # Init pbar

    #
    def initialize_data_arrays(self):
        ''' TODO: check to make sure all the possible data being recorded is being saved

        '''
        #
        self.ttl_values = []  # array to hold ttl data being read
        self.ttl_n_computed = []  # number of ttl pulses computed based on time elapsed
        self.ttl_n_detected = []  # number of ttl pulses detected based on TTL from NI board
        self.inter_ttl_time = []  # computed time between each detected TTL pluse
        self.abs_times = []  # Keep of every time TTL is read... important!
        # loop;   might be useful for debugging later on kernel interuptions etc.
        self.ttl_times = []  # ttl times to be saved
        self.previous_trigger = 0  # time of the previous TTL trigger to be used to determine if next trigger etc
        self.prev_max = 0  # TTL pulse previous read max value
        self.prev_min = 0  # TTL pulse previous read min value
        self.ttl_voltages = []  # ttl_voltages

        # self.initialize_n_ttl()
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
        self.shmem_reward_lockout_counter = shared_memory.SharedMemory(create=True,
                                                               size=aa.nbytes)

        #
        self.random_reward_lockout_counter = np.ndarray(aa.shape,
                                                        dtype=aa.dtype,
                                                        buffer=self.shmem_reward_lockout_counter.buf)

        #
        self.random_reward_lockout_counter[:] = aa[:]

    #
    def initialize_reward_times(self):

        ''' shared variable that tracks # of rewards

        '''

        # an array to hold the reward times
        aa = np.zeros((2, 1000), dtype=np.int64) - 1
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
        aa = np.zeros((1, 512, 512), dtype=np.uint16)
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
        aa = np.zeros((1, 512, 512), dtype=np.uint16)
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
        # print(" ttl counter initialized: ", self.n_ttl, self.shmem_n_ttl.name)

    #
    def run_BMI(self):

        #
        print('Running BMI - Calibrtion (ctrl-c to stop)')

        #
        self.now = time.perf_counter()  # time.perf_counter_ns()/1E9
        self.previous_trigger = time.perf_counter() - 2  # set the previous tirgger 2 sec prior to start

        #
        self.initialize_pbar()

        # abssolute start time
        self.start_time_acquisition = time.time()

        # save trial start for first trial
        #self.trials.append(self.last_trial_start_ttl)

        # start recording and acquisition
        # count number of frames; but probably safer to just count time;
        # TODO: merge ttl pulse counting and time tracking into a single while statement
        while self.ttl_computed < self.n_frames_to_be_acquired - 1:

            # read next bscope ttl pulse
            self.read_bscope_ttl()

            # check of ttl pulse when from high ~5 to low ~0
            if self.min_ < 3 and self.prev_max >= 3:
                #
                # print ("DETECTED TTL: ", self.n_ttl, ", self.ttl_computed: ", self.ttl_computed)

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
            # TODO: not sure if this needed; i don't think it's every been used; supposed to catch TTL misses / failures
            if (time.time() - self.start_time_acquisition) > self.max_n_seconds_session:
                print("Duration of BMI loop: ", time.time() - self.start_time_acquisition, 'sec',
                      "  , total requested: ", self.max_n_seconds_session)
                self.termination_flag[0]=1

            #
            if self.termination_flag[0]==1:
                break

        # save all data acquried during recording
        # TODO: try to save this on the fly if possible to avoid loosing data during crashes
        self.save_data()

        #
        self.bscope_ttl_task.stop()
        self.bscope_ttl_task.close()

    #
    def read_bscope_ttl(self):

        # get current bscope ttl pulse value from NI card output port
        # read_values = np.array(self.bscope_ttl_task.read(number_of_samples_per_channel=self.ttl_pts)).squeeze()
        read_values = self.bscope_ttl_task.read(number_of_samples_per_channel=self.ttl_pts)

        # print ("READ VALUES: ", read_values)

        # ttl bscope value
        ttl_value = read_values[0]  # .copy()

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
        self.now = time.perf_counter()  # perf_counter_ns()/1E9

        # this helps us figure out how fast this loop runs
        # TODO: we may want to introduce a delay of 5ms or so so we don't constanly read TTL pulses
        #     but this is probably not necessary as the NIDAQMX package was made to be pinged a lot
        self.abs_times.append(self.now)

    #
    def close(self):

        #
        print(" ... closing BMI, # of ttl pulses processed: ", self.n_ttl)
        #
        print(" ... SENDING TERMIANTION FLAG SIGNAL TO ALL PROCESSES ...")
        self.termination_flag[0] = 1

        #
        print("... EXITING BMI CLASS...")

        # give the rest of the modules a few sec to complete
        time.sleep(2)

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
            # self.bscope_ttl_task.ai_channels.add_ai_voltage_chan("Dev3/ai1",
            #                                              terminal_config=TerminalConfiguration.NRSE)
            # c
            self.bscope_ttl_task.timing.cfg_samp_clk_timing(self.sampleRate_NI,
                                                            # samps_per_chan=pointsToPlot*2,
                                                            sample_mode=AcquisitionType.CONTINUOUS)

            # start the TTL reader (not required in simulation mode)
            self.bscope_ttl_task.start()

        # list to add to when reading values
        self.bscope_read_values = np.zeros(2, dtype=np.float32)

    #
    def bmi_update(self):

        #
        self.compute_frame_number()

        #
        self.load_current_frame_and_apply_drift_correction()

		# load the [ca] imaging and compute activity in each ROI
        self.update_rois()

        # smooth the ROIs using the external function
        self.compute_dff_and_smooth_rois()

        # check if binning the rois
        if self.binning_flag:
            self.bin_rois()

        # check if doing dynamic f0 updates
        if self.dynamic_f0_flag[0]:
            self.dynamic_f0()

        # compute the ensemble activity from ROIs loaded
        self.update_ensembles()
       
        #
        self.n_ttl += 1




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
        if self.smooth_diff_function_flag and self.n_ttl[0] > self.rois_smooth_window:

            # loop over each cell
            for p in range(self.rois_traces_raw.shape[0]):
                #
                temp = self.rois_traces_raw[p, self.n_ttl[0] - self.rois_smooth_window:self.n_ttl[0]]

                # There are two options for deterneding and computing a DFF0
                # option 1: use the calibration time roi_f0s
                #  Note: this is risky to do:
                #          - sometimes there is signficant drift which we don't correct for (yet!)
                # WE FIXED THIS NOW
                # if False or self.n_ttl[0]<self.n_ttl_to_start_applying_DFF0_computation:
                if True:
                    temp = (temp - self.roi_f0s[p]) / self.roi_f0s[p]

                # Recompute baseline dynamically to ensure alignemtn of data
                # Note: this is also risky as this means the thresholds computed in the calibration step
                #        might not be completely accurate any longer
                #
                else:
                    # so here we feed current chunk of data going back n frames
                    #  plus the refrenc trace which should be the last n frames of raw data; usually take at least 30 seconds
                    _, temp = compute_dff0_with_reference(temp,
                                                          self.rois_traces_raw[p,
                                                          self.n_ttl[0] - self.n_ttl_to_start_applying_DFF0_computation:
                                                          self.n_ttl[0]]
                                                          )

                #
                self.rois_traces_smooth[p, self.n_ttl[0]] = smooth_ca_time_series5(temp)

        else:
            #
            self.rois_traces_smooth[:, self.n_ttl[0]] = self.rois_traces_raw[:, self.n_ttl[0]]

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
        if len(self.ttl_n_computed) == 0:

            #
            if self.read_data_flag:
                # import mmap
                print("  setting up memory map: shape: ", (self.n_frames_to_be_acquired, 512, 512))

                if True:
                    self.newfp = np.memmap(self.fname_fluorescence,
                                           dtype='uint16',
                                           mode='r',
                                           shape=self.n_frames_to_be_acquired * 512 * 512)

                #
                self.newfp = self.newfp.reshape(self.n_frames_to_be_acquired, 512, 512)

            # reset start time: requird becaues we start the BMI a few seconds before the BScope
            self.start = self.now

        # after arrays initialized
        else:

            # in simulation mode we just assume that we have correctly dected a TTL pulse and add 1 extra
            #   ttl pulse to the stack
            if self.simulation_mode_bmi == True:
                self.ttl_computed = self.n_ttl + 1  # move to next ttl.
                time.sleep(self.sleep_time_sec)

            else:
                # DISABLE TTL RECOMPUTATION FOR NOW JULY 25
                if False:
                    time_passed = self.now - self.start
                    self.ttl_computed = round((self.now - self.start) * self.sampleRate_2P)

                    #
                    if self.verbose:
                        print(" time passed: ", time_passed, "   bmi_update self.ttl_computed: ", self.ttl_computed)
                #
                self.ttl_computed = self.n_ttl[0]

    #
    def trigger_random_reward(self):

        # generate water reward only if we are not in a reward lockout state
        print(" ****giving random reward at time: ", self.n_ttl)

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
    def load_current_frame_and_apply_drift_correction(self):

        ''' This is an overkill function which cheks whether the n_ttl detected from TTL pulses
            doe sindeed have values in it, if so it

            TODO: either drop this first check - or implement it in full - which means going forward in time
                  until we get zeros in the data

        '''

        #
        if self.verbose:
            print("self.ttl_computed: ", self.ttl_computed)
            print("  detected frame #: ", self.n_ttl,
                  " computed_frame : ", self.ttl_computed)

        # search the very first ROI in time from previous frame to future frames until we get a non-zero pixel values;
        #  then we set the time i.e. n_ttl
        for z in range(-1, self.n_frames_search_forward, 1):

            # TODO ;could just check any part of the FOV to see if there is non zero values
            # roi_sum0 = self.newfp[self.n_ttl[0]+z][self.rois_pixels[0]].sum()
            # just check any part of the image to see if it's been saved yet
            roi_sum0 = self.newfp[self.n_ttl[0] + z][self.image_width // 2 - 5:self.image_width // 2 + 5].sum()

            #
            if roi_sum0 != 0:
                # TODO: reset the n_ttl value here - check that this is safe!!!
                # self.n_ttl[0] = self.n_ttl[0]+z

                break

        # Note: in calibration we do not do any motion correction
        self.live_frame_local = self.newfp[self.n_ttl[0] + z].copy()

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
            if self.reward_times[1, k] == -1:
                break
        print(" .. # of rewards: ", k, ", water volume dispensed (@ 20uL per reward): ", (k) * 0.020, "mL")

        #
        np.savez(self.fname_save_data,
                 ttl_voltages=self.ttl_voltages,
                 ttl_n_computed=self.ttl_n_computed,
                 ttl_n_detected=self.ttl_n_detected,
                 abs_times=self.abs_times,
                 ttl_times=self.ttl_times,
                 reward_times=self.reward_times,
                 rewarded_times_abs=self.rewarded_times_abs,
                 received_random_reward_lockout=self.received_random_reward_lockout,
                 n_rewards_per_minute = self.n_rewards_per_minute,
                 sampleRate_NI=self.sampleRate_NI,
                 ttl_pts=self.ttl_pts,
                 sampleRate_2P=self.sampleRate_2P,
                 image_width=self.image_width,
                 image_length=self.image_length,
                 max_n_seconds_session=self.max_n_seconds_session,

                 n_frames=self.n_frames,
                 n_frames_to_be_acquired=self.n_frames_to_be_acquired,  #
                 n_frames_search_forward=self.n_frames_search_forward,
                 lick_detector_abstime=self.lick_detector_abstime,
                 rotary_encoder1_abstime=self.rotary_encoder1_abstime,
                 rotary_encoder2_abstime=self.rotary_encoder2_abstime,
                 )

