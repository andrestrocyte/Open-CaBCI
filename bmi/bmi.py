'''
  
  C. Mitelut; github: "catubc"; mitelutco@gmail.com

'''

import matplotlib.pyplot as plt
import nidaqmx
from nidaqmx.constants import (AcquisitionType)  # https://nidaqmx-python.readthedocs.io/en/latest/constants.html
from nidaqmx.constants import TerminalConfiguration
import tqdm # tdqm
import os
import time
import numpy as np
from multiprocessing import shared_memory

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
        TODO:  this way we dont' even have to pass them to the other modules
    '''

    def __init__(self,
                 simulation_mode,
                 fname_root_path,
                 fname_fluorescence,
                 fname_rois,
                 fname_freq,
                 fname_ttl,
                 sampleRate_2P,
                 n_seconds_session):

        #
        self.simulation_mode = simulation_mode

        #
        self.fname_root_path = fname_root_path
        self.fname_fluorescence = fname_fluorescence
        self.fname_rois = fname_rois
        self.fname_freq = fname_freq
        self.fname_ttl = fname_ttl

        #
        self.fname_rois_pixels = os.path.join(fname_root_path,
                                              'rois.npz')

        # NOT SURE IF REQUIRED... TO DELETE
        # TODO flag was probably used during development toskip the reading step;
        self.read_data_flag = True

        # Define variables
        self.sampleRate_NI = 1E3     # Sample rate of NI card

        #
        self.ttl_pts = 1  			 # number of values to read from NI card - usually we read a single value to avoid buffering issues

        #
        self.sampleRate_2P = sampleRate_2P	    # Sample rate of BScope

        # number of frames to run BMI for
        self.n_frames = n_seconds_session*sampleRate_2P

        # TODO: why do we have 2 of these variables?
        self.n_frames_to_be_acquired = self.n_frames   # Number of frames from BScope

        #
        self.rois_smooth_window = 5   				# Number of frames to use to smooth the ROI traces
                                                    # to be developed/changed further
        # initialize smoothing fuction kernel
        self.initialize_smoothing_kernel()

        # start the ttl frame counter at 0
        self.ttl_computed = 0

        # number of frames to search forward in time to see if there is any neural data saved
        #   this is for the ROI reading step
        self.n_frames_search_forward  = 5

        # initizlie ROIs using either a text file or more specific code
        #   for now simple version uses some random centres in the imaging file
        #   MUST CHANGE
        self.initialize_ROIs()

        # initizlie the realtime value of the ensembel states (i.e. no history)
        # TODO: may wish to hold history somewhere also
        self.ensemble_activity_realtime = np.zeros(len(self.rois_pixels))

        # initialize all arrays to be used, mostly to save data after BMI run
        self.initialize_data_arrays()

        # initialize tone state
        self.initialize_tone_state()

        # initalize reward contidions based on ~15mins of pre BMI recorded data
        self.initialize_reward_conditions()

    #
    def drift_correction(self):

        ''' Unclear if this is needed for our application,

        '''
        pass

    #
    def initialize_smoothing_kernel(self):

        ''' This is important step where we get rid of potential aritfacts (e.g. x,y,z drift)

        '''

        # TODO: look at literature and figure out what the correct smoothing function IF ANY is being used
        self.smoothing_kernel = np.arange(0,self.rois_smooth_window,1)/self.rois_smooth_window


    #
    def initialize_reward_conditions(self):

        ''' This function should load the parameters computed by another script
            Input: filenames of several parameters:
                - minimum frequency (e.g. 1000Hz)
                - maximum frequency (e.g. 180000Hz)
                - threshold_low (e.g. -1.0)
                - threshold_high (e.g. +1.0)
        '''


        pass

    #
    def initialize_tone_state(self):

        # this variable keeps track of the locally computed tone state
        # ---

        # make a numpy array to hold the rois_traces
        aa = np.zeros(1,dtype=np.int64)
        self.shmem_tone_frequency = shared_memory.SharedMemory(create=True,
                                                       size=aa.nbytes)

        #
        self.tone_frequency = np.ndarray(aa.shape,
                                         dtype=aa.dtype,
                                         buffer=self.shmem_tone_frequency.buf)

        #
        self.tone_frequency [:] = aa[:]

        #
        print (" tone frequency initialized: ",
               self.tone_frequency ,
               self.shmem_tone_frequency.name)

    #
    def initialize_pbar(self):
        self.pbar = tqdm.tqdm(total=self.n_frames_to_be_acquired,
                              desc='% complete',
                              position=0,
                              leave=True,
                              ascii=True)  # Init pbar

    #
    def initialize_data_arrays(self):

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

        self.initialize_n_ttl()

    #
    def initialize_n_ttl(self):

        # this variable keeps track of how many frames the BMI has detected
        # --- needs to be shared with the plotting algorithm

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

        # load ROI centres from disk;
        # OLD WAY OF COMPUTING SQUARE AROND CENTRE OF CELL
        # self.roi_width = 15 # number of pixels around ROI to grab

        # NEW WAY OF LOADING EXACT PIXELS
        # TODO: generalize some of this code to allow different #s of cells; - not a priority
        data = np.load(self.fname_rois_pixels, allow_pickle=True)
        self.rois_pixels = []
        self.rois_pixels.append(data['cell0'])
        self.rois_pixels.append(data['cell1'])
        self.rois_pixels.append(data['cell2'])
        self.rois_pixels.append(data['cell3'])

        #
        print (" loaded rois pixels; # cells ", len(self.rois_pixels))

        #####################################
        # initialize the fluorescence time series called rois_traces which keeps
        #  track of time series for all the ROIs
        # Note: we have to share this with the plotting function so we use sharedmemory
        a = np.zeros((len(self.rois_pixels),self.n_frames),
                      dtype=np.float32)+1E-8

        #
        self.shmem_rois_traces = shared_memory.SharedMemory(create=True,
                                                            size=a.nbytes)

        #
        self.rois_traces = np.ndarray(a.shape,
                                      dtype=a.dtype,
                                      buffer=self.shmem_rois_traces.buf)

        #
        self.rois_traces[:] = a[:]

    #
    def run_BMI(self):

        #
        # #
        # start = time.perf_counter_ns()
        #
        # #
        # n_sec_to_stop_after_no_TTL_pluse = 5

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
        self.start = time.perf_counter()

        # start recording and acquisition
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

                # runs the bmi code whenever imaging frame is completed
                self.bmi_update()

                # update trigger time
                self.previous_trigger = self.now

                #
                self.pbar.update(n=1)

            #
            self.prev_min = self.min_
            self.prev_max = self.max_

            #

        # save all data acquried during recording
        # TODO: try to save this on the fly if possible to avoid loosing data during crashes
        self.save_data()

        #
        print("... DONE BMI...")

    def initialize_ttl_reader(self):

        #
        if self.simulation_mode == True:
            self.task_ttl = Simulation(self.fname_ttl)
        else:
            self.task_ttl = nidaqmx.Task('bmi_online')

            #
            print ("TODO: check if TLL voltages are being read in real time or buffering")
            print ("   if buffering, then it's a problem if the OS/kernel hang up and we fall behind too much")
            print ("   save workaround is to and read a data a few frames ahead of current count to ensure not behind")


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
                ss = time.time()
                self.newfp = np.memmap(self.fname_fluorescence, dtype='uint16', mode='r',
                                       shape=(self.n_frames_to_be_acquired,512,512))
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

        # generate water reward

        pass

    #
    def post_reward_state(self):

        # disable tone playback;
        self.tone_off()


        LENGTH = 10  # Number of iterations required to fill pbar

        #pbar = tqdm(total=LENGTH)  # Init pbar

        # run while loop until ensemble activit return to normal;
        # start recording and acquisition
        while self.now < t_end:

            # search for next TTL pulse
            data = self.task_ttl.read(number_of_samples_per_channel=pts)

            #  leave these in just in case we end up reading at higher bit rates and multiple samples at a atime
            self.min_ = np.min(data)
            self.max_ = np.max(data)
            self.ttl_state.append(data)

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
    def update_rois(self):

        if self.read_data_flag:

            #
            if self.verbose:
                print ("self.ttl_computed: ", self.ttl_computed)
                print("  detected frame #: ", self.n_ttl,
                   " computed_frame : ", self.ttl_computed)

            # update ROIS
            # for the very first ROI: we loop over the data from -1 frames back to up to n_frames_search_forward in the future
            #  - we are looking for the last frame that has data in it;
            #    we then exit and keep the counter in memroy

            # IMPORTANT #1
            # TODO: 2 options for computing activity in a cell ROI: sum vs. mean (there are others of course
            #


            # IMPORTANT #2
            # TODO: this algorithm essentially uses empirical data to check how far our imaging system has gone
            # - it is probably the best way to ensure that we are uptoday with real time (at least realtime with the 2p + writing times
            # - more to think about whether this can go wrong
            # - but for now, this next loop is quasi-guarantee that we are in real time
            for z in range(-1,self.n_frames_search_forward,1):

                # check
                #roi_sum0 = self.newfp[self.n_ttl[0]+z,
                #                      self.rois[0][0]-self.roi_width:self.rois[0][0]+self.roi_width,
                #                      self.rois[0][1]-self.roi_width:self.rois[0][1]+self.roi_width].sum()

                # TODO ;could just check any part of the FOV to see if there is non zero values
                roi_sum0 = self.newfp[self.n_ttl[0]+z][self.rois_pixels[0]].sum()

                #
                if roi_sum0 != 0:
                    break

            # TODO: we should reset the n_ttl here
            # - if we find that we needed to search x steps forward,
            #   we should then add x to n_ttl - and vice versa

            # loop over the remaning cells on the last frame 'z'
            for p in range(0,len(self.rois_pixels),1):

                # old way of summin pixels inside square centred on cell
                # roi_sum0 = self.newfp[self.n_ttl[0]+z,
                #                       self.rois[p][0]-self.roi_width:self.rois[p][0]+self.roi_width,
                #                       self.rois[p][1]-self.roi_width:self.rois[p][1]+self.roi_width].sum()

                # new way use exact pixel location
                temp = self.newfp[self.n_ttl[0]+z]
                temp = temp[self.rois_pixels[p].T[:, 0],
                            self.rois_pixels[p].T[:, 1]]

                # divide by the number of pixels in the ROI - NOT SURE IF THIS IS CORRECT?!
                # TODO: these algorithms must match the default water disposal algorithms
                roi_sum0 = temp / self.rois_pixels[p][0].shape[0]

                # sum
                # TODO: not sure this is the correct function; to check literature
                roi_sum0 = np.nansum(roi_sum0)

                #self.rois_traces[p].append(roi_sum0)
                self.rois_traces[p,self.n_ttl[0]] = roi_sum0



            if self.verbose:
                print ("")
                print ("")

    #
    def update_ensembles(self):

        # wait for at least a few frames to be grabbed (usually at least enough to apply smoothing window)
        if self.n_ttl[0]<self.rois_smooth_window:
            #for p in range(self.rois.shape[0]):
            self.ensemble_activity_realtime[:] = 0

        # update each ensemble based on some smoothing function
        else:
            for p in range(len(self.rois_pixels)):

                # grab last 5 frames (e.g.)
                if self.verbose:
                    print ("n_ttl: ", self.n_ttl)
                    print ("self.rois_smooth_window: ", self.rois_smooth_window)
                    print ("index 1: ", self.n_ttl-self.rois_smooth_window, " index 2: ", self.n_ttl)
                    print ("sefl rois traces[p]: ", self.rois_traces[p])

                # grab the last N time points
                temp = self.rois_traces[p, self.n_ttl[0]-self.rois_smooth_window:self.n_ttl[0]]

                # smooth them using linear decay/trinagle function
                # TODO: A BETTER SMOOTHING FUNCTION NEEDS TO BE IMPLEMENTED
                temp = temp*self.smoothing_kernel

                #
                if self.verbose:
                    print ("temp: ", temp)
                    print ("smooth function: ", self.smoothing_kernel)

                # take largest value
                # TODO: is this the correct approach?  UNCLEAR
                # TODO: MAKE SURE THIS METHOD MATCHES THE OFFLINE THRESHOLD CALCULATIONS
                temp = np.median(temp)

                #
                self.ensemble_activity_realtime[p] = temp

                # TODO: perhaps keep track of all realtime ensemble activity
                # i.e. save it somewhere, we might need it offline

        #
        if self.verbose:
            print (" ensembles realtime: ", self.ensemble_activity_realtime)


    def tone_off(self):

        # turn toneplayback off
        #  freq = 0
        #  np.save(self.fname_freq,freq)

        pass

    def compute_ensemble_to_tone_state(self):

        # for now we use a simple scaled difference
        # TODO: Mariona to finish up this function
        # compute ensemble -> tone trasnfer function
        # self.tone_frequency  <--- this should be the variable to change
        # - it is automatically monitored by a separate process

        self.tone_frequency[0] = np.random.randint(1000,18000)
        #print ("bmi computed tone: ", self.tone_frequency)


    def update_tone(self):
        ''' This function is not required any more, as we run a separate process
            tracking our self.tone_frequency variable
        '''

        # update current tone use self.freq
        #
        #  np.save(self.fname_freq,freq)

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

        #
        np.savez(os.path.join(self.fname_root_path,
                             "bmi_results.npz"),
                 ttl_voltages = self.ttl_voltages,
                 ttl_n_computed = self.ttl_n_computed,
                 ttl_n_detected = self.ttl_n_detected,
                 abs_times = self.abs_times,
                 ttl_times = self.ttl_times,
                 rois_traces = self.rois_traces)

