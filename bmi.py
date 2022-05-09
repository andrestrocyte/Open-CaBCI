'''
  
  C. Mitelut; github: "catubc"; mitelutco@gmail.com

'''

import time
import os
import nidaqmx
from nidaqmx.constants import (AcquisitionType)  # https://nidaqmx-python.readthedocs.io/en/latest/constants.html
import numpy as np
from nidaqmx.constants import TerminalConfiguration
import tqdm # tdqm
from tqdm import notebook
import os

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

        ttl_val = self.ttl[self.index:self.index+number_of_samples_per_channel]
        self.index += number_of_samples_per_channel

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
    '''

    def __init__(self,
                 simulation_mode,
                 fname_root_path,
                 fname_fluorescence,
                 fname_rois,
                 fname_freq,
                 fname_ttl,
                 sampleRate_2P,
                 n_frames):

        #
        self.simulation_mode = simulation_mode

        #
        self.fname_root_path = fname_root_path
        self.fname_fluorescence = fname_fluorescence
        self.fname_rois = fname_rois
        self.fname_freq = fname_freq
        self.fname_ttl = fname_ttl

        # NOT SURE IF REQUIRED... TO DELETE
        #  flag was probably used during development toskip the reading step; 
        self.read_data_flag = True

        
        # number of frames to run BMI for
        self.n_frames = n_frames

        # Define variables
        self.sampleRate_NI = 1E3     # Sample rate of NI card

        #
        self.ttl_pts = 1  			 # number of values to read from NI card - usually we read a single value to avoid buffering issues

        #
        self.sampleRate_2P = sampleRate_2P	    # Sample rate of BScope

        #
        self.n_frames_to_be_acquired = n_frames   # Number of frames from BScope

        #
        self.rois_smooth_window = 5   				# Number of frames to use to smooth the ROI traces
                                                    # to be developed/changed further
        # start the ttl frame counter at 0
        self.ttl_computed = 0


        # initizlie ROIs using either a text file or more specific code
        #   for now simple version uses some random centres in the imaging file
        #   MUST CHANGE
        self.initialize_ROIs()

        # initialize all arrays to be used:
        self.initialize_arrays()

        # initialize progress toolbar
        self.pbar = tqdm.tqdm(total=self.n_frames_to_be_acquired,
                         position=0,
                         leave=True,
                         ascii=True)  # Init pbar

    #
    def initialize_arrays(self):

        #
        self.ttl_values = []			# array to hold ttl data being read
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
        self.ttl_voltages = []          # ttl_voltages
    #
    def initialize_ROIs(self):
        '''
            Initialize the ROIs and ensemble arrays to be used below

            TODO: Must properly transfer ROIs to this function not just use a box aroudn a point of interest
        '''

        # load ROI centres from disk;
        #  TODO run proper ROI detection with irregular shape et.
        self.rois = np.loadtxt(self.fname_rois, delimiter = ',', dtype=np.int32)
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

        # contains the realtime value of the ensembel state (i.e. no history)
        self.ensemble_activity_realtime = np.zeros(self.rois.shape[0])

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

        # abssolute start time
        self.start = time.perf_counter()

        # make progress bar toolbox

        #
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

                #
                self.bmi_update()

                # update trigger time
                self.previous_trigger = self.now

                #
                self.pbar.update(1)

            #
            self.prev_min = self.min_
            self.prev_max = self.max_

            #

        # save all data acquried during recording
        # TODO: try to save this on the fly if possible to avoid loosing data during crashes
        self.save_data()


    def initialize_ttl_reader(self):

        #
        if self.simulation_mode == True:
            self.task_ttl = Simulation(self.fname_ttl)
        else:
            self.task_ttl = nidaqmx.Task('bmi_online')

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

        # Initialize ttl arrays; a

        # if no ttl values read or computed means this is the first time we are udpating BMI
        #  - need to set the memmap for the data (more to be explained here)
        #  -

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
                
            else:
                time_passed = self.now-self.start
                self.ttl_computed = round((self.now-self.start)*self.sampleRate_2P)

                # 
                if self.verbose:
                    print (" time passed: ", time_passed, "   bmi_update self.ttl_computed: ", self.ttl_computed)
    
    def trigger_reward(self):

        # generate water reward

        pass

    def post_reward_state(self):

        # disable tone playback;
        self.tone_off()


        LENGTH = 10  # Number of iterations required to fill pbar

        pbar = tqdm(total=LENGTH)  # Init pbar

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
            if self.verbose:
                print ("")
                print ("")

            
    #
    def update_ensembles(self):

        # wait for at least a few frames to be grabbed (usually at least enough to apply smoothing window)
        if self.n_ttl<self.rois_smooth_window:
            #for p in range(self.rois.shape[0]):
            self.ensemble_activity_realtime[:] = 0

        # update each ensemble based on some smoothing function
        else:
            for p in range(self.rois.shape[0]):

                # grab last 5 frames (e.g.)
                if self.verbose:
                    print ("n_ttl: ", self.n_ttl)
                    print ("self.rois_smooth_window: ", self.rois_smooth_window)
                    print ("index 1: ", self.n_ttl-self.rois_smooth_window, " index 2: ", self.n_ttl)
                    print ("sefl rois traces[p]: ", self.rois_traces[p])
                temp = self.rois_traces[p][self.n_ttl-self.rois_smooth_window:self.n_ttl]

                # scale using a linear decay/trinagle function
                # TODO: THIS IS NOT CORRECT; MORE NEEDS TO BE DONE HERE
                if self.verbose:
                    print ("temp: ", temp)
                    print ("smooth function: ", self.smooth_function)
                temp = temp*self.smooth_function

                # take largest value
                temp = np.max(temp)

                #
                self.ensemble_activity_realtime[p] = temp

        #
        if self.verbose:
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


# #######################################
# fname_root_path = r"D:\User Training"
# fname_fluorescence = r"D:\User Training\Readtest1\Image_001_001.raw"
# fname_freq =  r"F:\freq.npy"
# fname_rois = r"D:\User Training\rois.txt"

# # 			
# sampleRate_2P = 30
# n_frames = 10 * sampleRate_2P                   # number of seconds to run the BMI 
# simulation_mode = True							# Run BMI in simulation mode (i.e. don't need Bscope input)

# #
# bmi = BMI(simulation_mode,
# 			fname_root_path,
# 			fname_fluorescence,
# 			fname_rois,
# 			fname_freq,
# 			n_frames,
# 			sampleRate_2P)
# bmi.run_BMI()
