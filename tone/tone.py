import matplotlib.pyplot as plt
import nidaqmx
from nidaqmx.constants import (AcquisitionType)  # https://nidaqmx-python.readthedocs.io/en/latest/constants.html
from nidaqmx.constants import TerminalConfiguration
import tqdm  # tdqm
import os
from utils.utils import ensemble_to_tone_transfer_function_absolute, ensemble_to_tone_transfer_function_high_and_low, get_octave_frequencies
import time
import numpy as np
from multiprocessing import shared_memory
from nidaqmx import stream_writers
from scipy.signal import chirp, spectrogram
from scipy import stats


#################################################
############### TONE PLAYER CLASS ###############
#################################################
#
class PlayTone():
    ''' Class that play tones computed by BMI code
        Input: shared memory int64 that gives the frequecny computed in BMI
        Output: NI card output port sinusoid usually of 0.1 sec

        TODO: We must figure out the real relationship between the played tone and detected frequncy

    '''

    def __init__(self, fname_roi_pixels_and_thresholds,
                 shmem_ensemble_state,
                 shmem_tone_state,
                 shmem_termination_flag,
                 shmem_water_reward,
                 shmem_reward_lockout_counter,
                 shmem_dynamic_reward_lockout_state,
                 shmem_white_noise_state,
                 shmem_alignment_flag,
                 water_vol_ttl,
                 simulation_flag,
                 calibration_flag,
                 sleep_time_sec):

        #
        self.sleep_time_sec = sleep_time_sec

        #
        self.water_vol_ttl = water_vol_ttl

        #
        self.shmem_alignment_flag = shmem_alignment_flag

        #
        self.initialize_alignment_flag()

        #
        self.shmem_white_noise_state = shmem_white_noise_state

        #
        self.shmem_reward_lockout_counter = shmem_reward_lockout_counter

        #
        self.shmem_dynamic_reward_lockout_state = shmem_dynamic_reward_lockout_state

        #
        self.calibration_flag = calibration_flag

        #
        self.shmem_termination_flag = shmem_termination_flag

        #
        self.simulation_flag = simulation_flag

        #
        self.sampleRate_audio = 2E5

        #
        self.fname_roi_pixels_and_thresholds = fname_roi_pixels_and_thresholds

        #
        self.initialize_thresholds()

        #
        self.shmem_ensemble_state = shmem_ensemble_state

        #
        self.shmem_tone_state = shmem_tone_state

        #
        self.shmem_water_reward = shmem_water_reward

        # TODO: unclear what these units are; likely volts - but need to convert to dCB
        #self.amplitude = 0.07  # tone amplitude in ?
        self.amplitude = 0.01  # tone amplitude in ?

        # TODO: unclear what the correct duration of tone play and update
        # TODO: for now we update at 10hz
        self.duration = 0.1

        # number of seconds to play the reward tone of ~16Khz
        self.n_sec_reward_tone = 2
        
        # TODO: unclear what these units are?
        self.water_spout_ttl_duration = self.water_vol_ttl   # duration of water pulse in microseconds

        #
        self.water_spout_ttl_voltage = 5  # water spout voltage in millivolts (?)

        #
        self.initialize_tone_state()

        #
        self.initialize_white_tone_state()

        #
        self.octave_step = 0.25

        #
        self.initialize_octave_frequencies()

        #
        self.initialize_ensemble_state()

        #
        self.initialize_audio_writer()

        #
        self.initialize_termination_flag()

        #
        self.initialize_water_reward_variable()

        # ONLY initialize when reward is present
        # self.initialize_water_spout()

        #
        self.make_frequency_sweep()

        #
        self.make_white_noise()

        #
        self.initialize_reward_lockout_counter()
        
        #
        self.initialize_dynamic_reward_lockout_state()

        #
        while True:

            #
            self.update_water_spout()

            #
            self.update_tone()

            #
            if self.termination_flag:
                print("...EXITING TONE CLASS...")
                break

    #
    def close_audio_writer(self):

        #
        self.audio_Task.stop()
        self.audio_Task.close()

    #
    def close_water_writer(self):

        #
        self.water_Task.stop()
        self.water_Task.close()

    #
    def initialize_octave_frequencies(self):

        #
        self.octave_freqs = get_octave_frequencies(self.low_freq,
                                                   self.high_freq,
                                                   self.octave_step)

    #
    def initialize_reward_lockout_counter(self):

        #
        aa = np.zeros((1,), dtype=np.int64)

        # get the rois_traces from the shared memory name
        self.existing_shm_reward_lockout_counter = shared_memory.SharedMemory(name=self.shmem_reward_lockout_counter)

        #
        self.reward_lockout_counter = np.ndarray(aa.shape,
                                           dtype=aa.dtype,
                                           buffer=self.existing_shm_reward_lockout_counter.buf)

    #
    def initialize_dynamic_reward_lockout_state(self):

        #
        
        # make a numpy array to hold the rois_traces
        aa = np.zeros((1), dtype=np.int32)
        self.existing_dynamic_reward_lockout_state = shared_memory.SharedMemory(name=self.shmem_dynamic_reward_lockout_state)



        #
        self.dynamic_reward_lockout_state = np.ndarray(aa.shape,
													   dtype=aa.dtype,
													   buffer=self.existing_dynamic_reward_lockout_state.buf)
		
		#

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
    def initialize_thresholds(self):

        try:
            data = np.load(self.fname_roi_pixels_and_thresholds)
            self.low_threshold = data['low_threshold']
            self.high_threshold = data['high_threshold']

            #
            self.low_freq = data['low_freq']
            self.high_freq = data['high_freq']
        except:
            print(" couldn't find roi_pixels file ----> assuming it's a calibration session (setting default freqs)")
            self.low_freq = 2000
            self.high_freq = 18000
            self.low_threshold = 0
            self.high_threshold = 10

    #
    def make_tone(self, f, amp, duration):

        # TODO: Is there an easier way to generate tones on raw speakers???
        fs = self.sampleRate_audio  # 200kHz    (Sample Rate)
        x = np.arange(int(fs * duration))

        # original list; takes too long to create it;
        # TODO: double check the numpy array versio is identical ... but seems good
        # y = [amp * np.sin(2 * np.pi * f * (i / fs)) for i in x]
        # y = np.array(y)
        y2 = amp * np.sin(2 * np.pi * f * (x / fs))
        y2 = y2[:, None]

        #
        y_new = np.tile(y2, 1)

        return y_new

    #
    def make_frequency_sweep(self):

        #
        t = np.linspace(0, self.duration, int(self.sampleRate_audio * self.duration))

        #
        self.freq_sweep = chirp(t, f0=self.low_freq, f1=self.high_freq, t1=self.duration,
                                method='linear') * self.amplitude

    #
    def make_white_noise(self):

        #
        self.white_noise = stats.truncnorm(-1, 1,
                                           scale=min(2 ** 16, 2 ** self.amplitude)).rvs(
            int(self.sampleRate_audio * self.duration))

        #

    def compute_ensemble_to_tone_state(self):

        # for now we use a simple scaled difference
        # TODO: Mariona to finish up this function
        # compute ensemble -> tone trasnfer function
        # self.tone_frequency  <--- this should be the variable to change
        # - it is automatically monitored by a separate process

        # E1-E2 for current time point is already computed for us in BMI class
        # - it is contained in shared memory variable self.ensemble_state

        # compute the tone state in Hz
        # TODO: NOTE: This is the only place that the tone_state variable is ocmputed from the ensemble state
        self.tone_state[0] = ensemble_to_tone_transfer_function_high_and_low(self.ensemble_state,
                                                                             self.low_freq,
                                                                             self.high_freq,
                                                                             self.low_threshold,
                                                                             self.high_threshold,
                                                                             self.octave_freqs
                                                                            )

    #
    def play_reward_tone(self):

        ''' Playing white noise for reward tone
        '''

        #
        print("Playing reward tone: SET TO freq sweep)")

        #
        if self.simulation_flag:
            return

        #
        self.audio_Writer.write_many_sample(self.freq_sweep.squeeze())

    #
    def play_reward_tone_high_freq(self):

        ''' Playing white noise for reward tone
        '''

        #
        #print("Playing reward tone: SET TO high frequency")

        #
        self.tone_state[0] = 16000

        #
        if self.simulation_flag:
            return

		# TODO : this is not ideal; this fuctnion shoul donly play the tone, not do anything else,
		#      including generating tone data etc.

        #
        tone_data = self.make_tone(self.tone_state[0], self.amplitude, self.duration)
        
        #
        self.audio_Writer.write_many_sample(tone_data.squeeze())  

    #
    def update_tone(self):

        # quickly make tone
        # TODO: check a couple of things:
        # TODO: 1) whether this is too slow and we end up buffering which not real time
        # TODO: 2) what is the shortest/correct duration to play a tone (probably >10hz) that we wont' notice)

        #

        #######################################################
        ############# NO TONE FOR CERTAIN CONDITIONS ##########
        #######################################################
        # overwrite tone if we are in lockout state
        # play inaudible tone in calibration mode ; OR
        #   - while the reward lockout counter is locking out playback
        #   - while dynamic reward lockout is still to ON (i.e. 1)
        if (self.calibration_flag == True) or (self.reward_lockout_counter[0]>0) or (self.dynamic_reward_lockout_state==1):
            self.tone_state[0] = 100

        # do not play an udible tone
        elif self.alignment_flag[0]==1:
            self.tone_state[0] = 100

        # compute ensembel to tone by default
        else:
            self.compute_ensemble_to_tone_state()
            #time.sleep(5)
        #######################################################
        ################### WHITE NOISE LOOP ##################
        #######################################################
        # overwrite tone if we are in noise session; grab a random frequency; overwrite the low freq state
        #   Don't play tone during alignment sessions
        while self.shmem_white_noise_state[0]==1:

            # check if we should exit
            if self.termination_flag:
                print("...EXITING TONE CLASS...")
                break

            self.tone_state[0] = np.random.choice(np.arange(self.low_freq, self.high_freq,1))
            tone_data = self.make_tone(self.tone_state[0].copy(),
                                       self.amplitude,
                                       0.0002)

            # play tone only in non-simulation mode
            if self.simulation_flag==False and self.alignment_flag[0]==0:
                self.audio_Writer.write_many_sample(tone_data.squeeze())

        #######################################################
        ################ MAKE AND PLAY TONES ##################
        #######################################################


        # do not play an udible tone
        if self.alignment_flag[0]==1:
            self.tone_state[0] = 100

        # make sure you send a copy of the tone, not the tone
        tone_data = self.make_tone(self.tone_state[0].copy(),
                                   self.amplitude,
                                   self.duration)

        # exit if in simulation mode
        if self.simulation_flag:
            return

        #
        self.audio_Writer.write_many_sample(tone_data.squeeze())

    #
    def initialize_audio_writer(self):

        # is this required here???
        # self.compute_ensemble_to_tone_state()

        #
        if self.simulation_flag:
            return

        #  Initialize speaker task;
        self.audio_Task = nidaqmx.Task()

        #
        self.audio_Task.ao_channels.add_ao_voltage_chan('Dev3/ao0')
        self.audio_Task.timing.cfg_samp_clk_timing(rate=self.sampleRate_audio,
                                                   # samps_per_chan=100,  # in continuos mode this is the buffer
                                                   sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)

        #
        self.audio_Writer = nidaqmx.stream_writers.AnalogSingleChannelWriter(self.audio_Task.out_stream,
                                                                             auto_start=True)

    #
      #
    def initialize_alignment_flag(self):
        #
        # print("  ensemble state memory name : ", self.shmem_tone_state)

        aa = np.zeros((1,), dtype=np.int32)

        # get the rois_traces from the shared memory name
        self.existing_shm_alignment_flag = shared_memory.SharedMemory(name=self.shmem_alignment_flag)
        # print("existing shm: ", self.existing_shm_tone_state)

        #
        self.alignment_flag = np.ndarray(aa.shape,
                                     dtype=aa.dtype,
                                     buffer=self.existing_shm_alignment_flag.buf)

    #
    def initialize_tone_state(self):

        #
        aa = np.zeros((1,), dtype=np.float32)

        # get the rois_traces from the shared memory name
        self.existing_shm_tone_state = shared_memory.SharedMemory(name=self.shmem_tone_state)

        #
        self.tone_state = np.ndarray(aa.shape,
                                     dtype=aa.dtype,
                                     buffer=self.existing_shm_tone_state.buf)

    #
    def initialize_white_tone_state(self):
        #
        # print("  ensemble state memory name : ", self.shmem_tone_state)

        aa = np.zeros((1,), dtype=np.float32)

        # get the rois_traces from the shared memory name
        self.existing_shm_white_tone_state = shared_memory.SharedMemory(name=self.shmem_white_noise_state)

        #
        self.shmem_white_noise_state = np.ndarray(aa.shape,
                                                 dtype=aa.dtype,
                                                 buffer=self.existing_shm_white_tone_state.buf)

    #
    def initialize_ensemble_state(self):
        #
        # print("  ensemble state memory name : ", self.shmem_ensemble_state)

        aa = np.zeros((1,), dtype=np.float32)

        # get the rois_traces from the shared memory name
        self.existing_shm_ensemble_state = shared_memory.SharedMemory(name=self.shmem_ensemble_state)

        #
        self.ensemble_state = np.ndarray(aa.shape,
                                         dtype=aa.dtype,
                                         buffer=self.existing_shm_ensemble_state.buf)


    #
    def initialize_water_reward_variable(self):

        #
        aa = np.zeros((1,), dtype=np.float32)

        # get the rois_traces from the shared memory name
        self.existing_shm_water_reward = shared_memory.SharedMemory(name=self.shmem_water_reward)

        #
        self.water_reward = np.ndarray(aa.shape,
                                       dtype=aa.dtype,
                                       buffer=self.existing_shm_water_reward.buf)

    #
    def initialize_water_writer(self):

        #
        if self.simulation_flag:
            return

        #  Initialize water task
        self.water_Task = nidaqmx.Task()

        #
        self.water_Task.ao_channels.add_ao_voltage_chan('Dev3/ao1')
        self.water_Task.timing.cfg_samp_clk_timing(rate=self.sampleRate_audio,
                                                   # samps_per_chan=100,  # in continuos mode this is the buffer
                                                   sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)
        # sample_mode=nidaqmx.constants.AcquisitionType.FINITE)

        #
        self.water_Writer = nidaqmx.stream_writers.AnalogSingleChannelWriter(self.water_Task.out_stream,
                                                                             auto_start=True)

    #
    def update_water_spout(self):

        if self.water_reward[0] == 0:
            return

        elif self.water_reward[0] == 1:

            print(' will release water for ', self.water_spout_ttl_duration,
                  "frames (@100000Hz), about ",str(round(self.water_spout_ttl_duration/100000,4)), "sec, at: ", self.water_spout_ttl_voltage, " mV")

            # skip water release for alignmetn mode
            if self.alignment_flag[0]==1:
                print ("exiting water reward - alignment mode")
                self.water_reward[0] = 0
                return

            # give water and cycle through NI card ttls...
            if self.simulation_flag==True:

                # set the tone to high reward threshold
                self.tone_state[0] = 16000

                # simulate playing the tone state for duration of time
                # play hightest tone to indicate reward  for specific amount of time
                # TODO: this essentially counts some frames; ~ correct, but not quite
                #sleep_time = self.n_sec_reward_tone*self.sleep_time_sec*self.duration
                #print ("sleep time: ", sleep_time)
                #time.sleep(sleep_time)

            else:
                #self.water_reward[0] = 0
                #return

                # close the audio writer
                # TODO: update this entire function to handle water + tone in parallel/simultaneously
                print("closing audio writer")
                self.close_audio_writer()

                # initialize the output function for water dispesning
                print("initializing water writer")
                self.initialize_water_writer()

                # put water state to 5volts
                # TODO: not sure the loop is required? perhaps just write it once and then wait for duration!?
                # THIS FUNCTION WRITES 5v to the output for 10000 microseconds
                start = time.time()
                for p in range(self.water_spout_ttl_duration):
                    # print ("water reward loop: ", p)
                    self.water_Writer.write_one_sample(self.water_spout_ttl_voltage)
                print(" released water for: ", time.time() - start,
                      " sec. (Closing water port)")

                # return water ttl state to 0volts
                # TODO Not clear we have to take so long to reset the water writer; maybe 100 time points is enough
                for p in range(self.water_spout_ttl_duration):
                    self.water_Writer.write_one_sample(0)

                # close water writer
                self.close_water_writer()

                # reopen the audio tone after water release
                # TODO: this should technically be simultaneous with the water release!!
                self.initialize_audio_writer()

                # play hightest tone to indicate reward  for specific amount of time
                for k in range(int(self.n_sec_reward_tone / self.duration)):
                    self.play_reward_tone_high_freq()

            # NOTE: this is set to negative so that during calibration so there's no other sounds outside of
            # otherwise during online bmi this is overwritten shortly after exiting this conditional
            # TODO: perhaps remove this, not necessary for calibration...
            # TODO: currently not used; but could be implemented for varying types of calibration paradigms;
            # - e.g. if water rewards are paired to tone during calbration (randomly or not), may wish to turn off
            #      the tones after
            #self.ensemble_state[0] = -30000

            # return tone to default state
            # TODO: this doesn't seem necessary as the rest of pipeline takes care of this

            # This resets the water reward back to 0 to turn off rewards in future
            self.water_reward[0] = 0
