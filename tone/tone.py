import matplotlib.pyplot as plt
import nidaqmx
from nidaqmx.constants import (AcquisitionType)  # https://nidaqmx-python.readthedocs.io/en/latest/constants.html
from nidaqmx.constants import TerminalConfiguration
import tqdm # tdqm
import os
from utils.utils import ensemble_to_tone_transfer_function, get_octave_frequencies
import time
import numpy as np
from multiprocessing import shared_memory
from nidaqmx import stream_writers

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
                        simulation_flag,):

        #
        self.shmem_termination_flag = shmem_termination_flag

        #
        self.simulation_flag = simulation_flag
        #self.simulation_flag = False

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

        # TODO: unclear what these units are?
        self.amp = 0.05  # tone amplitude in ?

        # TODO: unclear what the correct duration of tone play and update
        # TODO: for now we update at 10hz
        self.duration = 0.1

        # TODO: unclear what these units are?
        self.water_spout_ttl_duration = 50000  # duration of water pulse in microseconds

        #
        self.water_spout_ttl_voltage = 5    # water spout voltage in millivolts (?)

        #
        self.initialize_tone_state()

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
        while True:

            #
            self.update_water_spout()

            #
            self.update_tone()

            #
            if self.termination_flag:
                print ("...EXITING TONE CLASS...")
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

        data = np.load(self.fname_roi_pixels_and_thresholds)
        self.low_threshold = data['low_threshold']
        self.high_threshold = data['high_threshold']

        #
        self.low_freq = data['low_freq']
        self.high_freq = data['high_freq']

    #
    def make_tone(self, f, amp, duration):

        # TODO: Is there an easier way to generate tones on raw speakers???
        fs = 200000  # 200kHz    (Sample Rate)
        x = np.arange(int(fs * duration))

        # original list; takes too long to create it;
        # TODO: double check the numpy array versio is identical ... but seems good
        # y = [amp * np.sin(2 * np.pi * f * (i / fs)) for i in x]
        # y = np.array(y)
        y2 = amp * np.sin(2 * np.pi * f * (x / fs))
        y2 = y2[:,None]

        #
        y_new = np.tile(y2, 1)

        return y_new

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
        self.tone_state[0] = ensemble_to_tone_transfer_function(self.ensemble_state,
                                                                self.low_freq,
                                                                self.high_freq,
                                                                self.low_threshold,
                                                                self.high_threshold
                                                                )
        #print ("tone: ", self.tone_state)

    #
    def update_tone(self):

        # quickly make tone
        # TODO: check a couple of things:
        # TODO: 1) whether this is too slow and we end up buffering which not real time
        # TODO: 2) what is the shortest/correct duration to play a tone (probably >10hz) that we wont' notice)

        #
        self.compute_ensemble_to_tone_state()

        # make sure you send a copy of the tone, not the tone
        tone_data = self.make_tone(self.tone_state[0].copy(),
								   self.amp,
								   self.duration)

        #
        if self.simulation_flag:
            return
		
        #print ("tone data: ", tone_data.shape)
		
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
    def initialize_tone_state(self):
        #
        #print("  ensemble state memory name : ", self.shmem_tone_state)

        aa = np.zeros((1,), dtype=np.float32)

        # get the rois_traces from the shared memory name
        self.existing_shm_tone_state = shared_memory.SharedMemory(name=self.shmem_tone_state)
        #print("existing shm: ", self.existing_shm_tone_state)

        #
        self.tone_state = np.ndarray(aa.shape,
                                 dtype=aa.dtype,
                                 buffer=self.existing_shm_tone_state.buf)

        #
        #print("  TONE state: ", self.tone_state)

    #
    def initialize_ensemble_state(self):
        #
        #print("  ensemble state memory name : ", self.shmem_ensemble_state)

        aa = np.zeros((1,), dtype=np.float32)

        # get the rois_traces from the shared memory name
        self.existing_shm_ensemble_state = shared_memory.SharedMemory(name=self.shmem_ensemble_state)
        #print("existing shm: ", self.existing_shm_ensemble_state)

        #
        self.ensemble_state = np.ndarray(aa.shape,
                                 dtype=aa.dtype,
                                 buffer=self.existing_shm_ensemble_state.buf)

        #
        #print("  TONE CLASS loaded ensemble_state: ", self.ensemble_state)

        #
        #self.ensemble_state_last = self.ensemble_state[0].copy()

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

        print('   releasing water for ', self.water_spout_ttl_duration,
              "microsec, at ", self.water_spout_ttl_voltage, " mV")

        # what is the point of this?
        self.water_reward[0] = 0

        if self.simulation_flag:
            return

        # close the audio writer
        print ("closeing audio writer")
        self.close_audio_writer()

        # initialize the output function for water dispesning
        print ("initilizeding water writer")
        self.initialize_water_writer()

        # put water state to 5volts
        # TODO: not sure the loop is required? perhaps just write it once and then wait for duration!?
        # THIS FUNCTION WRITES 5v to the output for 10000 microseconds
        start = time.time()
        for p in range(self.water_spout_ttl_duration):
            #print ("water reward loop: ", p)
            self.water_Writer.write_one_sample(self.water_spout_ttl_voltage)
        print (" >>>>>>>>>>>>>>>>>>>>>>>>>>>>released water for: ", time.time()-start, " sec. (Closing water port)")

        # return water ttl state to 0volts
        for p in range(self.water_spout_ttl_duration):
            self.water_Writer.write_one_sample(0)

        # close water writer
        self.close_water_writer()

        #
        self.initialize_audio_writer()


