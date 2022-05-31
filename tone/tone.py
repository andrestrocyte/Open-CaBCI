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
                        simulation_flag):

        #
        self.shmem_termination_flag = shmem_termination_flag

        #
        self.simulation_flag = simulation_flag

        #
        self.fname_roi_pixels_and_thresholds = fname_roi_pixels_and_thresholds

        #
        self.initialize_thresholds()

        #
        self.shmem_ensemble_state = shmem_ensemble_state

        #
        self.shmem_tone_state = shmem_tone_state

        # TODO: unclear what these units are?
        self.amp = 0.1  # tone amplitude in ?

        # TODO: unclear what the correct duration of tone play and update
        # TODO: for now we update at 10hz
        self.duration = 0.1

        #
        self.initialize_tone_state()

        #
        self.octave_step = 0.25

        #
        self.initialize_octave_frequencies()

        #
        self.initialize_ensemble_state()

        #
        self.initialize_tone_playback()

        #
        self.initialize_termination_flag()

        #
        while True:
            #start = time.time()
            self.update_tone()
            #time.sleep(0.01)
            if self.termination_flag:
                print ("...EXITING TONE CLASS...")
                break

        #
        quit()

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

        self.audio_Writer.write_many_sample(tone_data)

    #
    def initialize_tone_playback(self):

        self.compute_ensemble_to_tone_state()

        #
        if self.simulation_flag:
            return

        #  Initialize speaker task;
        self.audio_Task = nidaqmx.Task()

        #
        self.audio_Task.ao_channels.add_ao_voltage_chan('Dev3/ao0')
        self.audio_Task.timing.cfg_samp_clk_timing(rate=200000,
                                                  # samps_per_chan=100,  # in continuos mode this is the buffer
                                                  sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)

        #
        # sample_mode=nidaqmx.constants.AcquisitionType.FINITE)

        #
        self.audio_Writer = nidaqmx.stream_writers.AnalogSingleChannelWriter(audio_Task.out_stream,
                                                                             auto_start=True)


    #
    def initialize_tone_state(self):
        #
        print("  ensemble state memory name : ", self.shmem_tone_state)

        aa = np.zeros((1,), dtype=np.float32)

        # get the rois_traces from the shared memory name
        self.existing_shm_tone_state = shared_memory.SharedMemory(name=self.shmem_tone_state)
        #print("existing shm: ", self.existing_shm_tone_state)

        #
        self.tone_state = np.ndarray(aa.shape,
                                 dtype=aa.dtype,
                                 buffer=self.existing_shm_tone_state.buf)

        #
        print("  TONE state: ", self.tone_state)

    #
    def initialize_ensemble_state(self):
        #
        print("  ensemble state memory name : ", self.shmem_ensemble_state)

        aa = np.zeros((1,), dtype=np.float32)

        # get the rois_traces from the shared memory name
        self.existing_shm_ensemble_state = shared_memory.SharedMemory(name=self.shmem_ensemble_state)
        print("existing shm: ", self.existing_shm_ensemble_state)

        #
        self.ensemble_state = np.ndarray(aa.shape,
                                 dtype=aa.dtype,
                                 buffer=self.existing_shm_ensemble_state.buf)

        #
        print("  TONE CLASS loaded ensemble_state: ", self.ensemble_state)

        #
        self.ensemble_state_last = self.ensemble_state[0].copy()

