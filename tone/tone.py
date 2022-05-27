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
############### TONE PLAYER CLASS ###############
#################################################
#
class PlayTone():
    ''' Class that play tones computed by BMI code
        Input: shared memory int64 that gives the frequecny computed in BMI
        Output: NI card output port sinusoid usually of 0.1 sec

        TODO: We must figure out the real relationship between the played tone and detected frequncy

    '''

    def __init__(self, shmem_tone_frequency):

        #
        self.shmem_tone_frequency = shmem_tone_frequency

        #
        self.simulation_flag = True

        # TODO: unclear what these units are?
        self.amp = 0.1  # tone amplitude in ?

        # TODO: unclear what the correct duration of tone play and update
        # TODO: for now we update at 10hz
        self.duration = 0.1

        #
        self.initialize_tone_frequency()

        #
        self.initialize_tone_playback()

        #
        while True:
            start = time.time()
            self.update_tone()
            #print ("   tone playtime: ", time.time()-start, "sec")



    #
    def make_tone(self, f, amp, duration):

        # TODO: Is there an easier way to generate tones on raw speakers???
        fs = 200000  # 200kHz    (Sample Rate)
        x = np.arange(int(fs * duration))

        # original list; takes too long to create it;
        # doulbe check the numpy array is ok ... but seems good
        # y = [amp * np.sin(2 * np.pi * f * (i / fs)) for i in x]
        # y = np.array(y)
        y2 = amp * np.sin(2 * np.pi * f * (x / fs))
        y2 = y2[:,None]

        #
        y_new = np.tile(y2, 1)

        return y_new

    def compute_ensemble_to_tone_state(self):

        # for now we use a simple scaled difference
        # TODO: Mariona to finish up this function
        # compute ensemble -> tone trasnfer function
        # self.tone_frequency  <--- this should be the variable to change
        # - it is automatically monitored by a separate process

        # compute the E1-E2 for current time point
        self.ensemble_diff_current = self.ensemble_activity[0, self.n_ttl[0]] - self.ensemble_activity[1, self.n_ttl[1]]

        #
        self.tone_frequency = transfer_function(self.ensemble_diff_current)
        #
        # self.tone_frequency[0] = np.random.randint(1000,18000)
        # print ("bmi computed tone: ", self.tone_frequency)

    #
    def update_tone(self):

        # quickly make tone
        # TODO: check a couple of things:
        # TODO: 1) whether this is too slow and we end up buffering which not real time
        # TODO: 2) what is the shortest/correct duration to play a tone (probably >10hz) that we wont' notice)

        # make sure you send a copy of the tone, not the tone
        tone = self.make_tone(self.tone_frequency.copy(),
                              self.amp,
                              self.duration)

        #print ("tone: ", self.tone_frequency, "hz")

        if self.simulation_flag:
            return

        self.audio_Writer.write_many_sample(tone)

    #
    def initialize_tone_playback(self):

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
    def initialize_tone_frequency(self):
        #
        print("  tone freq memory name : ", self.shmem_tone_frequency)

        aa = np.zeros((1,), dtype=np.int64)

        # get the rois_traces from the shared memory name
        self.existing_shm_tone_frequency = shared_memory.SharedMemory(name=self.shmem_tone_frequency)
        # existing_shm = shared_memory.SharedMemory(name='testname')

        print("existing shm: ", self.existing_shm_tone_frequency)

        #
        self.tone_frequency = np.ndarray(aa.shape,
                                 dtype=aa.dtype,
                                 buffer=self.existing_shm_tone_frequency.buf)

        #
        print("  loaded tone freq: ", self.tone_frequency)

        #
        self.tone_frequency_last = self.tone_frequency[0].copy()

