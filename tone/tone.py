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
        self.initialize_tone_display()

        #
        while True:
            self.update_tone()

            #
            self.update_tone_display()

    def initialize_tone_display(self):
        ''' '''
        #
        self.fig = plt.figure(figsize=(3, 3))

        self.text = self.fig.text(0.5,0.96, str(self.tone_frequency))

        # self.ax = self.fig.add_subplot(111)
        # text = self.ax.text(0.5, 1.100, str(self.tone_frequency),
        #         bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 5},
        #         transform=self.ax.transAxes, ha="center")
        #
        # self.ax.text.remove()
        #
        # self.ax.set_title(str(self.tone_frequency))
        #
        # #
        # self.fig.canvas.flush_events()
        #
        # #
        # self.fig.canvas.draw()
        #
        # #
        # self.fig.canvas.flush_events()
        #
        # #
        # # self.fig.clf()
        #
        # # cache the background
        # #self.axbackground = self.fig.canvas.copy_from_bbox(self.ax.bbox)

        #
        plt.show(block=False)


    def update_tone_display(self):

        # restore background
        #self.fig.canvas.restore_region(self.axbackground)

        #
        pass
        #self.text.remove()
        #self.text = self.fig.text(0.5,0.96, str(self.tone_frequency))

        #
        # self.ax.title.set_text(self.tone_frequency)
        # self.ax.text(0.5, 1.100, str(self.tone_frequency),
        #         bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 5},
        #         transform=self.ax.transAxes, ha="center")
        #
        # #
        # #self.fig.canvas.restore_region(self.axbackground)
        #
        # # fill in the axes rectangle
        # self.fig.canvas.blit(self.ax.bbox)
        #
        # #
        # self.fig.canvas.blit(self.ax.bbox)
        #
        # #
        # self.fig.canvas.flush_events()

    #
    def make_tone(self, f, amp, duration):

        # TODO: Is there an easier way to generate tones on raw speakers???
        fs = 200000  # 200kHz    (Sample Rate)
        x = np.arange(int(fs * duration))
        y = [amp * np.sin(2 * np.pi * f * (i / fs)) for i in x]
        y_new = np.tile(y, 1)

        return y_new

    #
    def update_tone(self):

        # quickly make tone
        # TODO: check a couple of things:
        # TODO: 1) whether this is too slow and we end up buffering which not real time
        # TODO: 2) what is the shortest/correct duration to play a tone (probably >10hz) that we wont' notice)

        tone = self.make_tone(self.tone_frequency, self.amp, self.duration)

        #print ("playing tone: ", self.tone_frequency, "hz")

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

