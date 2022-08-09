import nidaqmx
from nidaqmx.constants import (AcquisitionType)  # https://nidaqmx-python.readthedocs.io/en/latest/constants.html

from utils.utils import ensemble_to_tone_transfer_function, get_octave_frequencies
import time
import numpy as np
from multiprocessing import shared_memory
from nidaqmx import stream_writers

#################################################
############### TONE PLAYER CLASS ###############
#################################################
#
class WaterReward():
    ''' Class that play tones computed by BMI code
        Input: shared memory int64 that gives the frequecny computed in BMI
        Output: NI card output port sinusoid usually of 0.1 sec

        TODO: We must figure out the real relationship between the played tone and detected frequncy

    '''

    def __init__(self, shmem_water_reward,
                       shmem_termination_flag,
					   simulation_flag,
					   ):


        #
        self.shmem_termination_flag = shmem_termination_flag

        #
        self.simulation_flag = simulation_flag

        #
        self.shmem_water_reward = shmem_water_reward

        #
        self.initialize_termination_flag()

        # not currently used
        # self.water_sleep_time = 0.01   # amoutn of seconds to sleep while looping over updates

        #
        self.initialize_water_reward_variable()

        #
        self.initialize_water_spout()

        # TODO: unclear what these units are?
        self.water_spout_ttl_duration = 12000  # duration of water pulse in microseconds

        #
        self.water_spout_ttl_voltage = 5    # water spout voltage in millivolts (?)

        #
        while True:
            #start = time.time()
            self.update_water_spout()
            #time.sleep(self.water_sleep_time)

            #
            if self.termination_flag:
                print ("... STOPPING WATER CLASS...")
                break

        #
        if self.simulation_flag==False:
            self.water_Task.stop()
            self.water_Task.close()

        #
        quit()

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

        print ("WATER CLASS - DONE termination flag - ")


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
    def initialize_water_spout(self):

        #
        if self.simulation_flag:
            return

        #  Initialize water task
        self.water_Task = nidaqmx.Task()

        #
        self.water_Task.ao_channels.add_ao_voltage_chan('Dev3/ao1')
        self.water_Task.timing.cfg_samp_clk_timing(rate=1000000,
                                              # samps_per_chan=100,  # in continuos mode this is the buffer
                                              sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)
                                              # sample_mode=nidaqmx.constants.AcquisitionType.FINITE)

        #
        self.water_Writer = nidaqmx.stream_writers.AnalogSingleChannelWriter(self.water_Task.out_stream,
                                                                             auto_start=True)
                                                                
    #
    def update_water_spout(self):

        if self.water_reward==0:
            return

        #print('   releasing water for ', self.water_spout_ttl_duration,
        #      "microsec, at ", self.water_spout_ttl_voltage, " mV")

        #
        self.water_reward[0] = 0

        if self.simulation_flag:
            return

        # put water state to 5volts
        # TODO: not sure the loop is required? perhaps just write it once and then wait for duration!?
        # THIS FUNCTION WRITES 5v to the output for 10000 microseconds
        for p in range(self.water_spout_ttl_duration):
            #print ("water rewwrd loop: ", p)
            self.water_Writer.write_one_sample(self.water_spout_ttl_voltage)
		
		# 
        print (" turned water reward OFF")

        # return water ttl state to 0volts
        self.water_Writer.write_one_sample(0)
