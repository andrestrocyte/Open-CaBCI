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
############### PLOTTING CLASS ##################
#################################################
#
class PlotROIs():
    ''' Class that visualizes the BMI ROI readouts as traces
        Input: shared memory

    '''

    #
    def __init__(self,
                 shmem_rois_traces,
                 shmem_n_ttl,
                 rois_traces_shape):


        print ("INITALIZED PLOTROIS FUCTIONS...")
        import matplotlib
        # matplotlib.use('qtagg')
        #%matplotlib tk
        import matplotlib.pyplot as plt
        plt.ion()

        #
        self.sampleRate_2P = 30
        print ("...assuming sampling rate is ", self.sampleRate_2P, "hz")

        #
        self.verbose2 = False


        self.shmem_rois_traces = shmem_rois_traces
        self.rois_traces_shape = rois_traces_shape
        self.shmem_n_ttl = shmem_n_ttl
        #
        self.initialize_rois_traces()

        #
        self.initalize_n_ttl()

        # #
        self.make_roi_plots()

        #
        self.ctr = 0

        #
        #ctr = 0
        while True:
            self.update_plots()
            #print ("looping ", self.ctr)# plot last X values depending
            #self.ctr+=1
            #time.sleep(.1)
    #
    def initalize_n_ttl(self):

        #
        print ("  nttl memory name : ", self.shmem_n_ttl)

        aa = np.zeros((1,), dtype=np.int64)

        # get the rois_traces from the shared memory name
        self.existing_shm_n_ttl = shared_memory.SharedMemory(name=self.shmem_n_ttl)
        #existing_shm = shared_memory.SharedMemory(name='testname')

        print ("existing shm: ", self.existing_shm_n_ttl)


        #
        self.n_ttl2 = np.ndarray(aa.shape,
                                 dtype=aa.dtype,
                                 buffer=self.existing_shm_n_ttl.buf)

        #
        print ("  loaded n_ttl: ", self.n_ttl2)

        #
        self.n_ttl_last = self.n_ttl2[0].copy()

           #
    def initialize_rois_traces(self):

        print ("  Plotter loaded: ", self.shmem_rois_traces,
               " size: ", self.rois_traces_shape)

        # get the rois_traces from the shared memory name
        self.existing_shm_rois_traces = shared_memory.SharedMemory(name=self.shmem_rois_traces)

        #
        print ("existing shm traces*****************: ",
               self.existing_shm_rois_traces)

        #
        self.rois_traces = np.ndarray((self.rois_traces_shape[0],
                                       self.rois_traces_shape[1]),
                                       dtype=np.float32,
                                       buffer=self.existing_shm_rois_traces.buf)

        #
        #print ("  loaded rois_traces: ", self.rois_traces)
        #print ("rois traces: ", self.rois_traces.shape)
        #print ("rois sums: ", self.rois_traces.sum())

        #print ("TOTAL NANS IN roi traces INSIDE ", np.isnan(self.rois_traces).sum())


    #shmem_rois_traces
    def make_roi_plots(self):

        #
        self.plot_y_scale = 1000

        #
        self.fig = plt.figure(figsize=(10,5))

        self.ax = self.fig.add_subplot(111)
        #self.ax.set_ylim(0, self.plot_y_scale*len(self.rois_traces)*1.1)
        self.ax.set_ylim(0, 10000)
        self.ax.set_xlim(-10,0)
        self.ax.set_xlabel("Time (sec)")
        #self.ax.set_yticks([])

        # initialize time:
        print ("TOTAL NANS IN roi traces OUTSIDE ", np.isnan(self.rois_traces).sum())

        self.plot_times = np.arange(-10*self.sampleRate_2P,0,1)/self.sampleRate_2P

        # make a list to hold the matplotlib line objects
        self.time_course_objects = []
        for k in range(self.rois_traces.shape[0]):

            #
            #print (k, self.plot_times.shape, self.rois_traces[k,:10*self.sampleRate_2P].shape)
            y_values = self.rois_traces[k,:10*self.sampleRate_2P]-1000*k
            #print ("yvalues iniatlized np.nans: ", np.isnan(y_values).sum(),
            #       np.isnan(self.plot_times).sum())
            lineobject, = self.ax.plot(self.plot_times,
                                       #self.rois_traces[k,-self.plot_times:]-1000*k,  # plot last X values depending on length of plttimes
                                       y_values,  # plot last X values depending on length of plttimes
                                       #'r-'
                                       )  # Returns a tuple of line objects, thus the comma
            #
            self.time_course_objects.append(lineobject)

        # self.ax.clear()

        #
        self.ax.set_title("DON T CLOSE MANUALLY")

        #
        self.fig.canvas.flush_events()

        #
        self.fig.canvas.draw()

        #
        self.fig.canvas.flush_events()

        #
        #self.fig.clf()

        # cache the background
        self.axbackground = self.fig.canvas.copy_from_bbox(self.ax.bbox)

        #
        plt.show(block=False)
        #plt.show(block=False)

        print (" DONE MAKING ROI LINE PLOTS")


    #
    def update_plots(self):

        '''  Function to dynamically update our plots
            TODO: move it to a separate core/process to not interfere with the main BMI loop/analysis code

        '''

        # update ROI line plots
        self.n_ttl_current = self.n_ttl2[0].copy()

        if self.n_ttl_current >= (self.n_ttl_last+1):
            pass
        else:
            return

        #
        if self.verbose2:
            start = time.time()

        # restore background
        self.fig.canvas.restore_region(self.axbackground)

        #
        self.n_ttl_last = self.n_ttl_current.copy()

        x_values = np.arange(0, min(self.n_ttl_current, 300), 1) / 30 - 10
        for k in range(self.rois_traces.shape[0]):

            #

            #x_values = self.plot_times[x_values]

            #
            y_values = self.rois_traces[k,max(0,self.n_ttl_current-300):self.n_ttl_current]+1000*k

            #
            #print (x_values.shape, y_values.shape)
            self.time_course_objects[k].set_data(x_values,
                                                y_values
                                                )
        #
        self.fig.canvas.restore_region(self.axbackground)

        # fill in the axes rectangle
        self.fig.canvas.blit(self.ax.bbox)

        for k in range(len(self.rois_traces)):
            self.ax.draw_artist(self.time_course_objects[k])

        #
        self.fig.canvas.blit(self.ax.bbox)

        #
        self.fig.canvas.flush_events()

        #
        if self.verbose2:
            print ("time for updating graph: ", time.time()-start)
            print ('')
            print ('')

