'''
  
  C. Mitelut; github: "catubc"; mitelutco@gmail.com

'''

import time
import numpy as np
from multiprocessing import shared_memory
import matplotlib.pyplot as plt

plt.ion()

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
                 fname_rois,
                 shmem_rois_traces,
                 shmem_n_ttl,
                 rois_traces_shape,
                 shmem_reward_times,
                 shmem_tone_state,
                 shmem_termination_flag,):

        # this is not really requried for visualizations (at this time anyways)
        # self.simulation_flag = simulation_flag

		#
        self.setup_complete = False

        #
        self.fname_rois = fname_rois

        #
        self.sampleRate_2P = 30
        print ("...assuming sampling rate is ", self.sampleRate_2P, "hz")

        #
        self.plotting_window_width = 30

        #
        self.verbose2 = False

        #
        self.shmem_termination_flag = shmem_termination_flag

        #
        self.shmem_rois_traces = shmem_rois_traces

        #
        self.rois_traces_shape = rois_traces_shape

        #
        self.shmem_n_ttl = shmem_n_ttl

        #
        self.shmem_reward_times = shmem_reward_times

        #
        self.shmem_tone_state = shmem_tone_state

        #
        self.initialize_tone_state()

        #
        self.initialize_rois_traces()

        #
        self.initalize_n_ttl()

        # #
        self.make_roi_plots()

        #
        self.initalize_reward_times()

        #
        self.initialize_termination_flag()

        #
        self.ctr = 0
        
        #
        self.setup_complete = True

        # enter plot update condition
        # optional use sleep to slow down plotting
        while True:
            self.update_plots()

            if self.termination_flag:
                print ("... EXITING PLOTTING CLASS ...")
                break
            # optional decrease plotting speed, may help in some cases
            # time.sleep()

        quit()
    #
    def initialize_termination_flag(self):

        #
        aa = np.zeros(1, dtype=np.int64)

        # get the rois_traces from the shared memory name
        self.existing_shm_termination_flag = shared_memory.SharedMemory(name=self.shmem_termination_flag)

        #
        self.termination_flag = np.ndarray(aa.shape,
                                           dtype=aa.dtype,
                                           buffer=self.existing_shm_termination_flag.buf)

    #
    def initalize_reward_times(self):

        #
        print ("  n_rewards memory name : ", self.shmem_reward_times)

        aa = np.zeros((2,1000), dtype=np.int64)

        # get the rois_traces from the shared memory name
        self.existing_shm_reward_times = shared_memory.SharedMemory(name=self.shmem_reward_times)
        print ("existing shm: ", self.existing_shm_reward_times)

        #
        self.reward_times = np.ndarray(aa.shape,
                                    dtype=aa.dtype,
                                    buffer=self.existing_shm_reward_times.buf)

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
        data = np.load(self.fname_rois)
        self.cell_f0s = data['cell_f0s']


    #
    def make_roi_plots(self):

        #
        self.plot_y_scale = 2500

        #
        self.fig = plt.figure(figsize=(10,5))

        self.ax = self.fig.add_subplot(111)
        #self.ax.set_ylim(0, self.plot_y_scale*len(self.rois_traces)*1.1)
        self.ax.set_ylim(0, self.plot_y_scale*4.5)
        self.ax.set_xlim(-self.plotting_window_width,0)
        self.ax.set_xlabel("Time (sec)")
        #self.ax.set_title("T=0: "+str(round(self.n_ttl2[0]/self.sampleRate_2P,2)))
        #self.ax.set_yticks([])

        self.plot_times = np.arange(-self.plotting_window_width*self.sampleRate_2P,0,1)/self.sampleRate_2P

        # make a list to hold the matplotlib line objects
        self.time_course_objects = []
        self.f0_objects = []
        for k in range(self.rois_traces.shape[0]):

            #
            #print (k, self.plot_times.shape, self.rois_traces[k,:10*self.sampleRate_2P].shape)
            y_values = self.rois_traces[k,:self.plotting_window_width*self.sampleRate_2P]+self.plot_y_scale*k

            #
            lineobject, = self.ax.plot(self.plot_times,
                                       #self.rois_traces[k,-self.plot_times:]-1000*k,  # plot last X values depending on length of plttimes
                                       y_values,  # plot last X values depending on length of plttimes
                                       #'r-'
                                       )  # Returns a tuple of line objects, thus the comma
            #
            self.time_course_objects.append(lineobject)

            #
            # also draw the f0 baseline of each cell!!
            f0object, = self.ax.plot([self.plot_times[0],self.plot_times[-1]],
                                       # self.rois_traces[k,-self.plot_times:]-1000*k,  # plot last X values depending on length of plttimes
                                       [0+self.plot_y_scale*k,0+self.plot_y_scale*k],  # plot last X values depending on length of plttimes
                                       'g--'
                                       )  # Return
            self.f0_objects.append(f0object)



        # self.ax.clear()

        #
        #self.ax.set_title("(if closed, can't reopen")

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

        # make t=0 tick to the current timer in seconds
        x_ticks = np.arange(-30,0.1,5)
        x_ticks_new = np.arange(-30,0.1,5)
        x_ticks_new[-1] = round(self.n_ttl2[0]/self.sampleRate_2P,2)
        self.ax.set_xticks(x_ticks, x_ticks_new)

        #
        self.n_ttl_last = self.n_ttl_current.copy()

        #
        x_values = np.arange(0, min(self.n_ttl_current, self.plotting_window_width*self.sampleRate_2P), 1) / 30 - self.plotting_window_width
        x_values1 = np.array([x_values[0], x_values[-1]])

        #
        for k in range(self.rois_traces.shape[0]):

            #
            y_values = self.rois_traces[k,
                                    max(0,self.n_ttl_current-self.plotting_window_width*self.sampleRate_2P):self.n_ttl_current]+self.plot_y_scale*k

            #
            self.time_course_objects[k].set_data(x_values,
                                                y_values
                                                )

            # redraw baseline for each cell
            # TODO: note that we want to reset to 0 not the actual f0 values
            #y_values1 = np.array([self.cell_f0s[k]+self.plot_y_scale*k,self.cell_f0s[k]+self.plot_y_scale*k])
            y_values1 = np.array([0+self.plot_y_scale*k,0+self.plot_y_scale*k])
            #print ("xavlues: ", x_values, y_values)
            self.f0_objects[k].set_data(x_values1,
                                        y_values1
                                       )

        #
        idx1 = np.where(self.reward_times[0]>-1)[0]
        idx2 = np.where(self.reward_times[1]>-1)[0]
        self.ax.set_title(" # rewards : "+str(idx1.shape[0])+
                          " "+str(idx2.shape[0]) + "\n Freq: " +str(int(self.tone_state[0]))+"hz")

        # Try to visualize rewarded state/time
        # TODO: this is a bit tricky and cumbersome as fast plotting requires
        #   declaraing these lines prior to starting (i.e. before there were any plots)
        # # show last rewarded time
        # if idx1.shape[0]>0:
        #
        #     # grab last rewarded time for high state reward
        #     high_state_time = self.reward_times[0,idx1[-1]]
        #     if high_state_time>x_values[0] and high_state_time<=x_values[-1]
        #         plt.plot([])

        #
        self.fig.canvas.restore_region(self.axbackground)

        # fill in the axes rectangle
        self.fig.canvas.blit(self.ax.bbox)

        # add the drawn lines to the plot
        for k in range(len(self.rois_traces)):
            self.ax.draw_artist(self.time_course_objects[k])
            self.ax.draw_artist(self.f0_objects[k])

        #
        self.fig.canvas.blit(self.ax.bbox)

        #
        self.fig.canvas.flush_events()

        #
        if self.verbose2:
            print ("time for updating graph: ", time.time()-start)
            print ('')
            print ('')

