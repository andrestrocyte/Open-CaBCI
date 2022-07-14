'''
  
  C. Mitelut; github: "catubc"; mitelutco@gmail.com

'''

import time
import numpy as np
from multiprocessing import shared_memory
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider, Button, RadioButtons

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
                 fname_rois_pixels_and_thresholds,
                 shmem_rois_traces,
                 shmem_n_ttl,
                 rois_traces_shape,
                 shmem_reward_times,
                 shmem_tone_state,
                 shmem_live_frame,
                 shmem_ensemble_state,
                 bmi_high_threshold,
                 shmem_termination_flag,
                 shmem_live_video_frame,
                 video_width,
                 video_length,
                 ):

        # video image
        self.video_width = video_width
        self.video_length = video_length
        
        #
        self.video_show_downscale_factor = 10

        #
        self.shmem_live_video_frame = shmem_live_video_frame
        
        #
        self.fname_rois_pixels_and_thresholds = fname_rois_pixels_and_thresholds

        #
        self.sampleRate_2P = 30
        print ("...assuming sampling rate is ", self.sampleRate_2P, "hz")

        #
        self.plotting_window_width = 30

        # No. of frames to use for the live image averaging
        self.live_image_average_n_frames = 5

        # How many frames must go by before updating
        # TODO: not clear why we need 2 of these variables
        self.live_image_update_n_frames = 5

        #
        self.live_image_vmin = 700
        self.live_image_vmax = 1200

        #
        self.show_contours_on_image = False

        #
        self.bmi_high_threshold = bmi_high_threshold

        #
        self.live_image_counter = 0

        #
        self.verbose2 = False

        #
        self.shmem_termination_flag = shmem_termination_flag

        #
        self.shmem_live_frame = shmem_live_frame

        #
        self.shmem_ensemble_state = shmem_ensemble_state

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
        self.initialize_rois_traces()

        #
        self.initialize_ROIs_contours()

        #
        self.initialize_live_frame_shared_memory()

        #
        self.initialize_ensemble_state()
        
        #
        self.initialize_ensemble_state_array()
        
        #
        self.initialize_tone_state()

        #
        self.initalize_n_ttl()

        #
        self.initalize_reward_times()

        #
        self.initialize_termination_flag()

        #
        self.initialize_camera_frame_shared_memory()

        #
        self.initialize_live_image_array()

        #
        self.initialize_plots()

        #
        self.ctr = 0

        # enter plot update condition
        # optional use sleep to slow down plotting
        while True:
            self.update_plots()

            if self.termination_flag:
                print ("... EXITING PLOTTING CLASS ...")
                break

            # optional decrease plotting speed, may help in some cases
            # time.sleep(self.plotter_sleep_time_between_updates)

    #
    def initialize_camera_frame_shared_memory(self):

        ''' shared variable that keeps current image in memeory for plotter to visualize

        '''

        # make a numpy array to hold the rois_traces
        aa = np.zeros((1,self.video_width,self.video_length), dtype=np.uint8)

        # get the rois_traces from the shared memory name
        self.existing_shm_video_frame = shared_memory.SharedMemory(name=self.shmem_live_video_frame)

        #
        self.live_video_frame = np.ndarray(aa.shape,
                                           dtype=aa.dtype,
                                           buffer=self.existing_shm_video_frame.buf
                                           )

        #self.video_frame[:] = aa[:]

    #
    def initialize_ensemble_state(self):
        
        #
        #print("  ensemble state memory name : ", self.shmem_ensemble_state)

        aa = np.zeros((1,), dtype=np.float32)

        # get the rois_traces from the shared memory name
        self.existing_shm_ensemble_state = shared_memory.SharedMemory(name=self.shmem_ensemble_state)

        #
        self.ensemble_state = np.ndarray(aa.shape,
                                 dtype=aa.dtype,
                                 buffer=self.existing_shm_ensemble_state.buf)

        #
        #self.ensemble_state_last = self.ensemble_state[0].copy()

    #
    def initialize_ROIs_contours(self):

        #####################################################
        # load individual cell ROIs as saved by the calibration step
        # TODO: generalize some of this code to allow different #s of cells; - not a priority
        data = np.load(self.fname_rois_pixels_and_thresholds,
                       allow_pickle=True)
        self.rois_contours = []
        self.rois_contours.append(data['cell0_contour'])
        self.rois_contours.append(data['cell1_contour'])
        self.rois_contours.append(data['cell2_contour'])
        self.rois_contours.append(data['cell3_contour'])

    #
    def initialize_live_image_array(self):

        # save the images as they come in so we can just plot averages not individual frames
        self.live_image_array = np.zeros((30,512,512), dtype=np.uint16)

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
        #print ("  n_rewards memory name : ", self.shmem_reward_times)

        aa = np.zeros((2,1000), dtype=np.int64)

        # get the rois_traces from the shared memory name
        self.existing_shm_reward_times = shared_memory.SharedMemory(name=self.shmem_reward_times)
        #print ("existing shm: ", self.existing_shm_reward_times)

        #
        self.reward_times = np.ndarray(aa.shape,
                                    dtype=aa.dtype,
                                    buffer=self.existing_shm_reward_times.buf)

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
    def initialize_live_frame_shared_memory(self):

        ''' shared variable that keeps current image in memeory for plotter to visualize

        '''

        # make a numpy array to hold the rois_traces
        aa = np.zeros((1,512,512), dtype=np.uint16)

        # get the rois_traces from the shared memory name
        self.existing_shm_live_frame = shared_memory.SharedMemory(name=self.shmem_live_frame)

        #
        self.live_frame = np.ndarray(aa.shape,
                                 dtype=aa.dtype,
                                 buffer=self.existing_shm_live_frame.buf)

    #
    def initalize_n_ttl(self):

        #
        #print ("  nttl memory name : ", self.shmem_n_ttl)

        aa = np.zeros((1,), dtype=np.int64)

        # get the rois_traces from the shared memory name
        self.existing_shm_n_ttl = shared_memory.SharedMemory(name=self.shmem_n_ttl)
        #existing_shm = shared_memory.SharedMemory(name='testname')

        #print ("existing shm: ", self.existing_shm_n_ttl)


        #
        self.n_ttl2 = np.ndarray(aa.shape,
                                 dtype=aa.dtype,
                                 buffer=self.existing_shm_n_ttl.buf)

        #
        #print ("  loaded n_ttl: ", self.n_ttl2)

        #
        self.n_ttl_last = self.n_ttl2[0].copy()

       #
    #
    def initialize_ensemble_state_array(self):
        
        #
        self.ensemble_state_array = np.zeros(self.rois_traces.shape[1]+100)  # add a small bufffer in case things don't shut down immediately
       
        
    #
    def initialize_rois_traces(self):

        #print ("  Plotter loaded: ", self.shmem_rois_traces,
        #       " size: ", self.rois_traces_shape)

        # get the rois_traces from the shared memory name
        self.existing_shm_rois_traces = shared_memory.SharedMemory(name=self.shmem_rois_traces)

        #
        # print ("existing shm traces*****************: ",
        #        self.existing_shm_rois_traces)

        #
        self.rois_traces = np.ndarray((self.rois_traces_shape[0],
                                       self.rois_traces_shape[1]),
                                       dtype=np.float32,
                                       buffer=self.existing_shm_rois_traces.buf)
        #
        data = np.load(self.fname_rois_pixels_and_thresholds)
        self.cell_f0s = data['cell_f0s']


    #
    def initialize_plots(self):

        #
        self.plot_y_scale = 1

        #
        self.fig = plt.figure(figsize=(8,8))

        self.grid = GridSpec(11, 10)#, left=0.55, right=0.98, hspace=0.05)

        #########################################################
        ######################### PLOT CA IMAGE #################
        #########################################################
        # TODO: refactor this plot to another function
        self.ax_image = self.fig.add_subplot(self.grid[:5, 5:])

        #
        self.image_obj = self.ax_image.imshow(self.live_frame[0],
                                              vmin=self.live_image_vmin,
                                              vmax=self.live_image_vmax)

        #
        #axcolor = 'lightgoldenrodyellow'
        axmin = self.fig.add_axes([0.55, 0.90, 0.1, 0.03])
        axmax  = self.fig.add_axes([0.55, 0.93, 0.1, 0.03])
        n_frame_ave  = self.fig.add_axes([0.83, 0.93, 0.10, 0.03])
        
        self.smin = Slider(axmin, 'Min', 0, 2048, valinit=self.live_image_vmin)
        self.smax = Slider(axmax, 'Max', 0, 2048, valinit=self.live_image_vmax)
        self.n_frame_ave = Slider(n_frame_ave, 'nFrames', 1, 30, valinit=self.live_image_average_n_frames)

        #
        def update_clim(val):
            self.image_obj.set_clim([self.smin.val,
                                     self.smax.val])
        def update_n_frames_average(val):
            self.live_image_average_n_frames = int(self.n_frame_ave.val)
            
        #S
        self.smin.on_changed(update_clim)
        self.smax.on_changed(update_clim)
        self.n_frame_ave.on_changed(update_n_frames_average)

		
        #
        self.ax_image.set_xlim(0,512)
        self.ax_image.set_ylim(512,0)

        # add contours on top of cells
        for c in range(len(self.rois_contours)):
            for k in range(len(self.rois_contours[c])-1):
                self.ax_image.plot([self.rois_contours[c][k][0], self.rois_contours[c][k+1][0]],
                                   [self.rois_contours[c][k][1], self.rois_contours[c][k + 1][1]],
                                    c='red')

        #########################################################
        ################# PLOT VIDEO IMAGE ######################
        #########################################################
        # TODO: refactor this plot to another function
        self.ax_camera = self.fig.add_subplot(self.grid[6:, :8])
        self.ax_camera.set_xticks([])
        self.ax_camera.set_yticks([])
        #
        self.camera_obj = self.ax_camera.imshow(self.live_video_frame[0].T[::self.video_show_downscale_factor,
																			::self.video_show_downscale_factor],
                                                vmin=0,
                                                vmax=255,
                                                aspect='auto',
                                                cmap='binary'
                                               #
                                              )


        #########################################################
        ##################### PLOT ROI TRACES ###################
        #########################################################
        # TODO: refactor this plot to another function
        self.ax_traces = self.fig.add_subplot(self.grid[:5, :5])

        self.ax_traces.set_ylim(0, self.plot_y_scale*4.5+ self.plot_y_scale*3)
        self.ax_traces.set_xlim(-self.plotting_window_width,0)
        self.ax_traces.set_xlabel("Time (sec)")

        self.plot_times = np.arange(-self.plotting_window_width*self.sampleRate_2P,0,1)/self.sampleRate_2P

        # make a list to hold the matplotlib line objects
        self.time_course_objects = []
        self.f0_objects = []
        
        # plot ROIS
        for k in range(self.rois_traces.shape[0]):

            #
            #print (k, self.plot_times.shape, self.rois_traces[k,:10*self.sampleRate_2P].shape)
            y_values = self.rois_traces[k,:self.plotting_window_width*self.sampleRate_2P]+self.plot_y_scale*k

            #
            #print (self.plot_times.shape, y_values.shape)
            lineobject, = self.ax_traces.plot(self.plot_times,
                                       #self.rois_traces[k,-self.plot_times:]-1000*k,  # plot last X values depending on length of plttimes
                                       y_values,  # plot last X values depending on length of plttimes
                                       #'r-'
                                       )  # Returns a tuple of line objects, thus the comma
            #
            self.time_course_objects.append(lineobject)

            #
            # also draw the f0 baseline of each cell!!
            f0object, = self.ax_traces.plot([self.plot_times[0],self.plot_times[-1]],
                                       # self.rois_traces[k,-self.plot_times:]-1000*k,  # plot last X values depending on length of plttimes
                                       [0+self.plot_y_scale*k,0+self.plot_y_scale*k],  # plot last X values depending on length of plttimes
                                       'g--'
                                       )  # Return
            self.f0_objects.append(f0object)


        # plot sum ensemble state
        y_values = self.ensemble_state_array[:self.plotting_window_width*self.sampleRate_2P]+self.plot_y_scale*(k+1.5)

        #
        #print (self.plot_times.shape, y_values.shape)
        lineobject, = self.ax_traces.plot(self.plot_times,
                                   #self.rois_traces[k,-self.plot_times:]-1000*k,  # plot last X values depending on length of plttimes
                                   y_values,  # plot last X values depending on length of plttimes
                                   linewidth = 2,
                                   #'r-'
                                   )  # Returns a tuple of line objects, thus the comma
        #
        self.time_course_objects.append(lineobject)

        # ADD F0 for ensemble states
        f0object, = self.ax_traces.plot([self.plot_times[0],self.plot_times[-1]],
                                   [0+self.plot_y_scale*(k+1.5),
                                    0+self.plot_y_scale*(k+1.5)],  # plot last X values depending on length of plttimes
                                   'b--',
                                   linewidth=2,
                                   )  # Return
                                   
        self.f0_objects.append(f0object)


        # ADD Reward level for ensemble states
        f0object, = self.ax_traces.plot([self.plot_times[0],self.plot_times[-1]],
                                   [self.plot_y_scale*(k+1.5)+self.bmi_high_threshold,
                                    self.plot_y_scale*(k+1.5)+self.bmi_high_threshold],  # plot last X values depending on length of plttimes
                                   'r--',
                                   linewidth=2,
                                   )  # Return
                                   
        self.f0_objects.append(f0object)


        #
        self.fig.canvas.flush_events()

        #
        self.fig.canvas.draw()

        #
        self.fig.canvas.flush_events()

        #
        #self.fig.clf()

        # cache the background
        self.axbackground = []
        self.axbackground.append(self.fig.canvas.copy_from_bbox(self.ax_traces.bbox))
        self.axbackground.append(self.fig.canvas.copy_from_bbox(self.ax_image.bbox))
        self.axbackground.append(self.fig.canvas.copy_from_bbox(self.ax_camera.bbox))


        #
        plt.show(block=False)

    #
    def update_plots(self):

        '''  Function to dynamically update our plots
            TODO: move it to a separate core/process to not interfere with the main BMI loop/analysis code

        '''

        # update ROI line plots
        self.n_ttl_current = self.n_ttl2[0].copy()
        
        # ensure we have moved over 1 ttl pulse first
        if self.n_ttl_current < (self.n_ttl_last+1):
            return

        # save the current ensembel state
        self.ensemble_state_array[self.n_ttl_current] = self.ensemble_state[0]

        #
        if self.verbose2:
            start = time.time()

        # restore background for all plots
        for k in range(len(self.axbackground)):
            self.fig.canvas.restore_region(self.axbackground[k])


        ####################################
        ########### UPDATE IMAGE ###########
        ####################################
        # TODO: note this takes 10ms of time to update,
        #     may wish to plot it only 1 - 5 x per second, no  real information here aside from nasty drift
        #start = time.time()

        # shift all the data one image over
        self.live_image_array[:-1] = self.live_image_array[1:]

        # add current image frame
        self.live_image_array[-1] = self.live_frame[0].copy()

        # dont' update thelive image frame until we have at least min # of frams
        if self.live_image_counter> self.live_image_average_n_frames:

            # reset counter
            #self.live_image_counter = 0

            # compute mean over last n_frames
            temp = np.mean(self.live_image_array[-self.live_image_average_n_frames:],axis=0)
            
            # if False:
            #     idx = np.where(temp<self.live_image_vmin)
            #     temp[idx] = self.live_image_vmin
            #     idx = np.where(temp>self.live_image_vmax)
            #     temp[idx] = self.live_image_vmax
                    
            #
            self.image_obj.set_data(temp)

        #else:
        self.live_image_counter+=1


        ####################################
        ######## UPDATE CAMERA OUTPUT ######
        ####################################
        self.camera_obj.set_data(self.live_video_frame[0].T[::self.video_show_downscale_factor,
															::self.video_show_downscale_factor])

        #print ("time to compute image update: ", time.time()-start)
        #########################################
        ############ UPDATE LINE PLOTS ##########
        #########################################
        # make t=0 tick to the current timer in seconds
        x_ticks = np.arange(-30,0.1,5)
        x_ticks_new = np.arange(-30,0.1,5)
        x_ticks_new[-1] = round(self.n_ttl2[0]/self.sampleRate_2P,2)
        self.ax_traces.set_xticks(x_ticks, x_ticks_new)

        #
        self.n_ttl_last = self.n_ttl_current.copy()

        #
        x_values = np.arange(0, min(self.n_ttl_current, self.plotting_window_width*self.sampleRate_2P), 1) / 30 - self.plotting_window_width
        x_values1 = np.array([x_values[0], x_values[-1]])

        # plot roi time courses
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
            y_values1 = np.array([self.plot_y_scale*k,self.plot_y_scale*k])

            #
            self.f0_objects[k].set_data(x_values1,
                                        y_values1
                                       )
        ################################################
        # update ensemble state
        y_values = self.ensemble_state_array[max(0,self.n_ttl_current-self.plotting_window_width*self.sampleRate_2P):
                                                self.n_ttl_current]+self.plot_y_scale*(k+1.5)

        self.time_course_objects[k+1].set_data(x_values,
                                               y_values
                                                )                   
        # update F0
        y_values1 = np.array([self.plot_y_scale*(k+1.5),self.plot_y_scale*(k*1.5)])
        self.f0_objects[k+1].set_data(x_values1,
                                    y_values1
                                   )    

        # update ensembel threshold
        y_values1 = np.array([self.plot_y_scale*(k+1.5)+self.bmi_high_threshold,
                              self.plot_y_scale*(k*1.5)+self.bmi_high_threshold])
        self.f0_objects[k+2].set_data(x_values1,
                                    y_values1
                                   )    
        
        ##################################################
        idx1 = np.where(self.reward_times[0]>-1)[0]
        idx2 = np.where(self.reward_times[1]>-1)[0]
        self.ax_traces.set_title(" # rewards : "+str(idx1.shape[0])+" "+str(idx2.shape[0]) + 
                                 ": Freq: " +str(int(self.tone_state[0]))+"hz"+
                                 "\n Ensemble state: "+str(round(self.ensemble_state[0],2)), fontsize=12)

        #
        for k in range(len(self.axbackground)):
            self.fig.canvas.restore_region(self.axbackground[k])

        # fill in the axes rectangle
        self.fig.canvas.blit(self.ax_traces.bbox)

        # add the drawn lines to the plot
        for k in range(len(self.rois_traces)):
            self.ax_traces.draw_artist(self.time_course_objects[k])
            self.ax_traces.draw_artist(self.f0_objects[k])

        #
        self.fig.canvas.blit(self.ax_traces.bbox)
        #self.fig.canvas.blit(self.ax_image.bbox)

        #
        self.fig.canvas.flush_events()

        #
        if self.verbose2:
            print ("time for updating graph: ", time.time()-start)
            print ('')
            print ('')

