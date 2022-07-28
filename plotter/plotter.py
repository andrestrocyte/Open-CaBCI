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
                 calibration_flag,
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
                 shmem_motion_correction_flag,
                 motion_flag,
                 shmem_dynamic_f0_flag,
                 shmem_manual_motion_correction_array,
                 ):

        #
        self.shmem_motion_correction_flag = shmem_motion_correction_flag

        #
        self.shmem_manual_motion_correction_array = shmem_manual_motion_correction_array

        #
        self.shmem_dynamic_f0_flag = shmem_dynamic_f0_flag

        #
        self.motion_flag = motion_flag

        #
        self.calibration_flag = calibration_flag

        # video image
        self.video_width = video_width
        self.video_length = video_length
        
        #
        self.video_show_downscale_factor = 5

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
        self.live_image_vmin = 1
        self.live_image_vmax = 2500

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
        if self.calibration_flag==False:

            #
            self.initialize_rois_traces()

            #
            self.initialize_ROIs_contours()

            #
            self.initialize_ensemble_state()

            #
            self.initialize_ensemble_state_array()

            #
            self.initialize_tone_state()

            #
            self.initialize_live_image_array()

            #
            self.initialize_motion_correction_flag()

        #
        self.initialize_live_frame_shared_memory()

        #
        self.initalize_n_ttl()

        #
        self.initalize_reward_times()

        #
        self.initialize_termination_flag()

        #
        self.initialize_camera_frame_shared_memory()

        #
        self.initialize_dynamic_f0_flag()

        #
        self.initialize_manual_motion_correction_array()

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
    def initialize_manual_motion_correction_array(self):

        #
        aa = np.zeros((2), dtype=np.int32)

        # get the rois_traces from the shared memory name
        self.existing_shmem_manual_motion_correction_array = shared_memory.SharedMemory(name=self.shmem_manual_motion_correction_array)

        #
        self.manual_motion_correction_array = np.ndarray(aa.shape,
                                     dtype=aa.dtype,
                                     buffer=self.existing_shmem_manual_motion_correction_array.buf)



    #
    def initialize_ensemble_state(self):

        #
        aa = np.zeros((1,), dtype=np.float32)

        # get the rois_traces from the shared memory name
        self.existing_shm_ensemble_state = shared_memory.SharedMemory(name=self.shmem_ensemble_state)

        #
        self.ensemble_state = np.ndarray(aa.shape,
                                 dtype=aa.dtype,
                                 buffer=self.existing_shm_ensemble_state.buf)

        #
        self.ensemble_state_counter = 0

    #
    def initialize_motion_correction_flag(self):

        #
        aa = np.zeros((1,), dtype=np.float32)

        #
        self.existing_motion_correction_flag = shared_memory.SharedMemory(name=self.shmem_motion_correction_flag)

        #
        self.motion_corection_flag = np.ndarray(aa.shape,
                                         dtype=aa.dtype,
                                         buffer=self.existing_motion_correction_flag.buf)

        #
        self.motion_corection_flag[0] = self.motion_flag

    #
    def initialize_dynamic_f0_flag(self):

        #
        aa = np.zeros((1,), dtype=np.float32)



        #
        self.existing_dynamic_f0_flag = shared_memory.SharedMemory(name=self.shmem_dynamic_f0_flag)

        #
        self.dynamic_f0_flag = np.ndarray(aa.shape,
                                         dtype=aa.dtype,
                                         buffer=self.existing_dynamic_f0_flag.buf)

        #
        self.dynamic_f0_flag[0] = 0

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
        self.contours_all_cells = data['contours_all_cells']
        print ("LOADED ALL CELL CONTOURS: ", len(self.contours_all_cells),
               "example cell contour shape: ", self.contours_all_cells[0].shape)


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
        #self.ensemble_state_array = np.zeros(self.rois_traces.shape[1]+100)  # add a small bufffer in case things don't shut down immediately
        self.ensemble_state_array = np.zeros(self.rois_traces.shape[1]+100) + np.nan # add a small bufffer in case things don't shut down immediately

    #
    def initialize_rois_traces(self):

        # get the rois_traces from the shared memory name
        self.existing_shm_rois_traces = shared_memory.SharedMemory(name=self.shmem_rois_traces)

        #
        self.rois_traces = np.ndarray((self.rois_traces_shape[0],
                                       self.rois_traces_shape[1]),
                                       dtype=np.float32,
                                       buffer=self.existing_shm_rois_traces.buf)

        # also load the F0 computed from the calibration session
        data = np.load(self.fname_rois_pixels_and_thresholds)
        self.cell_f0s = data['cell_f0s']


    ################################################################################################
    ##################################### INITIALIZE PLOTS #########################################
    ################################################################################################
    def initialize_plots(self):

        #
        self.plot_y_scale = 1

        #
        clrs=['blue','magenta','red','orange']

        #
        #if self.calibration_flag==True:
        self.fig = plt.figure(figsize=(8,5))
        #else:
        #    self.fig = plt.figure(figsize=(8,5))

        #
        self.grid = GridSpec(8, 12)#, left=0.55, right=0.98, hspace=0.05)

        #########################################################
        ######################### PLOT CA IMAGE #################
        #########################################################
        if self.calibration_flag==False:
            # TODO: refactor this plot to another function
            self.ax_image = self.fig.add_subplot(self.grid[:, 4:])

            #
            self.image_obj = self.ax_image.imshow(self.live_frame[0],
                                                  vmin=self.live_image_vmin,
                                                  vmax=self.live_image_vmax,
                                                  interpolation="none")
            #
            self.ax_image.set_xlim(0,512)
            self.ax_image.set_ylim(512,0)

            # PLOT ROI CONTOURS
            # add contours on top of cells
            for c in range(len(self.rois_contours)):
                for k in range(len(self.rois_contours[c])-1):
                    self.ax_image.plot([self.rois_contours[c][k][0], self.rois_contours[c][k+1][0]],
                                       [self.rois_contours[c][k][1], self.rois_contours[c][k + 1][1]],
                                        c=clrs[c],
                                       linewidth=3)

            # add random cells to the data
            ids = np.arange(0,max(80,len(self.contours_all_cells)),1)
            for c in ids:
                # plot each cell contour
                for k in range(len(self.contours_all_cells[c])-1):
                    self.ax_image.plot([self.contours_all_cells[c][k][0], self.contours_all_cells[c][k+1][0]],
                                       [self.contours_all_cells[c][k][1], self.contours_all_cells[c][k + 1][1]],
                                        c='white',
                                       linewidth=1)

            #################################################
            ############ ADD IMAGINING VIS BUTTONS ##########
            #################################################
            # add all the buttons/for visualization
            axmin = self.fig.add_axes([0.55, 0.94, 0.1, 0.03])
            axmax  = self.fig.add_axes([0.55, 0.96, 0.1, 0.03])
            n_frame_ave  = self.fig.add_axes([0.55, 0.92, 0.10, 0.03])

            # settings for image processing
            motion_correction_slider = self.fig.add_axes([0.83, 0.94, 0.10, 0.03])
            dynamic_f0_correction_slider = self.fig.add_axes([0.83, 0.96, 0.10, 0.03])

            #
            self.smin = Slider(axmin, 'Min', 0, self.live_image_vmax, valinit=self.live_image_vmin)
            self.smax = Slider(axmax, 'Max', 0, self.live_image_vmax, valinit=self.live_image_vmax)

            #
            vals_n_frames = np.arange(1,31,1)
            self.n_frame_ave = Slider(n_frame_ave, '# Frames', 1, 30, valinit=self.live_image_average_n_frames,
                                      valstep = vals_n_frames)

            #
            vals_motion = [0,1]
            self.motion_slider = Slider(motion_correction_slider, 'Motion correct', 0, 1, valinit=self.motion_corection_flag[0],
                                      valstep=vals_motion)

            #
            vals_dynamic_f0 = [0,1]
            self.dynamic_f0_slider = Slider(dynamic_f0_correction_slider, 'Dynamic F0', 0, 1,
                                            valinit=self.dynamic_f0_flag[0],
                                            valstep=vals_dynamic_f0)
            #
            def update_clim(val):
                self.image_obj.set_clim([self.smin.val,
                                         self.smax.val])

            #
            def update_n_frames_average(val):
                self.live_image_average_n_frames = int(self.n_frame_ave.val)

            #
            def update_motion_flag(val):
                self.motion_corection_flag[0] = int(self.motion_slider.val)

            #
            def update_dynamic_flag(val):
                self.dynamic_f0_flag[0] = int(self.dynamic_f0_slider.val)

            #
            self.smin.on_changed(update_clim)
            self.smax.on_changed(update_clim)
            self.n_frame_ave.on_changed(update_n_frames_average)
            self.motion_slider.on_changed(update_motion_flag)
            self.dynamic_f0_slider.on_changed(update_dynamic_flag)

        #########################################################
        ################# PLOT VIDEO IMAGE ######################
        #########################################################
        # TODO: refactor this plot to another function
        if self.calibration_flag==False:
            self.ax_camera = self.fig.add_subplot(self.grid[5:, :4])
        else:
            self.ax_camera = self.fig.add_subplot(self.grid[:, :])

        self.ax_camera.set_xticks([])
        self.ax_camera.set_yticks([])
        #
        self.camera_obj = self.ax_camera.imshow(self.live_video_frame[0].T[::self.video_show_downscale_factor,
																			::self.video_show_downscale_factor],
                                                vmin=0,
                                                vmax=255,
                                                aspect='auto',
                                                cmap='binary_r',
                                                interpolation="none"
                                               #
                                              )

        #########################################################
        ########## INITIALIZE DRIFT BUTTONS ######################
        #########################################################
        #
        if self.calibration_flag == False:
            #self.ax_camera = self.fig.add_subplot(self.grid[6:, :8])

            #
            axleft = plt.axes([0.9, 0.2, 0.04, 0.04])
            axright = plt.axes([.95, 0.2, 0.04, 0.04])
            axup = plt.axes([0.925, 0.25, 0.04, 0.04])
            axdown = plt.axes([0.925, 0.15, 0.04, 0.04])


            def shift_fov_right(event):
                self.manual_motion_correction_array[0]+=1
                print ("moving FOV right")

            def shift_fov_left(event):
                self.manual_motion_correction_array[0]-=1
                print ("moving FOV left")

            def shift_fov_up(event):
                self.manual_motion_correction_array[1]+=1
                print ("moving FOV up")

            def shift_fov_down(event):
                self.manual_motion_correction_array[1]-=1
                print ("moving FOV down")

            #
            bleft = Button(axleft, 'LEFT')
            bleft.on_clicked(shift_fov_left)
            bright = Button(axright, 'RIGHT')
            bright.on_clicked(shift_fov_right)
            bup = Button(axup, 'UP')
            bup.on_clicked(shift_fov_up)
            bdown = Button(axdown, 'DOWN')
            bdown.on_clicked(shift_fov_down)

        #########################################################
        ##################### PLOT ROI TRACES ###################
        #########################################################
        if self.calibration_flag==False:
            # TODO: refactor this plot to another function
            self.ax_traces = self.fig.add_subplot(self.grid[:4, :4])

            #self.ax_traces.set_ylim(0, self.plot_y_scale*4.5 + self.plot_y_scale*3)
            self.ax_traces.set_ylim(-0.25*self.plot_y_scale, self.plot_y_scale*(4+2.5)+2*self.bmi_high_threshold)
            self.ax_traces.set_xlim(-self.plotting_window_width,0)
            self.ax_traces.set_xlabel("Time (sec)")

            self.plot_times = np.arange(-self.plotting_window_width*self.sampleRate_2P,0,1)/self.sampleRate_2P

            # make a list to hold the matplotlib line objects
            self.time_course_objects = []
            self.f0_objects = []

            # plot ROIS
            for k in range(self.rois_traces.shape[0]):

                #
                y_values = self.rois_traces[k,:self.plotting_window_width*self.sampleRate_2P]+self.plot_y_scale*k

                #
                lineobject, = self.ax_traces.plot(self.plot_times,
                                           #self.rois_traces[k,-self.plot_times:]-1000*k,  # plot last X values depending on length of plttimes
                                           y_values,  # plot last X values depending on length of plttimes
                                           c=clrs[k]
                                              #'r-'
                                           )  # Returns a tuple of line objects, thus the comma
                #
                self.time_course_objects.append(lineobject)

                # also draw the f0 baseline of each cell!!
                f0object, = self.ax_traces.plot([self.plot_times[0],self.plot_times[-1]],
                                           # self.rois_traces[k,-self.plot_times:]-1000*k,  # plot last X values depending on length of plttimes
                                           [0+self.plot_y_scale*k,0+self.plot_y_scale*k],  # plot last X values depending on length of plttimes
                                           '--',
                                           c=clrs[k]
                                           )  # Return
                self.f0_objects.append(f0object)


            # plot sum ensemble state
            y_values = self.ensemble_state_array[:self.plotting_window_width*self.sampleRate_2P]+self.plot_y_scale*(k+2.5)

            #
            lineobject, = self.ax_traces.plot(self.plot_times,
                                       y_values,  # plot last X values depending on length of plttimes
                                       linewidth = 4,
                                       c='black'
                                       )  # Returns a tuple of line objects, thus the comma
            #
            self.time_course_objects.append(lineobject)

            # ADD F0 for ensemble states
            f0object, = self.ax_traces.plot([self.plot_times[0],self.plot_times[-1]],
                                       [0+self.plot_y_scale*(k+2.5),
                                        0+self.plot_y_scale*(k+2.5)],  # plot last X values depending on length of plttimes
                                       '--',
                                        c='black',
                                       linewidth=2,
                                       )  # Return

            self.f0_objects.append(f0object)


            # ADD Reward level for ensemble states
            f0object, = self.ax_traces.plot([self.plot_times[0],self.plot_times[-1]],
                                       [self.plot_y_scale*(k+2.5)+self.bmi_high_threshold,
                                        self.plot_y_scale*(k+2.5)+self.bmi_high_threshold],  # plot last X values depending on length of plttimes
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
        if self.calibration_flag==False:
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

        #############################################################
        ################## UPDATE ENSEMBLE ARRAY ####################
        #############################################################
        if self.calibration_flag==False:
            # save the current ensembel state
            while self.n_ttl_current > self.ensemble_state_counter:
                self.ensemble_state_array[self.ensemble_state_counter] = self.ensemble_state[0]
                self.ensemble_state_counter+=1

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
        if self.calibration_flag==False:
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

                #
                self.image_obj.set_data(temp)

            #else:
            self.live_image_counter+=1


        ####################################
        ######## UPDATE CAMERA OUTPUT ######
        ####################################
        self.camera_obj.set_data(self.live_video_frame[0].T[::self.video_show_downscale_factor,
															::self.video_show_downscale_factor])

        #########################################
        ######### UPDATE ROI TIME COURSES #######
        #########################################
        if self.calibration_flag==False:
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
        ############ UPDATE ENSEMBLE STATE #############
        ################################################
        if self.calibration_flag==False:
            # update ensemble state
            y_values = self.ensemble_state_array[max(0,self.n_ttl_current-self.plotting_window_width*self.sampleRate_2P):
                                                    self.n_ttl_current]+self.plot_y_scale*(k+1.5)

            #
            #print (" esnemble values: ", y_values)

            #
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
        
        ################################################
        ################# UPDATE TITLE #################
        ################################################
        if self.calibration_flag==False:
            idx1 = np.where(self.reward_times[0]>-1)[0]
            idx2 = np.where(self.reward_times[1]>-1)[0]
            self.ax_traces.set_title(" # rewards : "+str(idx1.shape[0])+" "+str(idx2.shape[0]) +
                                     ": Freq: " +str(int(self.tone_state[0]))+"hz"+
                                     "\n Ensemble state: "+str(round(self.ensemble_state[0],2)), fontsize=12)

        #####################################################
        ################# REFRESH PLOTS #####################
        #####################################################
        for k in range(len(self.axbackground)):
            self.fig.canvas.restore_region(self.axbackground[k])

        # add the drawn lines to the plot
        if self.calibration_flag==False:
            for k in range(len(self.rois_traces)):
                self.ax_traces.draw_artist(self.time_course_objects[k])
                self.ax_traces.draw_artist(self.f0_objects[k])

        if self.calibration_flag==False:
            # fill in the axes rectangle
            self.fig.canvas.blit(self.ax_traces.bbox)

            #
            self.fig.canvas.blit(self.ax_traces.bbox)
            # self.fig.canvas.blit(self.ax_image.bbox)

        #
        self.fig.canvas.flush_events()

        #
        if self.verbose2:
            print ("time for updating graph: ", time.time()-start)
            print ('')
            print ('')

