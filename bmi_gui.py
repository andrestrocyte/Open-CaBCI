import os
from multiprocessing import Process
import time

# 
#from water.water import WaterReward
#from tkinter import Tk #for Python 3.x
#from tkinter.filedialog import askopenfilename, askdirectory

#
from bmi.bmi import BMI
from calibration.calibration import BMICalibration
from plotter.plotter import PlotROIs
from tone.tone import PlayTone
from microphone.microphone import Microphone
from drift.drift import DriftCorrection
from gui.gui import gui
from camera.camera import Camera


########################################################################
########################################################################
########################################################################
if __name__ ==  '__main__':

    # hardcoded values
    sampleRate_2P = 30    # # frames of recording   +  buffer frames, usually 10-15 sec
    
    # values read from gui
    (fname_root_path,
     bmi_read,
     lick_read,
     tone_read,
     water_read,
     video_read,
     video_hardware_trigger,
     simulation_sleep,
     n_frames,
     video_width,
     video_height,
     calibration_read,
     motion_read,
     align_flag,
     water_vol_ttl)= gui()
    
    #print ("LOADED GUI Params: ", fname_root_path, bmi_read, lick_read, tone_read, water_read, video_read, video_hardware_trigger, simulation_sleep, n_frames)
    
    #######################################################################             
    ################### DEFAULT PARAMTERS FOR BMI #########################             
    #######################################################################             
    n_frames_session = int(float(n_frames))
    
    #
    simulation_flag_bmi = (bmi_read=="True")         # Runs the BMI class in simulation mode (i.e. don't need Bscope input)
                                        #  - set to true unless we have a real mouse in the BScope to get
                                        #    real time data from; otherwise data is read from disk at some location
                                        # TODO: in non simulation mode - have slightly different panels for 
                                        #       reading directories of the data as Bscope does not make them until 
                                        #       it starts up
    simulation_flag_licking = (lick_read=="True")        # Runs the tone class in simulation mode
    simulation_flag_tone = (tone_read=="True")        # Runs the tone class in simulation mode
    simulation_flag_water = (water_read=="True")       # Runs the water class in simulation mode
    simulation_flag_video_camera = (video_read=="True")       # Runs the water class in simulation mode
    video_hardware_trigger = (video_hardware_trigger=="True")       # Runs the water class in simulation mode
    calibration_flag = (calibration_read=="True")       # Runs the water class in simulation mode
    motion_flag = (motion_read=="True")       # Runs the motion correction algorithm

    #
    dynamic_template_flag= 0

    # parameter used for simulation mode to add a delay instead of waiting for ttl pulse
    sleep_time_sec = float(simulation_sleep)



    ##########################################################################
    #################### LOAD FILE/DIRECTORY LOCATIONS ####################### 
    ##########################################################################
    # TODO: build a better gui that also takes as input the # of frames
    
    # load calibration [ca] data
    if calibration_flag:
        fname_fluorescence = os.path.join(fname_root_path,
                                      'calibration',                   # this is the root directory of the .raw file saved by bscope
                                      'Image_001_001.raw')
    
    elif align_flag:
        fname_fluorescence = os.path.join(fname_root_path,
                                      'alignment',                   # this is the root directory of the .raw file saved by bscope
                                      'Image_001_001.raw')
    
    else:
        fname_fluorescence = os.path.join(fname_root_path,
                                      'data',                   # this is the root directory of the .raw file saved by bscope
                                      'Image_001_001.raw')

    # required for bmi simulation mode as there are no ttl -pulses being read
    fname_ttl = os.path.join('data_samples',
                             "ttl_pulses.npy")
                             
    # set location of rois files
    if align_flag or calibration_flag:
        fname_rois_pixels_and_thresholds = os.path.join(os.path.split(fname_root_path)[0],
                                                        'day0',
                                                        'rois_pixels_and_thresholds_day0.npz')
    else:
        fname_rois_pixels_and_thresholds = os.path.join(fname_root_path,
                                                    'rois_pixels_and_thresholds.npz')

    ###############################################################
    #################### INITIALIZE BMI MAIN ######################
    ###############################################################
    # compute the maximum number of seconds the session will run before existing in case there are no more TTL Pulses
    max_n_seconds_session = int(n_frames_session/sampleRate_2P) + 30  # gives bit of extra time... TODO: Not required any longer?!

    #
    #if calibration_flag:
    #    bmi = BMICalibration(simulation_flag_bmi,
    #                          simulation_flag_licking,
    #                          fname_root_path,
    #                          fname_fluorescence,
    #                          fname_ttl,
    #                          sampleRate_2P,
    #                          fname_rois_pixels_and_thresholds,
    #                          max_n_seconds_session,
    #                          n_frames_session,
    #                          video_width,
    #                          video_height,
    #                          motion_flag,
    #                          align_flag
    #                          )
    #else:
    bmi = BMI(simulation_flag_bmi,
              simulation_flag_licking,
              fname_root_path,
              fname_fluorescence,
              fname_ttl,
              sampleRate_2P,
              fname_rois_pixels_and_thresholds,
              max_n_seconds_session,
              n_frames_session,
              video_width,
              video_height,
              motion_flag,
              align_flag
              )
              
    #
    bmi.water_vol_ttl = water_vol_ttl
    print (" Water volume ttl: ", bmi.water_vol_ttl)

    # for simulation mode we sometimes want to slow down the processing;
    # ... not as necessary 
    bmi.sleep_time_sec = sleep_time_sec # Delay in simulation mode

    # Flag to print out information from the proessing
    bmi.verbose = False
    bmi.verbose2 = False    # this displays the time it takes to copute ROI

    ###############################################################
    ###### INITIALIZE AND START TONE PLAYBACK (+ WATER SPOUT) #####
    ###############################################################
    '''  Here we pass only the ensemble state (i.e. E1-E2) to the 
        tone player. The tone player alone then computes the transfer function
        as this is not related to anything else in the BMI class
    '''
    #
    if calibration_flag==False:
        tone_player_ = Process(target=PlayTone, args=(fname_rois_pixels_and_thresholds,
                                                  bmi.shmem_ensemble_state.name,
                                                  bmi.shmem_tone_state.name,
                                                  bmi.shmem_termination_flag.name,
                                                  bmi.shmem_water_reward.name,
                                                  bmi.shmem_reward_lockout_counter.name,
                                                  bmi.shmem_dynamic_reward_lockout_state.name,
                                                  bmi.shmem_white_noise_state.name,
                                                  bmi.shmem_alignment_flag.name,
                                                  bmi.water_vol_ttl,
                                                  simulation_flag_tone,
                                                  calibration_flag,
                                                  bmi.sleep_time_sec,
                                                  ))
        tone_player_.start()

    ###############################################################
    ############## INITIALIZE AND START PLOTTER ###################
    ###############################################################
    '''  This is the plotting functions that visualize ROI time sries
    '''
    #
    plotter_ = Process(target=PlotROIs, args=(
                                            calibration_flag,
                                            fname_rois_pixels_and_thresholds,
                                            bmi.shmem_rois_traces_ensemble1.name,
                                            bmi.shmem_rois_traces_ensemble2.name,
                                            bmi.shmem_n_ttl.name,
                                            bmi.rois_traces_raw_ensemble1.shape,
                                            bmi.rois_traces_raw_ensemble2.shape,
                                            bmi.shmem_reward_times.name,
                                            bmi.shmem_tone_state.name,
                                            bmi.shmem_live_frame_plotter.name,
                                            bmi.shmem_ensemble_state.name,
                                            bmi.high_threshold,
                                            bmi.shmem_termination_flag.name,
                                            bmi.shmem_live_video_frame.name,
                                            bmi.shmem_high_threshold_state.name,
                                            video_width,
                                            video_height,
                                            bmi.shmem_motion_correction_flag.name,
                                            motion_flag,
                                            bmi.shmem_dynamic_f0_flag.name,
                                            bmi.shmem_manual_motion_correction_array.name,
                                            bmi.shmem_contingency_degradation.name,
                                              ))
    plotter_.start()


    ###############################################################
    ############ INITIALIZE AND START MICROPHONE ##################
    ###############################################################
    '''  Microphone recording
    '''
    #
    if calibration_flag==False:
        rec_time_n_sec = int(n_frames_session/sampleRate_2P)+1000 # add extra 10 seconds; audio always starts before video and eveyrthing else by a few sec
        mic_ =  Process(target=Microphone, args=(
                                               fname_root_path,
                                               rec_time_n_sec,
                                               bmi.shmem_termination_flag.name,
                                               ))
        
    ###############################################################
    ############ INITIALIZE AND START DRIFT CORRECTION ############
    ###############################################################
    '''  Functions that use phase correlation to fix motion artifcats
    '''
    #
    if calibration_flag==False:
        drift_ = Process(target=DriftCorrection, args=(
                                                fname_rois_pixels_and_thresholds,
                                                bmi.shmem_live_frame_motion_detector.name,
                                                bmi.shmem_drift_xy_values.name,
                                                bmi.shmem_termination_flag.name,
                                                dynamic_template_flag,
                                                ))
        drift_.start()


    ###############################################################
    ########### INITIALIZE AND START CAMERA RECORDING #############
    ###############################################################
    
    #
    root_fname_video = os.path.split(os.path.split(fname_fluorescence)[0])[0]
    
    
    if calibration_flag:
        fname_video = os.path.join(root_fname_video,
                                   "video_calibration.avi")
    elif align_flag:
        fname_video = os.path.join(root_fname_video,
                                   "video_alignment.avi")
    else:
        fname_video = os.path.join(root_fname_video,
                                   "video_data.avi")

    #
    camera_player_ = Process(target=Camera, args=(
                                                #fname_rois_pixels_and_thresholds,
                                                fname_fluorescence,
                                                simulation_flag_video_camera,
                                                video_hardware_trigger,
                                                bmi.shmem_n_ttl.name,
                                                bmi.shmem_termination_flag.name,
                                                n_frames_session,
                                                bmi.shmem_live_video_frame.name,
                                                video_width,
                                                video_height,
                                                fname_video,
                                        ))
    camera_player_.start()

    ###############################################################
    ######################### RUN BMI #############################
    ###############################################################

    # loop to wait 2 sec until plotting is initialized:
    # TODO: autod detect when plotting is initialized 
    time.sleep(2)
    
   # mic_.start()


    #
    bmi.run_BMI()
    
    
    
    # close all classes
    bmi.close()

    #
    plotter_.close()
    
    #
    camera_player_.close()
    
    if calibration_flag==False:
        tone_player_.close()
    #water_reward_.close()

    quit()



    ###############################################################
    ############## INITIALIZE WATER REWARD ########################
    ###############################################################
    #  ################# DO NOT DELETE ########################
    # '''  Here we pass only the ensemble state (i.e. E1-E2) to the
    #   tone player. The tone player alone then computes the transfer function
    #   as this is not related to anything else in the BMI class
    #   - NOT USED  used currently as it is combined with the TONE player for sequential/serial activation
    #     of NI Card as we only have one of those
    #   - TODO : use digital output on the NI card to asynchronosly send singals to both speaker/water valve
    # '''
    #
    # #
    # if False:
    #   water_reward_ = Process(target=WaterReward, args=(bmi.shmem_water_reward.name,
    #                                                     bmi.shmem_termination_flag.name,
    #                                                     simulation_flag_water,
    #                                                     ))
    #   water_reward_.start()
