from tkinter import * 
import tkinter as tk
from tkinter.filedialog import askopenfilename, askdirectory
import os
import numpy as np
  
def exit_gui():
    global window, bmi_flag, lick_flag, tone_flag, water_flag, video_flag, fname_root_path, simulation_sleep_box_data, n_frames_box_data, width_box_data, length_box_data, bmi_read, lick_read, tone_read, water_read, simulation_sleep, n_frames, width, length, video_read, video_hardware_trigger_flag

    bmi_read = bmi_flag.get()
    lick_read = lick_flag.get()
    tone_read = tone_flag.get()
    water_read = water_flag.get()
    video_read = video_flag.get()
    video_hardware_trigger_flag = video_hardware_trigger_flag.get()

    #
    simulation_sleep = simulation_sleep_box_data.get()
    n_frames = n_frames_box_data.get()
    width = width_box_data.get(),
    length = length_box_data.get()
    #
    
    print ("Loaded BMI params...")
    print ("fname_root_path: ", fname_root_path)
    print ("bmi_simulation: ", bmi_read)
    print ("lick_simulation: ", lick_read)
    print ("tone_simulation: ", tone_read)
    print ("water_simulation: ", water_read)
    print ("simulation_sleep: ", simulation_sleep , " sec")
    print ("n_frames: ", n_frames)
    print ("video_simulation: ", video_read)
    print ("video_hardware_trigger_flag: ", video_hardware_trigger_flag)
    #print (fname_root_path, bmi_read, lick_read, tone_read, water_read, simulation_sleep, n_frames, video_read, video_hardware_trigger_flag )

    #
    print ("RETURNING TO BMI...")   
    
              
    window.destroy()


#
def gui():
    global window, bmi_flag, lick_flag, tone_flag, water_flag, video_flag, fname_root_path, simulation_sleep_box_data, n_frames_box_data, width_box_data, length_box_data, bmi_read, lick_read, tone_read, water_read, simulation_sleep, n_frames, width, length, video_read, video_hardware_trigger_flag

    #
    OPTIONS = [
    "True",
    "False",
    ] #etc

    #
    window = Tk()
    window.geometry('800x800')

    fontsize = 10
    default_option = 0


    # button box to read the directory of the location of the Bscope raw data 
    fname_root_path = askdirectory()
    T = Text(window, height = 1, width = 30)
    l = Label(window, text = "path of Bscope raw data")
    l.config(font =("Courier", fontsize))
    T.insert(tk.END, fname_root_path)

    T.grid(column=1, row=0)
    l.grid(column=0, row=0)

    # simulation fals for bmi, tone and water
    #T2 = Text(window, height = 1, width = 52)

    # #
    bmi_box = Label(window, text = "bmi simulation mode")
    bmi_box.config(font =("Courier", fontsize))
    bmi_box.grid(column=0, row=1)

    bmi_flag = StringVar(window)
    bmi_flag.set(OPTIONS[default_option]) # default value
    bmi_menu = OptionMenu(window, bmi_flag, *OPTIONS)
    bmi_menu.grid(column=1,row=1)
    
      # #
    lick_box = Label(window, text = "lick detector sim mode")
    lick_box.config(font =("Courier", fontsize))
    lick_box.grid(column=0, row=2)

    lick_flag = StringVar(window)
    lick_flag.set(OPTIONS[default_option]) # default value
    lick_menu = OptionMenu(window, lick_flag, *OPTIONS)
    lick_menu.grid(column=1,row=2)
    
    #
    tone_box = Label(window, text = "tone simulation mode")
    tone_box.config(font =("Courier", fontsize))
    tone_box.grid(column=0, row=3)

    tone_flag = StringVar(window)
    tone_flag.set(OPTIONS[default_option]) # default value
    tone_menu = OptionMenu(window, tone_flag, *OPTIONS)
    tone_menu.grid(column=1,row=3)

    # #
    water_box = Label(window, text = "water simulation mode")
    water_box.config(font =("Courier", fontsize))
    water_box.grid(column=0, row=4)

    water_flag = StringVar(window)
    water_flag.set(OPTIONS[default_option]) # default value
    water_menu = OptionMenu(window, water_flag, *OPTIONS)
    water_menu.grid(column=1,row=4)

    # #
    video_box = Label(window, text = "video camera simulation mode")
    video_box.config(font =("Courier", fontsize))
    video_box.grid(column=0, row=5)

    video_flag = StringVar(window)
    video_flag.set(OPTIONS[default_option]) # default value
    video_menu = OptionMenu(window, video_flag, *OPTIONS)
    video_menu.grid(column=1,row=5)

    #
    video_box2 = Label(window, text = "hardware trigger mode")
    video_box2.config(font =("Courier", fontsize))
    video_box2.grid(column=2, row=5)

    video_hardware_trigger_flag = StringVar(window)
    video_hardware_trigger_flag.set(OPTIONS[1]) # default value
    video_menu2 = OptionMenu(window, video_hardware_trigger_flag, *OPTIONS)
    video_menu2.grid(column=3,row=5)

    # # sleep timer for simulation mode
    simulation_sleep_box = Label(window, text = "simulation mode sleep (in seconds)")
    simulation_sleep_box.config(font =("Courier", fontsize))
    simulation_sleep_box.grid(column=0,row=6)

    simulation_sleep_box_data = tk.Entry(window) 
    simulation_sleep_box_data.insert(END, 0.0001)
    simulation_sleep_box_data.grid(column=1,row=6)
    simulation_sleep_box_data.focus_force()

    # # sleep timer for simulation mode
    n_frames_box = Label(window, text = "# of frames to acquire")
    n_frames_box.config(font =("Courier", fontsize))
    n_frames_box.grid(column=0,row=7)

    n_frames_box_data = tk.Entry(window) 
    n_frames_box_data.insert(END, 1000)
    n_frames_box_data.grid(column=1,row=7)

    # # image size width/length input box
    width_box = Label(window, text = "imaging window - width (pixels)")
    width_box.config(font =("Courier", fontsize))
    width_box.grid(column=0,row=8)

    width_box_data = tk.Entry(window) 
    width_box_data.insert(END, 512)
    width_box_data.grid(column=1,row=8)

    # # image size width/length input box
    length_box = Label(window, text = "imaging window - length (pixels)")
    length_box.config(font =("Courier", fontsize))
    length_box.grid(column=0,row=9)

    length_box_data = tk.Entry(window) 
    length_box_data.insert(END, 512)
    length_box_data.grid(column=1,row=9)

    #
    button1 = tk.Button(text='Run BMI', 
                        command=exit_gui
                        )

    #
    button1.grid(column=0, row=15)

   
    #
    simulation_sleep = simulation_sleep_box_data.get()
    n_frames = n_frames_box_data.get()
    
    #
    tk.mainloop()

    # return values to main script:
    

      #
    fname_gui_params = os.path.join(fname_root_path, "gui_params.npz")
    np.savez(fname_gui_params,
             bmi_read = bmi_read,
             lick_read = lick_read,
             tone_read = tone_read,
             water_read = water_read,
             video_read = video_read,
             video_hardware_trigger_flag = video_hardware_trigger_flag,
             simulation_sleep = simulation_sleep,
             n_frames = n_frames,
             width = width,
             length = length
             )
    #
    return (fname_root_path, bmi_read, lick_read, tone_read, water_read, video_read, video_hardware_trigger_flag, simulation_sleep, n_frames )
