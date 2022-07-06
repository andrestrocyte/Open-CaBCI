from tkinter import * 
import tkinter as tk
from tkinter.filedialog import askopenfilename, askdirectory
import os
import numpy as np
  
def exit_gui():
    global window, bmi_flag, lick_flag, tone_flag, water_flag, fname_root_path, simulation_sleep_box_data, n_frames_box_data, width_box_data, length_box_data, bmi_read, lick_read, tone_read, water_read, simulation_sleep, n_frames, width, length    

    bmi_read = bmi_flag.get()
    lick_read = lick_flag.get()
    tone_read = tone_flag.get()
    water_read = water_flag.get()
    
    #
    simulation_sleep = simulation_sleep_box_data.get()
    n_frames = n_frames_box_data.get()
    width = width_box_data.get(),
    length = length_box_data.get()
    #
    
    print ("Loaded BMI params...")
    print ("fname_root_path, bmi_read, lick_read, tone_read, water_read, simulation_sleep, n_frames ")
    print (fname_root_path, bmi_read, lick_read, tone_read, water_read, simulation_sleep, n_frames )

    #
    print ("RETURNING TO BMI...")   
    
              
    window.destroy()


#
def gui():
    global window, bmi_flag, lick_flag, tone_flag, water_flag, fname_root_path, simulation_sleep_box_data, n_frames_box_data, width_box_data, length_box_data, bmi_read, lick_read, tone_read, water_read, simulation_sleep, n_frames, width, length    

    #
    OPTIONS = [
    "True",
    "False",
    ] #etc

    #
    window = Tk()
    window.geometry('800x800')

    # button box to read the directory of the location of the Bscope raw data 
    fname_root_path = askdirectory()
    T = Text(window, height = 1, width = 52)
    l = Label(window, text = "path of Bscope raw data")
    l.config(font =("Courier", 14))  
    T.insert(tk.END, fname_root_path)

    T.grid(column=1, row=0)
    l.grid(column=0, row=0)

    # simulation fals for bmi, tone and water
    #T2 = Text(window, height = 1, width = 52)

    # #
    bmi_box = Label(window, text = "bmi simulation mode (True/False)")
    bmi_box.config(font =("Courier", 14))  
    bmi_box.grid(column=0, row=1)

    bmi_flag = StringVar(window)
    bmi_flag.set(OPTIONS[0]) # default value
    bmi_menu = OptionMenu(window, bmi_flag, *OPTIONS)
    bmi_menu.grid(column=1,row=1)
    
      # #
    lick_box = Label(window, text = "lick detector mode (True/False)")
    lick_box.config(font =("Courier", 14))  
    lick_box.grid(column=0, row=2)

    lick_flag = StringVar(window)
    lick_flag.set(OPTIONS[1]) # default value
    lick_menu = OptionMenu(window, lick_flag, *OPTIONS)
    lick_menu.grid(column=1,row=2)
    
    #
    tone_box = Label(window, text = "tone simulation mode (True/False)")
    tone_box.config(font =("Courier", 14))  
    tone_box.grid(column=0, row=3)

    tone_flag = StringVar(window)
    tone_flag.set(OPTIONS[1]) # default value
    tone_menu = OptionMenu(window, tone_flag, *OPTIONS)
    tone_menu.grid(column=1,row=3)


    # #
    water_box = Label(window, text = "water simulation mode (True/False)")
    water_box.config(font =("Courier", 14))  
    water_box.grid(column=0, row=4)

    water_flag = StringVar(window)
    water_flag.set(OPTIONS[1]) # default value
    water_menu = OptionMenu(window, water_flag, *OPTIONS)
    water_menu.grid(column=1,row=4)

    # # sleep timer for simulation mode

    simulation_sleep_box = Label(window, text = "simulation mode sleep (in seconds)")
    simulation_sleep_box.config(font =("Courier", 14))  
    simulation_sleep_box.grid(column=0,row=5)

    simulation_sleep_box_data = tk.Entry(window) 
    simulation_sleep_box_data.insert(END, 0.0001)
    simulation_sleep_box_data.grid(column=1,row=5)
    simulation_sleep_box_data.focus_force()

    # # sleep timer for simulation mode
    n_frames_box = Label(window, text = "# of frames to acquire")
    n_frames_box.config(font =("Courier", 14))  
    n_frames_box.grid(column=0,row=6)

    n_frames_box_data = tk.Entry(window) 
    n_frames_box_data.insert(END, 10000)
    n_frames_box_data.grid(column=1,row=6)

    # # image size width/length input box
    width_box = Label(window, text = "imaging window - width (pixels)")
    width_box.config(font =("Courier", 14))  
    width_box.grid(column=0,row=7)

    width_box_data = tk.Entry(window) 
    width_box_data.insert(END, 512)
    width_box_data.grid(column=1,row=7)

    # # image size width/length input box
    length_box = Label(window, text = "imaging window - length (pixels)")
    length_box.config(font =("Courier", 14))  
    length_box.grid(column=0,row=8)

    length_box_data = tk.Entry(window) 
    length_box_data.insert(END, 512)
    length_box_data.grid(column=1,row=8)


    #
    button1 = tk.Button(text='Run BMI', 
                        command=exit_gui
                        #command=check_values
                        )
                        
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
             simulation_sleep = simulation_sleep,
             n_frames = n_frames,
             width = width,
             length = length
             )
    #
    return (fname_root_path, bmi_read, lick_read, tone_read, water_read, simulation_sleep, n_frames )
