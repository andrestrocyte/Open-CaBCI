from tkinter import * 
import tkinter as tk
from tkinter.filedialog import askopenfilename, askdirectory
  
  
def exit_gui():
    global window
    window.destroy()

#
def gui():
    global window
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
    w1 = OptionMenu(window, bmi_flag, *OPTIONS)
    w1.grid(column=1,row=1)

    #
    tone_box = Label(window, text = "tone simulation mode (True/False)")
    tone_box.config(font =("Courier", 14))  
    tone_box.grid(column=0, row=2)

    tone_flag = StringVar(window)
    tone_flag.set(OPTIONS[0]) # default value
    w2 = OptionMenu(window, tone_flag, *OPTIONS)
    w2.grid(column=1,row=2)


    # #
    water_box = Label(window, text = "water simulation mode (True/False)")
    water_box.config(font =("Courier", 14))  
    water_box.grid(column=0, row=3)

    water_flag = StringVar(window)
    water_flag.set(OPTIONS[0]) # default value
    w3 = OptionMenu(window, water_flag, *OPTIONS)
    w3.grid(column=1,row=3)

    # # sleep timer for simulation mode

    simulation_sleep_box = Label(window, text = "simulation mode sleep (in seconds)")
    simulation_sleep_box.config(font =("Courier", 14))  
    simulation_sleep_box.grid(column=0,row=4)

    simulation_sleep_box_data = tk.Entry(window) 
    simulation_sleep_box_data.insert(END, 0.01)
    simulation_sleep_box_data.grid(column=1,row=4)

    # # sleep timer for simulation mode
    n_frames_box = Label(window, text = "# of frames to acquire")
    n_frames_box.config(font =("Courier", 14))  
    n_frames_box.grid(column=0,row=5)

    n_frames_box_data = tk.Entry(window) 
    n_frames_box_data.insert(END, 10000)
    n_frames_box_data.grid(column=1,row=5)

    # # image size width/length input box
    width_box = Label(window, text = "imaging window - width (pixels)")
    width_box.config(font =("Courier", 14))  
    width_box.grid(column=0,row=6)

    width_box_data = tk.Entry(window) 
    width_box_data.insert(END, 512)
    width_box_data.grid(column=1,row=6)

    # # image size width/length input box
    length_box = Label(window, text = "imaging window - length (pixels)")
    length_box.config(font =("Courier", 14))  
    length_box.grid(column=0,row=7)

    length_box_data = tk.Entry(window) 
    length_box_data.insert(END, 512)
    length_box_data.grid(column=1,row=7)

    #
    button1 = tk.Button(text='Run BMI', 
                        command=exit_gui)
    button1.grid(column=0, row=15)
    
    bmi_read = bmi_flag.get()
    tone_read = tone_flag.get()
    water_read = water_flag.get()
    
    #
    simulation_sleep = simulation_sleep_box_data.get()
    n_frames = n_frames_box_data.get()
    
    
    #
    tk.mainloop()

    # return values to main script:
    
    print ("RUNNING BMI...")

    return fname_root_path, bmi_read, tone_read, water_read, simulation_sleep, n_frames 

