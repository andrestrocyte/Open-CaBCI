from tkinter import *
import tkinter as tk
from tkinter.filedialog import askopenfilename, askdirectory
import os
import numpy as np


def run_BMI():
    global window, bmi_flag, lick_flag, tone_flag, water_flag, video_flag, fname_root_path, simulation_sleep_box_data, n_frames_box_data, width_box_data, length_box_data, bmi_read, lick_read, tone_read, water_read, simulation_sleep, n_frames, width, length, video_read, video_hardware_trigger_flag, video_width_box_data, video_length_box_data, video_width, video_length, calibration_read, motion_flag, motion_read, template_flag, template_read, align_flag, water_vol_ttl, water_vol_box_data

    #
    bmi_read = bmi_flag.get()
    lick_read = lick_flag.get()
    tone_read = tone_flag.get()
    water_read = water_flag.get()
    video_read = video_flag.get()
    motion_read = motion_flag.get()
    template_read = template_flag.get()
    video_hardware_trigger_flag = video_hardware_trigger_flag.get()
    calibration_read = "False"
    align_flag = 0

    #
    simulation_sleep = simulation_sleep_box_data.get()
    n_frames = n_frames_box_data.get()
    width = width_box_data.get()
    length = length_box_data.get()
    video_width = int(video_width_box_data.get())
    video_length = int(video_length_box_data.get())
    water_vol_ttl = int(water_vol_box_data.get())

    #
    print("Loaded BMI params...")
    print("fname_root_path: ", fname_root_path)
    print("bmi_simulation: ", bmi_read)
    print("lick_simulation: ", lick_read)
    print("tone_simulation: ", tone_read)
    print("water_simulation: ", water_read)
    print("simulation_sleep: ", simulation_sleep, " sec")
    print("n_frames: ", n_frames)
    print("video_simulation: ", video_read)
    print("video_hardware_trigger_flag: ", video_hardware_trigger_flag)
    print("video width: ", video_width)
    print("video length: ", video_length)
    print("calibration flag: ", calibration_read)
    print("template update flag: ", template_read)
    print("align flag: ", align_flag)
    print ("GUI ttl water: ", water_vol_ttl)

    # print (fname_root_path, bmi_read, lick_read, tone_read, water_read, simulation_sleep, n_frames, video_read, video_hardware_trigger_flag )

    #
    print("RETURNING TO BMI...")

    window.destroy()


def run_Calibration():
    global window, bmi_flag, lick_flag, tone_flag, water_flag, video_flag, fname_root_path, simulation_sleep_box_data, n_frames_box_data, width_box_data, length_box_data, bmi_read, lick_read, tone_read, water_read, simulation_sleep, n_frames, width, length, video_read, video_hardware_trigger_flag, video_width_box_data, video_length_box_data, video_width, video_length, calibration_read, motion_flag, motion_read, template_flag, template_read, align_flag

    #
    bmi_read = bmi_flag.get()
    lick_read = lick_flag.get()
    tone_read = tone_flag.get()
    water_read = water_flag.get()
    video_read = video_flag.get()
    motion_read = motion_flag.get()
    template_read = template_flag.get()
    video_hardware_trigger_flag = video_hardware_trigger_flag.get()
    calibration_read = "True"
    align_flag = 0

    #
    simulation_sleep = simulation_sleep_box_data.get()
    n_frames = n_frames_box_data.get()
    width = width_box_data.get()
    length = length_box_data.get()
    video_width = int(video_width_box_data.get())
    video_length = int(video_length_box_data.get())
    water_vol_ttl = int(water_vol_box_data.get())

    #
    print("Loaded BMI params...")
    print("fname_root_path: ", fname_root_path)
    print("bmi_simulation: ", bmi_read)
    print("lick_simulation: ", lick_read)
    print("tone_simulation: ", tone_read)
    print("water_simulation: ", water_read)
    print("simulation_sleep: ", simulation_sleep, " sec")
    print("n_frames: ", n_frames)
    print("video_simulation: ", video_read)
    print("video_hardware_trigger_flag: ", video_hardware_trigger_flag)
    print("video width: ", video_width)
    print("video length: ", video_length)
    print("calibration flag: ", calibration_read)
    print("align flag: ", align_flag)
    print ("GUI ttl water: ", water_vol_ttl)

    #
    print("RETURNING TO BMI...")

    window.destroy()




def run_Alignment():
    global window, bmi_flag, lick_flag, tone_flag, water_flag, video_flag, fname_root_path, simulation_sleep_box_data, n_frames_box_data, width_box_data, length_box_data, bmi_read, lick_read, tone_read, water_read, simulation_sleep, n_frames, width, length, video_read, video_hardware_trigger_flag, video_width_box_data, video_length_box_data, video_width, video_length, calibration_read, motion_flag, motion_read, template_flag, template_read, align_flag, water_vol_ttl, water_vol_box_data

    #
    bmi_read = bmi_flag.get()
    lick_read = lick_flag.get()
    tone_read = tone_flag.get()
    water_read = water_flag.get()
    video_read = video_flag.get()
    motion_read = motion_flag.get()
    template_read = template_flag.get()
    video_hardware_trigger_flag = video_hardware_trigger_flag.get()
    calibration_read = "False"
    align_flag = 1


    #
    simulation_sleep = simulation_sleep_box_data.get()
    n_frames = n_frames_box_data.get()
    width = width_box_data.get()
    length = length_box_data.get()
    video_width = int(video_width_box_data.get())
    video_length = int(video_length_box_data.get())

    #
    print("Loaded BMI params...")
    print("fname_root_path: ", fname_root_path)
    print("bmi_simulation: ", bmi_read)
    print("lick_simulation: ", lick_read)
    print("tone_simulation: ", tone_read)
    print("water_simulation: ", water_read)
    print("simulation_sleep: ", simulation_sleep, " sec")
    print("n_frames: ", n_frames)
    print("video_simulation: ", video_read)
    print("video_hardware_trigger_flag: ", video_hardware_trigger_flag)
    print("video width: ", video_width)
    print("video length: ", video_length)
    print("calibration flag: ", calibration_read)
    print("align flag: ", align_flag)

    #
    print("RETURNING TO BMI...")

    window.destroy()



#
def gui():
    global window, bmi_flag, lick_flag, tone_flag, water_flag, video_flag, fname_root_path, simulation_sleep_box_data, n_frames_box_data, width_box_data, length_box_data, bmi_read, lick_read, tone_read, water_read, simulation_sleep, n_frames, width, length, video_read, video_hardware_trigger_flag, video_width_box_data, video_length_box_data, video_width, video_length, calibration_flag, motion_flag, motion_read, template_flag, template_read, water_vol_ttl, water_vol_box_data

    #
    OPTIONS = [
        "True",
        "False",
    ]  # etc

    #
    window = Tk()
    window.geometry('1000x800')

    fontsize = 10
    default_option = 0

    # button box to read the directory of the location of the Bscope raw data
    fname_root_path = askdirectory()
    T = Text(window, height=1, width=30)
    l = Label(window, text="path of Bscope raw data")
    l.config(font=("Courier", fontsize))
    T.insert(tk.END, fname_root_path)

    T.grid(column=1, row=0)
    l.grid(column=0, row=0)

    # simulation fals for bmi, tone and water
    # T2 = Text(window, height = 1, width = 52)

    # #
    bmi_box = Label(window, text="bmi simulation mode")
    bmi_box.config(font=("Courier", fontsize))
    bmi_box.grid(column=0, row=1)

    bmi_flag = StringVar(window)
    bmi_flag.set(OPTIONS[default_option])  # default value
    bmi_menu = OptionMenu(window, bmi_flag, *OPTIONS)
    bmi_menu.grid(column=1, row=1)

    # #
    lick_box = Label(window, text="lick detector sim mode")
    lick_box.config(font=("Courier", fontsize))
    lick_box.grid(column=0, row=2)

    lick_flag = StringVar(window)
    lick_flag.set(OPTIONS[default_option])  # default value
    lick_menu = OptionMenu(window, lick_flag, *OPTIONS)
    lick_menu.grid(column=1, row=2)

    #
    tone_box = Label(window, text="tone simulation mode")
    tone_box.config(font=("Courier", fontsize))
    tone_box.grid(column=0, row=3)

    tone_flag = StringVar(window)
    tone_flag.set(OPTIONS[default_option])  # default value
    tone_menu = OptionMenu(window, tone_flag, *OPTIONS)
    tone_menu.grid(column=1, row=3)

    # #
    water_box = Label(window, text="water simulation mode")
    water_box.config(font=("Courier", fontsize))
    water_box.grid(column=0, row=4)

    water_flag = StringVar(window)
    water_flag.set(OPTIONS[default_option])  # default value
    water_menu = OptionMenu(window, water_flag, *OPTIONS)
    water_menu.grid(column=1, row=4)

    # #
    video_box = Label(window, text="video camera simulation mode")
    video_box.config(font=("Courier", fontsize))
    video_box.grid(column=0, row=5)

    video_flag = StringVar(window)
    video_flag.set(OPTIONS[default_option])  # default value
    video_menu = OptionMenu(window, video_flag, *OPTIONS)
    video_menu.grid(column=1, row=5)

    #
    video_box2 = Label(window, text="hardware trigger mode")
    video_box2.config(font=("Courier", fontsize))
    video_box2.grid(column=2, row=5)

    video_hardware_trigger_flag = StringVar(window)
    video_hardware_trigger_flag.set(OPTIONS[1])  # default value
    video_menu2 = OptionMenu(window, video_hardware_trigger_flag, *OPTIONS)
    video_menu2.grid(column=3, row=5)

    # #
    motion_box = Label(window, text="motion detection")
    motion_box.config(font=("Courier", fontsize))
    motion_box.grid(column=0, row=6)

    motion_flag = StringVar(window)
    motion_flag.set(OPTIONS[1])  # default value
    motion_menu = OptionMenu(window, motion_flag, *OPTIONS)
    motion_menu.grid(column=1, row=6)

    # #
    template_box = Label(window, text="dynamic template updates")
    template_box.config(font=("Courier", fontsize))
    template_box.grid(column=0, row=7)

    template_flag = StringVar(window)
    template_flag.set(OPTIONS[default_option])  # default value
    template_menu = OptionMenu(window, template_flag, *OPTIONS)
    template_menu.grid(column=1, row=7)

    # # image size width/length input box
    video_width_box = Label(window, text="video - width (pixels; DO NOT CHANGE)")
    video_width_box.config(font=("Courier", fontsize))
    video_width_box.grid(column=0, row=8)

    video_width_box_data = tk.Entry(window)
    video_width_box_data.insert(END, 1824)
    video_width_box_data.grid(column=1, row=8)

    # # image size width/length input box
    video_length_box = Label(window, text="video - length (pixels; DO NOT CHANGE)")
    video_length_box.config(font=("Courier", fontsize))
    video_length_box.grid(column=0, row=9)

    video_length_box_data = tk.Entry(window)
    video_length_box_data.insert(END, 1200)
    video_length_box_data.grid(column=1, row=9)

    # # sleep timer for simulation mode
    simulation_sleep_box = Label(window, text="simulation mode sleep (in seconds)")
    simulation_sleep_box.config(font=("Courier", fontsize))
    simulation_sleep_box.grid(column=0, row=10)

    simulation_sleep_box_data = tk.Entry(window)
    simulation_sleep_box_data.insert(END, 0.0001)
    simulation_sleep_box_data.grid(column=1, row=10)
    simulation_sleep_box_data.focus_force()

    # # sleep timer for simulation mode
    n_frames_box = Label(window, text="# of frames to acquire")
    n_frames_box.config(font=("Courier", fontsize))
    n_frames_box.grid(column=0, row=11)

    n_frames_box_data = tk.Entry(window)
    n_frames_box_data.insert(END, 1000)
    n_frames_box_data.grid(column=1, row=11)

    # # image size width/length input box
    width_box = Label(window, text="imaging window - width (pixels)")
    width_box.config(font=("Courier", fontsize))
    width_box.grid(column=0, row=12)

    width_box_data = tk.Entry(window)
    width_box_data.insert(END, 512)
    width_box_data.grid(column=1, row=12)

    # # image size width/length input box
    length_box = Label(window, text="imaging window - length (pixels)")
    length_box.config(font=("Courier", fontsize))
    length_box.grid(column=0, row=13)

    length_box_data = tk.Entry(window)
    length_box_data.insert(END, 512)
    length_box_data.grid(column=1, row=13)

    # # image size width/length input box
    water_vol_box = Label(window, text="water volume (TTL duration)")
    water_vol_box.config(font=("Courier", fontsize))
    water_vol_box.grid(column=0, row=14)

    water_vol_box_data = tk.Entry(window)
    water_vol_box_data.insert(END, 15000)
    water_vol_box_data.grid(column=1, row=14)


    #
    button1 = tk.Button(text='Run BMI',
                        command=run_BMI
                        )

    #
    button1.grid(column=0, row=17)

    #
    button2 = tk.Button(text='Run Calibration',
                        command=run_Calibration
                        )

    #
    button2.grid(column=1, row=17)

    #
    button3 = tk.Button(text='Run Alignment',
                        command=run_Alignment
                        )

    button3.grid(column=3, row=17)

    #

    #
    simulation_sleep = simulation_sleep_box_data.get()
    n_frames = n_frames_box_data.get()

    #################################################################
    tk.mainloop()

    # return values to main script:

    #
    if calibration_read == "True":
        fname_gui_params = os.path.join(fname_root_path, "gui_params.npz")
    else:
        fname_gui_params = os.path.join(fname_root_path, "gui_params.npz")

    #
    np.savez(fname_gui_params,
             bmi_read=bmi_read,
             lick_read=lick_read,
             tone_read=tone_read,
             water_read=water_read,
             video_read=video_read,
             motion_read=motion_read,
             template_read=template_read,
             video_hardware_trigger_flag=video_hardware_trigger_flag,
             simulation_sleep=simulation_sleep,
             n_frames=n_frames,
             width=width,
             length=length,
             video_width=video_width,
             video_length=video_length,
             water_vol_ttl = water_vol_ttl
             )
    #
    return (fname_root_path, bmi_read, lick_read, tone_read, water_read, video_read, video_hardware_trigger_flag,
            simulation_sleep, n_frames, video_width, video_length, calibration_read, motion_read, align_flag,
            water_vol_ttl)
