import pyaudio
import wave
import time
import numpy as np
import os
from multiprocessing import shared_memory

#
class Microphone():
    
    #
    def __init__(self, 
                 audio_root_dir,
                 recording_duration_sec,
                 shmem_termination_flag,
                 ):
        
        self.shmem_termination_flag = shmem_termination_flag

        print ("audio_root_dir: ", audio_root_dir)
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.CHUNK = 512
        self.RECORD_SECONDS = recording_duration_sec
        self.WAVE_OUTPUT_FILENAME = os.path.join(audio_root_dir,"audio_data.wav")
        self.fname_out_audio_npy = os.path.join(audio_root_dir,"audio_data.npy")
        self.fname_out_audio_time_stamps = os.path.join(audio_root_dir, "audio_data_timestamps.npy")
        self.audio = pyaudio.PyAudio()
        self.device_index = 1 

        #
        print ("INITIALIZING MICROPHONE ... ")
        self.stream = self.audio.open(format=self.FORMAT, 
                            channels=self.CHANNELS,
                            rate=self.RATE, 
                            input=True,
                            input_device_index = self.device_index,
                            frames_per_buffer=self.CHUNK)

        #
        self.initialize_termination_flag()

        #
        self.start2()
        
    
        #
    def initialize_termination_flag(self):

        aa = np.zeros(1, dtype=np.int64)
        self.existing_shm_termination_flag = shared_memory.SharedMemory(name=self.shmem_termination_flag)
        self.termination_flag = np.ndarray(aa.shape,
                                           dtype=aa.dtype,
                                           buffer=self.existing_shm_termination_flag.buf)
                    
        
    #
    def start2(self):
        
        print ("AUDIO REC started....")
        self.record_frames = []
        record_times = []
        for i in range(0, int(self.RATE / self.CHUNK * self.RECORD_SECONDS)):
            data = self.stream.read(self.CHUNK)
            self.record_frames.append(data)
            record_times.append(time.time())
            if self.termination_flag[0]==1:
                break

        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()
        
        print ("AUDIO REC ended....")
        print ("EXITING AUDIO/MICROPHONE CLASS")
        
        np.save(self.fname_out_audio_npy, self.record_frames)
        np.save(self.fname_out_audio_time_stamps, record_times)
        
        self.save_audio_file()
        
    #
    def save_audio_file(self):
        
        waveFile = wave.open(self.WAVE_OUTPUT_FILENAME, 'wb')
        waveFile.setnchannels(self.CHANNELS)
        waveFile.setsampwidth(self.audio.get_sample_size(self.FORMAT))
        waveFile.setframerate(self.RATE)
        waveFile.writeframes(b''.join(self.record_frames))
        waveFile.close()
