import matplotlib.pyplot as plt
import time
import numpy as np
from tqdm import trange
from pypylon import pylon
import cv2
from multiprocessing import shared_memory
import os

class Grab():
	''' An additional class required by the CameraSimulation class
	    - not used in live imaging
	'''
	#
	def __init__(self, width, height, randomize = True):

		#
		self.randomize=True

		# for some reason need to change order of heigh-width here
		self.frame = np.zeros((height,width), dtype='uint8')

	#
	def GetArray(self):

		#
		if self.randomize:
			return np.random.randint(0,255, size=self.frame.shape).astype('uint')
		
		return self.frame 
		
#
class CameraSimulation():
	''' A basic simulation class for testing camera + bmi offline
	'''

	#
	def __init__(self, width, height):
		self.width = width
		self.height = height

		self.grab = Grab(self.width,
					     self.height)

	#
	def RetrieveResult(self, A, B):
		
		return self.grab
		

###########################################
class Camera():
	
	def __init__(self, 
                 fname_rois_pixels_and_thresholds,
				 camera_simulation_flag,
				 hardware_trigger_flag,
				 shmem_n_ttl,
                 shmem_termination_flag,
				 n_frames,
				):

		#
		self.fname_rois_pixels_and_thresholds = fname_rois_pixels_and_thresholds

		# run camera in simulation mode, doesn't require a camera
		self.camera_simulation_flag=camera_simulation_flag

		#
		self.fname_video = os.path.join(os.path.split(self.fname_rois_pixels_and_thresholds)[0],
													  "video.avi")
		
		#
		self.hardware_trigger_flag = hardware_trigger_flag

		#
		self.n_frames = n_frames

		# fps to be saved for video; Hardcoded to 30fps
		# TODO: may wish to modify/provide this option in the GUI
		self.video_single_channel_flag = True
		self.fps = 30
		self.video_height = 1200
		self.video_width = 1824
		self.frame_exposure_time = 10000   # time to collect light for each frame in uSec
										# TODO: need to lower this and increase intensity etc for higher frame rates

		# a list for the n_ttl and absolute time values
		self.times = []

		#
		self.shmem_n_ttl = shmem_n_ttl

		#
		self.shmem_termination_flag = shmem_termination_flag
		
		#
		self.video_frame_times = []

		# initialize video recording
		self.initialize_video_out()

		# initalize camera hardware
		if self.camera_simulation_flag==False:
			self.initialize_camera()
		else:
			self.initialize_camera_simulation()

		#
		self.initialize_termination_flag()
		
		#
		self.initialize_n_ttl()
		
		#
		if self.hardware_trigger_flag:
			self.hardware_trigger_record()
		else:
			self.software_trigger_record()

		#
		self.save_data()

	def save_data(self):

		np.savez(self.fname_video[:-4] + ".npz",
				 vide_frame_times = self.video_frame_times)

	#
	def initialize_video_out(self):

		# initialize video writer
		if self.video_single_channel_flag==False:
			fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
			self.video_out = cv2.VideoWriter(self.fname_video,
											 fourcc,
											 self.fps,
											 (self.video_width, self.video_height))

		#
		else:
			print("TO TEST Single channel video.... (uncomment below)")
			fourcc = cv2.VideoWriter_fourcc(*'XVID')  
			self.video_out = cv2.VideoWriter(self.fname_video,
											  fourcc,
											  self.fps,
											  (self.video_width,self.video_height),
											  0                  # this seems to make 1 channel vids... might be faster than before
											  )

	#
	def initialize_camera_simulation(self):
		'''
		'''
		self.camera = CameraSimulation(self.video_width,
									   self.video_height)

	#
	def initialize_n_ttl(self):

		#
		aa = np.zeros((1,), dtype=np.int64)
		self.existing_shm_n_ttl = shared_memory.SharedMemory(name=self.shmem_n_ttl)
		self.n_ttl = np.ndarray(aa.shape,
                                 dtype=aa.dtype,
                                 buffer=self.existing_shm_n_ttl.buf)
		#
		self.n_ttl_last = self.n_ttl[0].copy()   # not sure if copy is required for broadcasts!?

	#
	def initialize_termination_flag(self):

		aa = np.zeros(1, dtype=np.int64)
		self.existing_shm_termination_flag = shared_memory.SharedMemory(name=self.shmem_termination_flag)
		self.termination_flag = np.ndarray(aa.shape,
                                           dtype=aa.dtype,
                                           buffer=self.existing_shm_termination_flag.buf)
					 
	#
	def initialize_camera(self):

		# TODO: is this required anymore?
		try:
			self.camera.Close()
		except:
			pass

		#
		tl_factory = pylon.TlFactory.GetInstance()
		devices = tl_factory.EnumerateDevices()
		for ctr,device in enumerate(devices):
			print("vid camera: ", ctr, device.GetFriendlyName())

		camera_id = 1
		print ("Selecting video camera ID #1")

		tl_factory = pylon.TlFactory.GetInstance()
		self.camera = pylon.InstantCamera()
		self.camera.Attach(tl_factory.CreateDevice(devices[camera_id]))
		print("DeviceClass: ", self.camera.GetDeviceInfo().GetDeviceClass())
		print("DeviceFactory: ", self.camera.GetDeviceInfo().GetDeviceFactory())
		print("ModelName: ", self.camera.GetDeviceInfo().GetModelName())

		# 
		self.camera.Open()
		self.camera.Width = self.video_width
		self.camera.Height = self.video_height
		self.camera.ExposureTime.SetValue(self.frame_exposure_time)  # exposure time in microseconds

		# put camera in hardware trigger mode if selected
		if self.hardware_trigger_flag:
			self.camera.TriggerSource.SetValue("Line4")
			self.camera.TriggerMode.SetValue("On")
			self.camera.TriggerActivation.SetValue('RisingEdge')
			self.camera.TriggerSelector.SetValue('FrameStart')
		
		# start the camera to wait for triggers
		self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

	#
	def check_ttl_change(self):
		''' In software trigger mode, checks to see if a new ttl pulse was detected
		'''
		
		if self.n_ttl_last<self.n_ttl[0]:
			return True
		
		return False

	# 
	def software_trigger_record(self):
		
		''' This record function uses ttl state signal from the main BMI 
		    - it should be generally be used when developing/debugging the
		      code, otherwise the hardware trigger may be more preferrable (not clear)
		'''
		
		while True:
		#for n in trange(self.n_frames, desc='camera processing'):
			if True:
				if self.check_ttl_change():
					grab = self.camera.RetrieveResult(10000, 
												       pylon.TimeoutHandling_ThrowException)
						
					# get the image data from array
					frame = grab.GetArray()

					#
					#print ("grabbed frame: ", frame.shape, type(frame[0,0]))
						
					# format the image to be saved for
					if self.video_single_channel_flag == False:
						gray = cv2.normalize(frame, None, 255, 0,
												 norm_type=cv2.NORM_MINMAX,
												 dtype=cv2.CV_8U)
						gray_3c = cv2.merge([gray, gray, gray])
						self.video_out.write(gray_3c)
					else:
						gray = cv2.normalize(frame, None, 255, 0,
												 norm_type=cv2.NORM_MINMAX,
												 dtype=cv2.CV_8U)
						self.video_out.write(gray)

					#
					self.video_frame_times.append([self.n_ttl[0] ,time.time()])
					
					#
					self.n_ttl_last = self.n_ttl[0]


					#print ("Camera: grabbed frame: ", self.n_ttl)

					
				# can check for termination flag
				if self.termination_flag[0]==1:
					break
					
				# can also check if n_ttl = total frames
				if self.n_frames == (self.n_ttl[0]-1):
					break

			#except:
			#	print ("Video camera crashed early....")

	# 
	def hardware_trigger_record(self):
		
		''' This record function uses the pin break out Line4 on the back of the 
		    Basler camera to receive and process the Bscope TTL signal indicating a 2p image has been completed
		    - this is essentially an independent signal from the main BMI 
		      processing of the TTL signal
		'''
		
		#
		#fourcc = cv2.VideoWriter_fourcc(*'XVID')
		print ("CAMERA HARDWARE VERSION NOT FULLY TESTED...")
		
		#
		for k in trange(self.n_frames):
			try:
				# get a frame with a timeout of 10ms (actually seems to be much longer, about 5-10sec
				# TODO: check why this time out takes so long...
				grab = self.camera.RetrieveResult(10000, 
										 pylon.TimeoutHandling_ThrowException)
				
							# get the image data from array
				frame = grab.GetArray()

				#
				#print ("grabbed frame: ", frame.shape, type(frame[0,0]))
					
				# format the image to be saved for
				if self.video_single_channel_flag == False:
					gray = cv2.normalize(frame, None, 255, 0,
											 norm_type=cv2.NORM_MINMAX,
											 dtype=cv2.CV_8U)
					gray_3c = cv2.merge([gray, gray, gray])
					self.video_out.write(gray_3c)
				else:
					gray = cv2.normalize(frame, None, 255, 0,
											 norm_type=cv2.NORM_MINMAX,
											 dtype=cv2.CV_8U)
					self.video_out.write(gray)

				#
				self.video_frame_times.append([self.n_ttl[0] ,time.time()])
				
				#
				self.n_ttl_last = self.n_ttl[0]


				#print ("Camera: grabbed frame: ", self.n_ttl)

					
				# can check for termination flag
				if self.termination_flag[0]==1:
					break
					
				# can also check if n_ttl = total frames
				if self.n_frames == (self.n_ttl[0]-1):
					break
				
			except:
				print ("TIME OUT - hardware TTL not detected or long periods... exiting")
				break

		try:
			# close out the video writer
			self.video_out.release()
		except:
			print ("WARNING VIDEO WRITER DIDN'T CLOSE")
		try:
			self.camera.Close()
		except:
			print ("WARNING CAMERA didn't close")





