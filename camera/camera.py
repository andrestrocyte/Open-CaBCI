import matplotlib.pyplot as plt
import time
from tqdm import trange
from pypylon import pylon
import cv2


class Grab():
	
	def __init__(self, width, height):
		
		self.frame = np.zeros((1200,1824))
	
	def GetArray(self):
		
		return self.frame
		

class CameraSimulation()

	def __init__(self, width, height):
		self.width = width
		self.height = height
		
		
	def RetrieveResult(self, A, B):
		
		grab = Grab(self.width, 
					self.height)
		
		return grab
		

###########################################
class VideoCamera():
	
	def __init__(self, 
                 fname_rois_pixels_and_thresholds,
				 fps,
				 camera_simulation_flag,
				 hardware_trigger_flag,
				 shmem_n_ttl,
                 shmem_termination_flag,
				):
		
		#
		self.fname_rois_pixels_and_thresholds = fname_rois_pixels_and_thresholds

		#
		self.fname_video = os.path.join(os.path.split(self.fname_rois_pixels_and_thresholds)[0],
													  "video.avi")
		
		#
		self.hardware_trigger_flag = hardware_trigger_flag
		  
		# run camera in simualtion mode, doesn't require a camera
		self.camera_simulation_flag = camera_simulation_flag
		  
		#
		self.shmem_n_ttl = shmem_n_ttl
        
        #
        self.shmem_termination_flag = shmem_termination_flag
		
		# 
		self.video_frame_times = []

		# initialize params 
		# TODO: expose these in time.
		self.fps = fps
		self.video_height = 1200
		self.video_width = 1824
		self.exposure_time = 20000   # time to collect light for each frame in uSec
								     # TODO: need to lower this and increase intensity etc for higher frame rates

		# initialize video writer
		fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
		self.video_out = cv2.VideoWriter(self.fname_video, 
										 fourcc, 
										 fps, 
										 (width, height))
		
		# 
		print ("TOTEST Single channel video.... (uncomment below)")
		#fourcc = cv2.VideoWriter_fourcc(*'XVID')  # not sure this is the right one?!
		#self.video_out = cv2.VideoWriter(self.fname_video, 
		#								  fourcc, 
		#								  fps, 
		#									  (width, height), 
		#								  0                  # this seems to make 1 channel vids... might be faster than before
		#								  )
		
		# 
		if self.camera_simulation_mode==False:
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

	def initalize_camera_simulation(self):
		'''
		'''
		self.camera = CameraSimulation()
		
    #
    def initalize_n_ttl(self):

        #
        print ("  nttl memory name : ", self.shmem_n_ttl)

        aa = np.zeros((1,), dtype=np.int64)

        # get the rois_traces from the shared memory name
        self.existing_shm_n_ttl = shared_memory.SharedMemory(name=self.shmem_n_ttl)

        #
        self.n_ttl = np.ndarray(aa.shape,
                                 dtype=aa.dtype,
                                 buffer=self.existing_shm_n_ttl.buf)
                                 
        #
        self.n_ttl_last = self.n_ttl[0].copy()   # not sure if copy is required for broadcasts!?
        
					
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

		tl_factory = pylon.TlFactory.GetInstance()
		self.camera = pylon.InstantCamera()
		self.camera.Attach(tl_factory.CreateDevice(devices[1]))
		print("DeviceClass: ", self.camera.GetDeviceInfo().GetDeviceClass())
		print("DeviceFactory: ", self.camera.GetDeviceInfo().GetDeviceFactory())
		print("ModelName: ", self.camera.GetDeviceInfo().GetModelName())

		# 
		self.camera.Open()
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
			
			try:
				if self.check_ttl_change():
					grab = self.camera.RetrieveResult(10000, 
												 pylon.TimeoutHandling_ThrowException)
						
					# get the image data from array
					frame = grab.GetArray()
						
					# format the image to be saved for 
					gray = cv2.normalize(frame, None, 255, 0, 
											 norm_type=cv2.NORM_MINMAX, 
											 dtype=cv2.CV_8U)
					gray_3c = cv2.merge([gray, gray, gray])
					self.video_out.write(gray_3c)
						
					#
					self.times.append([k ,time.time()])
					
					#
					self.n_ttl_last = self.n_ttl[0]

					
				# can check for termination flag
				if self.termination_flag[0]==1:
					break
					
				# can also check if n_ttl = total frames
				if self.n_frames = (self.n_ttl[0]-1):
					break

			except:
				print ("Video camera crashed early....")

	# 
	def hardware_trigger_record(self):
		
		''' This record function uses the pin break out Line4 on the back of the 
		    Basler camera to receive and process the Bscope TTL signal indicating a 2p image has been completed
		    - this is essentially an independent signal from the main BMI 
		      processing of the TTL signal
		'''
		
		#
		fourcc = cv2.VideoWriter_fourcc(*'XVID')
		
		#
		for k in trange(self.n_frames):
			try:
				# get a frame with a timeout of 10ms (actually seems to be much longer, about 5-10sec
				# TODO: check why this time out takes so long...
				grab = self.camera.RetrieveResult(10000, 
										 pylon.TimeoutHandling_ThrowException)
				
				# get the image data from array
				frame = grab.GetArray()
				
				# format the image to be saved for 
				gray = cv2.normalize(frame, None, 255, 0, 
									 norm_type=cv2.NORM_MINMAX, 
									 dtype=cv2.CV_8U)
				gray_3c = cv2.merge([gray, gray, gray])
				self.video_out.write(gray_3c)
				
				#
				self.times.append([k ,time.time()])
			except:
				print ("TIME OUT")
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

		np.save(self.fname_video[:-4]+"_times.npy", self.video_frame_times)

print ("Done")




