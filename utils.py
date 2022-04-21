import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

import scipy
import cv2
#
class ComputeROIs(object):
	
	#
	def __init__(self, fname):
		
		#
		self.fname = fname

		#
		self.vmin = 200
		self.vmax = 500
		
		#
		self.subsample = 30 # for std computation downsample to single second 
				
		#
		self.binarize_thresh =.25
		self.sigma = .5
		self.order = 0
		self.n_smooth_steps = 1
		
	#
	def make_std_map(self):

		data = np.memmap(self.fname, dtype='uint16', mode='r')
		data = data.reshape(-1,512,512)
		print ("memmap : ", data.shape)

		data_sparse = data[::self.subsample]
		print ("data into analysis: ", data_sparse.shape)

		# filter once got remove much of the white noise
		sigma = 1
		order = 0
		print (" gaussian filter width: ", sigma, ", order: ", order)
		data_sparse = scipy.ndimage.gaussian_filter(data_sparse, 
													sigma, 
													order)
													
		std = np.std(data_sparse,axis=0)

		#
		std = (std-self.vmin)/(self.vmax-self.vmin)
		idx = np.where(std<0)
		std[idx]=0
		idx = np.where(std>1)
		std[idx]=1

		# 
		plt.figure()
		plt.imshow(std,
				   #vmin=vmin,
				   #vmax=vmax
				  )
		plt.show()
		
		self.std_map = std
		
		return std

	def area_inside_convex_hull(self, pts):
		lines = np.hstack([pts,np.roll(pts,-1,axis=0)])
		area = 0.5*abs(sum(x1*y2-x2*y1 for x1,y1,x2,y2 in lines))
		return area

	def binarize_data(self, img, thresh):
		
		#thresh = .15
		idx = np.where(img>thresh)
		img[idx]=1
		idx = np.where(img<=thresh)
		img[idx]=0
			
		return img

	#
	def find_roi_boundaries(self, image):

		for k in range(self.n_smooth_steps):
			image = scipy.ndimage.gaussian_filter(image, 
												  self.sigma, 
												  self.order)

			image = self.binarize_data(image, self.binarize_thresh)
		
		#
		image = image.astype('int32')
						
		# run watershed segmentation
		distance = ndi.distance_transform_edt(image)
		coords = peak_local_max(distance, 
								footprint=np.ones((17, 17)), 
								labels=image)

		# 
		mask = np.zeros(distance.shape, dtype=bool)
		mask[tuple(coords.T)] = True
		markers, _ = ndi.label(mask)
		labels = watershed(-distance, 
						   markers, 
						   mask=image)
		#
		labels = labels.astype('float32')

		# remove very small and very large ROIs
		min_size = 15
		max_size = 250 #???
		roi_centres = []
		indexes = []
		for k in np.unique(labels):
			idx = np.where(labels==k)
			
			
			if idx[0].shape[0]<min_size or idx[0].shape[0]>max_size:
				labels[idx]=np.nan
			else:
				
				roi_centres.append([np.median(idx[0]),
									 np.median(idx[1])])
				indexes.append(idx)

		# 
		if False:
			# scrabmel the labels
			import sklearn
			idxrand = sklearn.utils.shuffle(np.unique(labels))
			#print (idxrand)

			# 
			labels_rand = np.zeros(labels.shape)
			for ctr,k in enumerate(np.unique(labels)):
				idx = np.where(labels==k)
				#print (k, idxrand[ctr])
				labels_rand[idx] = idxrand[ctr]

			# 
			ax=plt.subplot(111)
			ax.imshow(labels_rand, 
					  cmap=plt.cm.nipy_spectral,
					  interpolation='none')
			ax.set_axis_off()

			plt.show()

		#return None
		
		self.rois = np.vstack(roi_centres)
		self.indexes = indexes
		
		#
		return self.rois, self.indexes
	# 


	def show_contour_map(self, std_map, indexes, new_plot=True):
		
		if new_plot:
			plt.figure()
			
		plt.imshow(std_map)
		for p in range(len(indexes)):
			temp = np.zeros(std_map.shape)
			temp[indexes[p]]=1
			temp = temp.astype('uint8')
			contour, _ = cv2.findContours(temp, 
											cv2.RETR_TREE, 
											cv2.CHAIN_APPROX_SIMPLE)
			contour = contour[0].squeeze()
			contour = np.vstack((contour, contour[0]))

			# 
			for k in range(len(contour)-1):
				plt.plot([contour[k][0], contour[k+1][0]],
						 [contour[k][1], contour[k+1][1]],
						c='white')
			#print (indexes[p])
			z = np.vstack(indexes[p]).T
			plt.text(np.median(z[:,1]), np.median(z[:,0]), str(p),c='red')

		plt.show()
    
		
	def compute_traces(self):
		
		data = np.memmap(self.fname, dtype='uint16', mode='r')
		data = data.reshape(-1,512,512)
		print ("memmap : ", data.shape)
			
		#  
		plt.figure()
		traces = []
		ctr=0
		ax=plt.subplot(121)
		roi_traces = []
		for k in trange(0,len(self.rois)):
			loc = np.int32(np.array(self.rois[k])/1.5)
			
			# check every .3 secs
			skip = self.subsample
			t = np.arange(0, data.shape[0], 10)/30.
			traces = []
			# step in time
			for p in range(0, data.shape[0], 10):
				
				# grab frame
				temp = data[p]
				
				# grab roi
				temp = temp[self.indexes[k]]
				
				# normalize by surface area
				if True:
					temp = temp/self.indexes[k][0].shape[0]
				
				# add data inside roi
				temp = np.nansum(temp)
				
				# save
				traces.append(temp)

			#
			traces = np.array(traces)
			traces = traces- np.median(traces)

			#
			roi_traces.append(traces)
			
			plt.plot(t, traces+ctr*self.scale)
			ctr+=1
			
		labels = np.arange(len(self.rois))
		labels_old = np.arange(0,ctr*self.scale,self.scale)
		plt.yticks(labels_old, labels)
		
		ax=plt.subplot(122)
		new_plot = False
		self.show_contour_map(self.std_map,self.indexes, new_plot)

		plt.show()
		
		self.roi_traces = roi_traces
		
		return roi_traces
    
