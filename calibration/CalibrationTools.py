import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange, tqdm
#import multiprocessing
#multiprocessing.set_start_method('spawn')

from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import signal
import scipy
import scipy.ndimage
import cv2
from matplotlib.widgets import Slider, Button, RadioButtons
import os, pickle


#from stardist.models import StarDist2D
from utils.utils import smooth_ca_time_series4, compute_dff0, compute_dff0_with_reference, get_mode


##############################
##############################
##############################
class CalibrationTools(object):

    #
    def __init__(self, fname):

        #
        self.fname = fname

        #
        self.binarize_thresh = .05
        self.sigma = .5
        self.order = 0
        self.n_smooth_steps = 1

        #
        # data = np.memmap(self.fname, dtype='uint16', mode='r+')
        # data = np.memmap(self.fname, dtype='uint16', mode='r')
        data = np.fromfile(self.fname, dtype='uint16')
        self.data = data.reshape(-1, 512, 512)
        print("memmap : ", self.data.shape)

    def load_data_mmap(self, fname, n_frames):

        #
        data = np.memmap(fname,
                         dtype='uint16',
                         mode='r',
                         shape=n_frames * 512 * 512).reshape(n_frames, 512, 512)

        print("loaded: ", fname, data.shape)

        return data

    #
    def make_corr_map(self):
        ''' Not yet working or tested etc.

		'''

        data = np.memmap(self.fname, dtype='uint16', mode='r')
        data = data.reshape(-1, 512, 512)
        print("memmap : ", data.shape)

        data_sparse = data[::self.subsample]
        print("data into analysis: ", data_sparse.shape)

        #
        img = scipy.signal.correlate2d(data_sparse[0],
                                       data_sparse[1],
                                       mode='same')

        #
        plt.figure()
        plt.imshow(img,
                   )
        plt.show()

        return img

    def process2(self, order_type):

        #
        cell_ids = np.arange(len(self.footprints))

        #
        data = np.memmap(self.fname, dtype='uint16', mode='r')
        data = data.reshape(-1, 512, 512)

        #####################################################
        ################ COMPUTE ROI TRACES #################
        #####################################################
        #
        roi_traces = []
        for k in range(len(cell_ids)):
            roi_traces.append([])

        # loop over each frame
        for p in trange(0, data.shape[0], self.subsample,
                        desc='computing roi traces for SNR indexing'):

            # grab frame
            frame = data[p]

            # loop over ROIS
            ctr = 0
            for k in cell_ids:
                # grab roi
                temp = frame[self.footprints[k]]

                # normalize by surface area so that cells don't look way different because of footprint size
                if True:
                    temp = temp / self.footprints[k][0].shape[0]

                # add pixel values inside roi
                temp = np.nansum(temp)

                # save
                roi_traces[ctr].append(temp)
                ctr += 1
        #
        self.roi_traces = np.array(roi_traces)
        plt.figure()
        for k in range(len(roi_traces)):
            plt.plot(self.roi_traces[k]+k*5000)
        plt.show()
        ###########################################################
        ################### COMPUTE F0 AND SNR ####################
        ###########################################################
        # compute the baseline f0 of the cells in order to be able to offset it in the BMI
        # TODO: this is important; it functions as a rough DFF method
        #    TODO: we may wish to implement a more complex version of this
        self.roi_f0s = np.zeros(self.roi_traces.shape[0], dtype=np.float32)
        self.roi_snrs = np.zeros(self.roi_traces.shape[0], dtype=np.float32)
        for k in cell_ids:
            #
            f0 = np.median(self.roi_traces[k])

            #
            self.roi_f0s[k] = f0

            #
            self.roi_snrs[k] = np.max(self.roi_traces[k] / f0)

        ###########################################################
        ################# REORDER CELLS BY SNR  ###################
        ###########################################################
        if order_type == 'f0':
            idx = np.argsort(self.roi_f0s)[::-1]

        elif order_type == 'snr':
            idx = np.argsort(self.roi_snrs)[::-1]
        else:
            print(" ERROR - type not known")

        #
        self.roi_traces = self.roi_traces[idx]


    #
    def make_max_proj_map(self):

        #
        data_sparse = self.data[::self.subsample]
        print("data into analysis: ", data_sparse.shape)

        # filter once to remove much of the white noise
        if False:
            sigma = 1
            order = 0
            print(" gaussian filter width: ", sigma, ", order: ", order)
            data_sparse = scipy.ndimage.gaussian_filter(data_sparse,
                                                        sigma,
                                                        order)
            print("done filtering... (TO CHECK which axis are we filtering!!)")

        maxproj = np.max(data_sparse, axis=0)

        # std = np.std(data_sparse, axis=0)

        return maxproj

    #
    def filter_data_make_std_map(self):

        #
        data_sparse = self.data[::self.subsample]
        print("data into analysis: ", data_sparse.shape)

        # filter once to remove much of the white noise
        if True:
            #sigma = 1
            #order = 0
            print(" gaussian filter width: ", self.sigma, ", order: ", self.order)
            self.data_filtered = scipy.ndimage.gaussian_filter(data_sparse,
                                                        self.sigma,
                                                        self.order)

            print("done filtering... (TO CHECK which axis are we filtering!!)")

        #
        if False:
            kernel = [7, 1, 1]  # filter only across time
            print(" median filter width: ", kernel)
            data_sparse = signal.medfilt(data_sparse, kernel)
            print("done median filtering... ")

        #
        if False:
            # scipy.ndimage.gaussian_filter1d
            # import scipy.ndimage # import gaussian_filter1d
            # kernel = [1,1,7]
            kernel = 30
            print(" filter1d: ", kernel)
            data_sparse = scipy.ndimage.gaussian_filter1d(data_sparse, kernel)
            print("done filter1d... ", data_sparse.shape)

        #
        if False:

            #
            if False:
                import parmap
                n_cores = 8
                idx = np.array_split(np.arange(data_sparse.shape[1]), n_cores)
                # print ("data split idx: ", idx)

                res = parmap.map(convolve_parallel,
                                 idx,
                                 data_sparse,
                                 pm_processes=n_cores,
                                 pm_pbar=True)

                #
                print(" len res: ", len(res), res[0].shape)

                #
                data_sparse = np.sum(data_sparse, axis=0)
                print("recombined data sparse", data_sparse.shape)

            #
            else:
                data_out = np.zeros(data_sparse.shape)
                for k in trange(data_sparse.shape[1]):
                    for p in range(data_sparse.shape[2]):
                        data_out[:, k, p] = np.convolve(data_sparse[:, k, p], kernel, mode='same')

            print("done window smoothing...")

        std = np.std(self.data_filtered, axis=0)

        print ("self.data_filtered: ", self.data_filtered.shape)

        return std

    def plot_std_map(self, std):
        #
        temp = std.copy()
        print("staring computing std...")
        print("done computing std...")
        #
        idx = np.where(temp < self.vmin)
        temp[idx] = 0
        idx = np.where(temp > self.vmax)
        temp[idx] = self.vmax

        #
        plt.figure()
        plt.imshow(temp,
                   )
        plt.show()

        return temp

    def area_inside_convex_hull(self, pts):
        lines = np.hstack([pts, np.roll(pts, -1, axis=0)])
        area = 0.5 * abs(sum(x1 * y2 - x2 * y1 for x1, y1, x2, y2 in lines))
        return area

    def binarize_data(self, img, thresh):

        # thresh = .15
        idx1 = np.where(img > thresh)
        idx2 = np.where(img <= thresh)
        img[idx1] = 1
        img[idx2] = 0

        return img

    #
    def find_roi_boundaries(self, data):

        #
        image = data.copy()

        for k in trange(self.n_smooth_steps, desc='gaussian filtering data'):
            image = scipy.ndimage.gaussian_filter(image,
                                                  self.sigma,
                                                  self.order)

        image = image.astype('int32')

        #
        image = self.binarize_data(image, self.vmin)

        #
        image = image.astype('int32')

        # run watershed segmentation
        distance = ndi.distance_transform_edt(image)
        coords = peak_local_max(distance,
                                footprint=np.ones((1, 1)),
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
        min_size = self.min_size_roi
        max_size = self.max_size_roi
        roi_centres = []
        footprints = []
        for k in tqdm(np.unique(labels), desc='looping over cells'):
            idx = np.where(labels == k)

            if idx[0].shape[0] < min_size or idx[0].shape[0] > max_size:
                labels[idx] = np.nan
            else:

                roi_centres.append([np.median(idx[0]),
                                    np.median(idx[1])])
                footprints.append(idx)

        self.rois = np.vstack(roi_centres)
        self.footprints = footprints

    #
    def compute_contour_map(self, std_map, cell_ids):
        ''' Compute contours and save them to disk also

		'''

        #
        contour_array = []
        for cell_id in cell_ids:
            temp = np.zeros(std_map.shape, dtype='uint8')
            temp[self.footprints[cell_id]] = 1

            #
            contour, _ = cv2.findContours(temp,
                                          cv2.RETR_TREE,
                                          cv2.CHAIN_APPROX_SIMPLE)
            contour = contour[0].squeeze()
            try:
                contour = np.vstack((contour, contour[0]))
            except:
                print ("contour is broken, skipping: ")
                contour = []

            #
            contour_array.append(contour)

        return contour_array


    #
    def show_contour_map2(self, std_map, footprints, cell_ids, fig=False):

        #
        if fig is True:
            plt.figure()

        #
        plt.imshow(std_map,
                   vmin=self.vmin * 0.7,
                   vmax=self.vmax * 1.3)



        # add cell contours
        clrs=['white']
        for p in range(len(footprints)):
            temp = np.zeros(std_map.shape)
            temp[footprints[p]] = 1
            temp = temp.astype('uint8')
            contour, _ = cv2.findContours(temp,
                                          cv2.RETR_TREE,
                                          cv2.CHAIN_APPROX_SIMPLE)
            contour = contour[0].squeeze()
            try:
                contour = np.vstack((contour, contour[0]))
            except:
                continue

            #
            for k in range(len(contour) - 1):
                plt.plot([contour[k][0], contour[k + 1][0]],
                         [contour[k][1], contour[k + 1][1]],
                         c='white')
            #
            z = np.vstack(footprints[p]).T
            plt.text(np.median(z[:, 1])-5, np.median(z[:, 0])+5, str(p), c='white',fontsize=12)


        # add cell contours
        clrs=['blue','red','green','pink']
        print ("cell ids: ", cell_ids)
        for ctr,p in enumerate(cell_ids):
            color = clrs[ctr//2]
            temp = np.zeros(std_map.shape)
            temp[footprints[p]] = 1
            temp = temp.astype('uint8')
            contour, _ = cv2.findContours(temp,
                                          cv2.RETR_TREE,
                                          cv2.CHAIN_APPROX_SIMPLE)
            contour = contour[0].squeeze()
            contour = np.vstack((contour, contour[0]))

            #
            for k in range(len(contour) - 1):
                plt.plot([contour[k][0], contour[k + 1][0]],
                         [contour[k][1], contour[k + 1][1]],
                         linewidth=2,
                         c=color)
            #
            z = np.vstack(footprints[p]).T
            #plt.text(np.median(z[:, 1]), np.median(z[:, 0]), str(p), c=color, fontsize=15)

        plt.show()
    #

    def show_contour_map3(self, std_map, footprints, cell_ids, fig=False):

        #
        if fig is True:
            plt.figure()

        #
        plt.imshow(std_map,
                   vmin=self.vmin * 0.7,
                   vmax=self.vmax * 1.3)



        # add cell contours
        clrs=['white']
        ctr=0
        for contour in self.contours_all_cells[:self.max_n_cells]:
            #
            #print (contour)
            contour = contour.squeeze().item()
            for k in range(len(contour) - 1):
                plt.plot([contour[k][0], contour[k + 1][0]],
                         [contour[k][1], contour[k + 1][1]],
                         c='white')

            #


            z = np.mean(contour,axis=0)
            try:
                plt.text(z[0]-5, z[1]+5, str(ctr), c='white',fontsize=12)
            except:
                pass
            ctr+=1

        # add cell contours
        clrs=['blue','red','green','pink']
        print ("cell ids: ", cell_ids)
        contours = np.vstack((self.ensemble1_contours, self.ensemble2_contours)).squeeze()
        for ctr,contour in enumerate(contours):
            color = clrs[ctr//2]

            #
            for k in range(len(contour) - 1):
                plt.plot([contour[k][0], contour[k + 1][0]],
                         [contour[k][1], contour[k + 1][1]],
                         linewidth=2,
                         c=color)
            #
            #z = np.vstack(footprints[p]).T
            #plt.text(np.median(z[:, 1]), np.median(z[:, 0]), str(p), c=color, fontsize=15)

        plt.show()
    #
    #
    def show_contour_map(self, std_map, footprints, cell_ids, fig=False):

        #
        if fig is True:
            plt.figure()

        #
        plt.imshow(std_map,
                   vmin=self.vmin * 0.7,
                   vmax=self.vmax * 1.3)



        # add cell contours
        clrs=['white']
        for p in range(len(footprints)):
            temp = np.zeros(std_map.shape)
            #print ("footrpints: ", footprints[p])
            temp[footprints[p]] = 1
            temp = temp.astype('uint8')
            contour, _ = cv2.findContours(temp,
                                          cv2.RETR_TREE,
                                          cv2.CHAIN_APPROX_SIMPLE)
            contour = contour[0].squeeze()
            #print ("contour: ", contour.shape)
            try:
                contour = np.vstack((contour, contour[0]))
            except:
                continue

            #
            for k in range(len(contour) - 1):
                plt.plot([contour[k][0], contour[k + 1][0]],
                         [contour[k][1], contour[k + 1][1]],
                         c='white')
            #
            z = np.vstack(footprints[p]).T
            plt.text(np.median(z[:, 1]), np.median(z[:, 0]), str(p), c='white', fontsize=15)

        #
        # # add cell contours
        # clrs=['blue','red','green','pink']
        # print ("cell ids: ", cell_ids)
        # for p in cell_ids:
        #     color = clrs[p//2]
        #     temp = np.zeros(std_map.shape)
        #     temp[footprints[p]] = 1
        #     temp = temp.astype('uint8')
        #     contour, _ = cv2.findContours(temp,
        #                                   cv2.RETR_TREE,
        #                                   cv2.CHAIN_APPROX_SIMPLE)
        #     contour = contour[0].squeeze()
        #     contour = np.vstack((contour, contour[0]))
        #
        #     #
        #     for k in range(len(contour) - 1):
        #         plt.plot([contour[k][0], contour[k + 1][0]],
        #                  [contour[k][1], contour[k + 1][1]],
        #                  c=color)
        #     #
        #     z = np.vstack(footprints[p]).T
        #     plt.text(np.median(z[:, 1]), np.median(z[:, 0]), str(p), c='red')

        plt.show()
    #
    def compute_roi_traces_f0_and_reorder_cells(self,
                                   order_type):

        #
        cell_ids = np.arange(len(self.footprints))

        #
        data = np.memmap(self.fname, dtype='uint16', mode='r')
        data = data.reshape(-1, 512, 512)

        #####################################################
        ################ COMPUTE ROI TRACES #################
        #####################################################
        #
        roi_traces = []
        for k in range(len(cell_ids)):
            roi_traces.append([])

        # loop over each frame
        for p in trange(0, data.shape[0], self.subsample,
                        desc='computing roi traces for SNR indexing'):

            # grab frame
            frame = data[p]

            # loop over ROIS
            ctr = 0
            for k in cell_ids:
                # grab roi
                #print (self.footprints[k])
                temp = frame[self.footprints[k]].copy()

                # normalize by surface area so that cells don't look way different because of footprint size
                if True:
                    temp = temp / self.footprints[k][0].shape[0]

                # add pixel values inside roi
                temp = np.nansum(temp)

                # save
                roi_traces[ctr].append(temp)
                ctr += 1
        #
        self.roi_traces = np.array(roi_traces)

        ###########################################################
        ################### COMPUTE F0 AND SNR ####################
        ###########################################################
        # compute the baseline f0 of the cells in order to be able to offset it in the BMI
        # TODO: this is important; it functions as a rough DFF method
        #    TODO: we may wish to implement a more complex version of this
        self.roi_f0s = np.zeros(self.roi_traces.shape[0], dtype=np.float32)
        self.roi_snrs = np.zeros(self.roi_traces.shape[0], dtype=np.float32)
        for k in cell_ids:

            #
            #f0 = np.median(self.roi_traces[k])
            f0 = get_mode(self.roi_traces[k])

            #
            self.roi_f0s[k] = f0

            #
            self.roi_snrs[k] = np.max((self.roi_traces[k]-f0)/f0)

        ###########################################################
        ################# REORDER CELLS BY SNR  ###################
        ###########################################################
        if order_type=='f0':
            idx = np.argsort(self.roi_f0s)[::-1]

        elif order_type=='snr':
            idx = np.argsort(self.roi_snrs)[::-1]
        else:
            print (" ERROR - type not known")

        #
        self.roi_traces = self.roi_traces[idx]

        #
        self.footprints_temp = []
        self.rois_f0s_temp = []
        for k in range(idx.shape[0]):
            self.footprints_temp.append(self.footprints[idx[k]])
            self.rois_f0s_temp.append(self.roi_f0s[idx[k]])

        self.footprints = self.footprints_temp
        self.roi_f0s = self.rois_f0s_temp


    #
    #
    #
    #
    # def compute_roi_traces_f0_and_reorder_cells(self,
    #                                order_type):
    #
    #     #
    #     cell_ids = np.arange(len(self.footprints))
    #
    #     #
    #     data = np.memmap(self.fname, dtype='uint16', mode='r')
    #     data = data.reshape(-1, 512, 512)
    #
    #     #####################################################
    #     ################ COMPUTE ROI TRACES #################
    #     #####################################################
    #     #
    #     roi_traces = []
    #     for k in range(len(cell_ids)):
    #         roi_traces.append([])
    #
    #     # loop over each frame
    #     for p in trange(0, data.shape[0], self.subsample,
    #                     desc='computing roi traces for SNR indexing'):
    #
    #         # grab frame
    #         frame = data[p]
    #
    #         # loop over ROIS
    #         ctr = 0
    #         for k in cell_ids:
    #             # grab roi
    #             temp = frame[self.footprints[k]]
    #
    #             # normalize by surface area so that cells don't look way different because of footprint size
    #             if True:
    #                 temp = temp / self.footprints[k][0].shape[0]
    #
    #             # add pixel values inside roi
    #             temp = np.nansum(temp)
    #
    #             # save
    #             roi_traces[ctr].append(temp)
    #             ctr += 1
    #     #
    #     self.roi_traces = np.array(roi_traces)
    #     plt.figure()
    #     plt.plot(self.roi_traces[20])
    #     plt.show()
    #     ###########################################################
    #     ################### COMPUTE F0 AND SNR ####################
    #     ###########################################################
    #     # compute the baseline f0 of the cells in order to be able to offset it in the BMI
    #     # TODO: this is important; it functions as a rough DFF method
    #     #    TODO: we may wish to implement a more complex version of this
    #     self.roi_f0s = np.zeros(self.roi_traces.shape[0], dtype=np.float32)
    #     self.roi_snrs = np.zeros(self.roi_traces.shape[0], dtype=np.float32)
    #     for k in cell_ids:
    #
    #         #
    #         f0 = np.median(self.roi_traces[k])
    #
    #         #
    #         self.roi_f0s[k] = f0
    #
    #         #
    #         self.roi_snrs[k] = np.max(self.roi_traces[k]/f0)
    #
    #     ###########################################################
    #     ################# REORDER CELLS BY SNR  ###################
    #     ###########################################################
    #     if order_type=='f0':
    #         idx = np.argsort(self.roi_f0s)[::-1]
    #
    #     elif order_type=='snr':
    #         idx = np.argsort(self.roi_snrs)[::-1]
    #     else:
    #         print (" ERROR - type not known")
    #
    #     #
    #     self.roi_traces = self.roi_traces[idx]
    #
    #     #
    #     self.footprints_temp = []
    #     for k in range(idx.shape[0]):
    #         self.footprints_temp.append(self.footprints[idx[k]])
    #
    #     self.footprints = self.footprints_temp


    def compute_traces_ensembles(self, std_map):
        """ Same as below but visualize every single frame
        """

        self.trace_subsample = 1  # Subsample the time series to go faster;
        self.scale = 3

        data = np.memmap(self.fname, dtype='uint16', mode='r')
        data = data.reshape(-1, 512, 512)

        ########################################################
        ########################################################
        ########################################################
        # loop over each frame
        self.ensemble1_traces = []
        for k in range(len(self.ensemble1)):
            self.ensemble1_traces.append([])

        #
        self.ensemble2_traces = []
        for k in range(len(self.ensemble2)):
            self.ensemble2_traces.append([])

        #
        for p in trange(0, data.shape[0], self.trace_subsample):

            # grab frame
            frame = data[p]

            # loop over ensemble1 traces
            ctr = 0
            for k in self.ensemble1:
                # grab roi
                temp = frame[self.footprints[k]]

                # normalize by surface area so that cells don't look way different because of footprint size
                if True:
                    temp = temp / self.footprints[k][0].shape[0]

                # add pixel values inside roi
                temp = np.nansum(temp)

                # save
                self.ensemble1_traces[ctr].append(temp)
                ctr += 1

            # loop over ensemble2 traces
            ctr = 0
            for k in self.ensemble2:
                # grab roi
                temp = frame[self.footprints[k]]

                # normalize by surface area so that cells don't look way different because of footprint size
                if True:
                    temp = temp / self.footprints[k][0].shape[0]

                # add pixel values inside roi
                temp = np.nansum(temp)

                # save
                self.ensemble2_traces[ctr].append(temp)
                ctr += 1

        ###############################################
        plt.figure()
        ax = plt.subplot(121)
        ax.tick_params(axis='both', which='both', labelsize=20)
        plt.ylabel("Neuron ID ", fontsize=20)

        # plot ensemble 1 cells
        ctr2=0
        for ctr,k in enumerate(self.ensemble1):
            temp = self.ensemble1_traces[ctr]

            # normalize by the correct cell id, not the one computed above
            temp = (temp - self.roi_f0s[self.ensemble1[ctr]])/self.roi_f0s[self.ensemble1[ctr]]

            # we update the selected traces time dynamics
            self.ensemble1_traces[ctr] = temp

            #
            t = np.linspace(0, data.shape[0], temp.shape[0]) / 30.
            plt.plot(t, self.ensemble1_traces[ctr] + ctr2 * self.scale,
                     c='blue')

            ctr2 += 1

        # plot ensemble 2 cells
        for ctr,k in enumerate(self.ensemble2):
            temp = self.ensemble2_traces[ctr]

            # normalize by the correct cell id, not the one computed above
            temp = (temp - self.roi_f0s[self.ensemble2[ctr]]) / self.roi_f0s[self.ensemble2[ctr]]

            # we update the selected traces time dynamics
            self.ensemble2_traces[ctr] = temp

            #
            t = np.linspace(0, data.shape[0], temp.shape[0]) / 30.
            plt.plot(t, self.ensemble2_traces[ctr] + ctr2 * self.scale,
                     c='red')

            ctr2 += 1

        #
        cell_ids = np.hstack((self.ensemble1, self.ensemble2))

        #
        labels = cell_ids
        labels_old = np.arange(0, ctr2 * self.scale, self.scale)

        #
        plt.yticks(labels_old, labels, fontsize=10)
        plt.xlabel("Time (sec)", fontsize=20)

        #
        plt.subplot(122)
        new_plot = False
        self.show_contour_map2(std_map,
                              self.footprints,
                              cell_ids,
                              new_plot)

        plt.show()




    def compute_traces_ensembles_new_day(self, std_map):
        """ Same as below but visualize every single frame
        """

        self.trace_subsample = 1  # Subsample the time series to go faster;
        self.scale = 3

        #
        data = np.memmap(self.fname, dtype='uint16', mode='r')
        data = data.reshape(-1, 512, 512)

        ########################################################
        ########################################################
        ########################################################
        # loop over each frame
        self.ensemble1_traces = []
        for k in range(len(self.ensemble1)):
            self.ensemble1_traces.append([])

        #
        self.ensemble2_traces = []
        for k in range(len(self.ensemble2)):
            self.ensemble2_traces.append([])

        #
        for p in trange(0, data.shape[0], self.trace_subsample):

            # grab frame
            frame = data[p]

            # loop over ensemble1 traces
            ctr = 0
            for k in self.ensemble1:
                # grab roi
                #print ("self.footprints: ", self.footprints[k][0])
                #temp = frame[self.footprints[k]]
                temp = frame[self.footprints[k][0],
                             self.footprints[k][1]]

                # normalize by surface area so that cells don't look way different because of footprint size
                if True:
                    temp = temp / self.footprints[k][0].shape[0]

                # add pixel values inside roi
                temp = np.nansum(temp)

                # save
                self.ensemble1_traces[ctr].append(temp)
                ctr += 1

            # loop over ensemble2 traces
            ctr = 0
            for k in self.ensemble2:
                # grab roi
                temp = frame[self.footprints[k][0],
                             self.footprints[k][1]]
                # normalize by surface area so that cells don't look way different because of footprint size
                if True:
                    temp = temp / self.footprints[k][0].shape[0]

                # add pixel values inside roi
                temp = np.nansum(temp)

                # save
                self.ensemble2_traces[ctr].append(temp)
                ctr += 1

        ###############################################
        plt.figure()
        ax = plt.subplot(121)
        ax.tick_params(axis='both', which='both', labelsize=20)
        plt.ylabel("Neuron ID ", fontsize=20)

        # plot ensemble 1 cells
        ctr2=0
        for ctr,k in enumerate(self.ensemble1):
            temp = self.ensemble1_traces[ctr]

            # normalize by the correct cell id, not the one computed above
            temp = (temp - self.roi_f0s[self.ensemble1[ctr]])/self.roi_f0s[self.ensemble1[ctr]]

            # we update the selected traces time dynamics
            self.ensemble1_traces[ctr] = temp

            #
            t = np.linspace(0, data.shape[0], temp.shape[0]) / 30.
            plt.plot(t, self.ensemble1_traces[ctr] + ctr2 * self.scale,
                     c='blue')

            ctr2 += 1

        # plot ensemble 2 cells
        for ctr,k in enumerate(self.ensemble2):
            temp = self.ensemble2_traces[ctr]

            # normalize by the correct cell id, not the one computed above
            temp = (temp - self.roi_f0s[self.ensemble2[ctr]]) / self.roi_f0s[self.ensemble2[ctr]]

            # we update the selected traces time dynamics
            self.ensemble2_traces[ctr] = temp

            #
            t = np.linspace(0, data.shape[0], temp.shape[0]) / 30.
            plt.plot(t, self.ensemble2_traces[ctr] + ctr2 * self.scale,
                     c='red')

            ctr2 += 1

        #
        cell_ids = np.hstack((self.ensemble1, self.ensemble2))

        #
        labels = cell_ids
        labels_old = np.arange(0, ctr2 * self.scale, self.scale)

        #
        plt.yticks(labels_old, labels, fontsize=10)
        plt.xlabel("Time (sec)", fontsize=20)

        #
        plt.subplot(122)
        new_plot = False
        self.show_contour_map3(std_map,
                              self.footprints,
                              cell_ids,
                              new_plot)

        plt.show()

    def compute_traces_ensembles2(self, std_map):
        """ Same as below but visualize every single frame
        """

        self.trace_subsample = 1  # Subsample the time series to go faster;
        self.scale = 3

        data = np.memmap(self.fname, dtype='uint16', mode='r')
        data = data.reshape(-1, 512, 512)

        ########################################################
        ########################################################
        ########################################################
        # loop over each frame
        self.ensemble1_traces = []
        for k in range(len(self.ensemble1)):
            self.ensemble1_traces.append([])

        #
        self.ensemble2_traces = []
        for k in range(len(self.ensemble2)):
            self.ensemble2_traces.append([])

        #
        for p in trange(0, data.shape[0], self.trace_subsample):

            # grab frame
            frame = data[p]

            # loop over ensemble1 traces
            ctr = 0
            for k in self.ensemble1:
                # grab roi
                temp = frame[self.footprints[k]]

                # normalize by surface area so that cells don't look way different because of footprint size
                if True:
                    temp = temp / self.footprints[k][0].shape[0]

                # add pixel values inside roi
                temp = np.nansum(temp)

                # save
                self.ensemble1_traces[ctr].append(temp)
                ctr += 1

            # loop over ensemble2 traces
            ctr = 0
            for k in self.ensemble2:
                # grab roi
                temp = frame[self.footprints[k]]

                # normalize by surface area so that cells don't look way different because of footprint size
                if True:
                    temp = temp / self.footprints[k][0].shape[0]

                # add pixel values inside roi
                temp = np.nansum(temp)

                # save
                self.ensemble2_traces[ctr].append(temp)
                ctr += 1

        ###############################################
        plt.figure()
        ax = plt.subplot(121)
        ax.tick_params(axis='both', which='both', labelsize=20)
        plt.ylabel("Neuron ID ", fontsize=20)

        # plot ensemble 1 cells
        ctr2 = 0
        for ctr, k in enumerate(self.ensemble1):
            temp = self.ensemble1_traces[ctr]

            # normalize by the correct cell id, not the one computed above
            temp = (temp - self.roi_f0s[self.ensemble1[ctr]]) / self.roi_f0s[self.ensemble1[ctr]]

            # we update the selected traces time dynamics
            self.ensemble1_traces[ctr] = temp

            #
            t = np.linspace(0, data.shape[0], temp.shape[0]) / 30.
            plt.plot(t, self.ensemble1_traces[ctr] + ctr2 * self.scale,
                     c='blue')
            plt.plot(t, t*0 + np.median(self.ensemble1_traces[ctr])+ ctr2 * self.scale,'--', c='black')

            ctr2 += 1

        # plot ensemble 2 cells
        for ctr, k in enumerate(self.ensemble2):
            temp = self.ensemble2_traces[ctr]

            # normalize by the correct cell id, not the one computed above
            temp = (temp - self.roi_f0s[self.ensemble2[ctr]]) / self.roi_f0s[self.ensemble2[ctr]]

            # we update the selected traces time dynamics
            self.ensemble2_traces[ctr] = temp

            #
            t = np.linspace(0, data.shape[0], temp.shape[0]) / 30.
            plt.plot(t, self.ensemble2_traces[ctr] + ctr2 * self.scale,
                     c='red')
            plt.plot(t, t*0 + np.median(self.ensemble2_traces[ctr])+ ctr2 * self.scale,'--', c='black')

            ctr2 += 1
        plt.xlim(t[0],t[-1])
        #
        cell_ids = np.hstack((self.ensemble1, self.ensemble2))

        #
        labels = cell_ids
        labels_old = np.arange(0, ctr2 * self.scale, self.scale)

        #
        # print (labels_old)
        # print (labels)
        plt.yticks(labels_old, labels, fontsize=10)
        plt.xlabel("Time (sec)", fontsize=20)

        #
        plt.subplot(122)
        new_plot = False
        self.show_contour_map2(std_map,
                              self.footprints,
                              cell_ids,
                              new_plot)

        plt.show()

        #
    def compute_traces2(self, std_map, cell_ids=None, fig=None):
        """ Same as below but visualize every single frame
        """

        if cell_ids is None:
            cell_ids = np.arange(len(self.footprints))
        print("plotting cells: ", cell_ids)

        data = np.memmap(self.fname, dtype='uint16', mode='r')
        data = data.reshape(-1, 512, 512)

        ########################################################
        #
        roi_traces = []
        for k in range(len(cell_ids)):
            roi_traces.append([])

        # loop over each frame
        for p in trange(0, data.shape[0], self.trace_subsample):

            # grab frame
            frame = data[p]

            # loop over ROIS
            ctr = 0
            for k in cell_ids:
                # grab roi
                temp = frame[self.footprints[k]]

                # normalize by surface area so that cells don't look way different because of footprint size
                if True:
                    temp = temp / self.footprints[k][0].shape[0]

                # add pixel values inside roi
                temp = np.nansum(temp)

                # save
                roi_traces[ctr].append(temp)
                ctr += 1
        #
        roi_traces = np.array(roi_traces)

        #
        plt.figure()
        ax = plt.subplot(121)
        ax.tick_params(axis='both', which='both', labelsize=20)
        plt.ylabel("Neuron ID ", fontsize=20)
        self.roi_traces_fullres_dff = []
        ctr=0
        for k in range(len(roi_traces)):
            temp = roi_traces[k]
            temp = (temp - self.roi_f0s[k])/self.roi_f0s[k]

            # we update the selected traces time dynamics
            self.roi_traces_fullres_dff.append(temp)

            #
            t = np.linspace(0, data.shape[0], temp.shape[0]) / 30.
            plt.plot(t, self.roi_traces_fullres_dff[k] + ctr * self.scale)

            ctr += 1
        #
        labels = cell_ids
        labels_old = np.arange(0, ctr * self.scale, self.scale)

        #
        plt.yticks(labels_old, labels, fontsize=10)
        plt.xlabel("Time (sec)", fontsize=20)

        #
        plt.subplot(122)
        new_plot = False
        self.show_contour_map(std_map,
                              self.footprints,
                              cell_ids, new_plot)

        plt.show()

    #
    def visualize_traces_snr_order(self, std_map, cell_ids=None):
        """ Same as below but visualize every single frame
        """

        #
        if cell_ids==None:
            cell_ids = np.arange(len(self.footprints))

        #
        data = np.memmap(self.fname, dtype='uint16', mode='r')
        data = data.reshape(-1, 512, 512)
        print("memmap : ", data.shape)

        ###########################################################
        ################## PLOT CELLS IN TYPE ORDER ################
        ###########################################################
        plt.figure()

        #
        #self.roi_f0s = np.zeros(self.roi_traces.shape[0], dtype=np.float32)

        j = 0
        ctr = 0
        cell_ids = []
        for k in range(self.roi_traces.shape[0]):

            ax=plt.subplot(1,2,j+1)
            ax.tick_params(axis='both', which='both', labelsize=20)
            plt.ylabel("Neuron ID ", fontsize=20)

            temp = self.roi_traces[k].copy()
            #if self.roi_f0s[k]==0:
            #    print ("FOUND F0 = 0", k, self.roi_f0s[k])
            temp = (temp - self.roi_f0s[k])/self.roi_f0s[k]

            if False:
                temp = (temp - np.min(temp))/(np.max(temp)-np.min(temp))

            # each cell might have different time signatures in case some have higher temporal resolution
            t = np.linspace(0, data.shape[0], temp.shape[0]) / 30.

            plt.plot(t, temp-np.median(temp) + ctr * self.scale)

            cell_ids.append(k)

            ctr += 1
            if ctr > self.max_n_cells:
                break




        labels = cell_ids
        labels_old = np.arange(0, ctr * self.scale, self.scale)

        #
        plt.yticks(labels_old, labels, fontsize=10)
        plt.xlabel("Time (sec)", fontsize=20)

        ###########################################################
        ################### PLOT IMAGE OF [CA] ####################
        ###########################################################
        plt.subplot(1,2,2)
        new_plot = False
        self.show_contour_map(std_map,
                              self.footprints[:self.max_n_cells],
                              cell_ids,
                              new_plot)

        plt.show()

    def visualize_traces_snr_order2(self, std_map):
        """ Same as below but visualize every single frame
        """

         #
        data = np.memmap(self.fname, dtype='uint16', mode='r')
        data = data.reshape(-1, 512, 512)
        print("memmap : ", data.shape)

        ###########################################################
        ################## PLOT CELLS IN TYPE ORDER ################
        ###########################################################
        plt.figure()

        #
        # self.roi_f0s = np.zeros(self.roi_traces.shape[0], dtype=np.float32)

        j = 0
        ctr = 0
        cell_ids = []
        for k in range(self.roi_traces.shape[0]):

            ax = plt.subplot(1, 2, 1)
            ax.tick_params(axis='both', which='both', labelsize=20)
            plt.ylabel("Neuron ID ", fontsize=20)

            temp = self.roi_traces[k].copy()
            # if self.roi_f0s[k]==0:
            #    print ("FOUND F0 = 0", k, self.roi_f0s[k])
            temp = (temp - self.roi_f0s[k]) / self.roi_f0s[k]

            if False:
                temp = (temp - np.min(temp)) / (np.max(temp) - np.min(temp))

            # each cell might have different time signatures in case some have higher temporal resolution
            t = np.linspace(0, data.shape[0], temp.shape[0]) / 30.

            plt.plot(t, temp - np.median(temp) + ctr * self.scale)

            cell_ids.append(k)

            ctr += 1

            if ctr>self.max_n_cells:
                break

        labels = cell_ids
        labels_old = np.arange(0, ctr * self.scale, self.scale)

        #
        plt.yticks(labels_old, labels, fontsize=10)
        plt.xlabel("Time (sec)", fontsize=20)

        ###########################################################
        ################### PLOT IMAGE OF [CA] ####################
        ###########################################################
        plt.subplot(1, 2, 2)
        new_plot = False

        self.show_contour_map3(std_map,
                              self.footprints[:self.max_n_cells],
                              self.ensemble_ids,
                              new_plot)

        plt.show()

    #

    #
    def show_traces_ids(self, ids):

        #
        plt.figure()

        #
        plt.title("Cell Ids: " + str(ids))
        #
        t = np.arange(0, self.roi_traces[0].shape[0], 1) / 30. * self.trace_subsample
        ctr = 0
        for k in ids:
            temp = self.roi_traces[k]
            #f0_local = np.median(self.roi_traces[k])
            temp = (temp - self.roi_f0s[k])/self.roi_f0s[k]
            #temp = (temp - f0_local)/f0_local
            plt.plot(t, temp + ctr * self.scale)

            ctr += 1

        labels = ids
        labels_old = np.arange(0, ctr * self.scale, self.scale)

        #
        plt.yticks(labels_old, labels)
        plt.xlabel("Time (sec)")

        plt.show()

    # #
    # def find_reward_thresholds_absolute(self, normalize_peaks=True):
    #     '''  Computes the aboslute |E1-E2|
	# 	     and rewards anytime the ensembel goes above this value
	# 	     Note that self.roi_traces contains only the 4 neurons from the ensembes selected
	# 	     now
	# 	     - TODO: change this for the high and high_low functions also
    #
	# 	'''
    #
    #     # TODO: refactor this part and send it to the BMI session code
    #
    #     # run smoothing on each ensemble
    #     if self.smooth_diff_function_flag:
    #
    #         # ensemble #1
    #         for p in range(2):
    #             smooth = np.zeros(self.roi_traces[p].shape)
    #             for k in trange(self.rois_smooth_window, self.roi_traces[p].shape[0], 1):
    #                 smooth[k] = self.smooth_ca_time_series(self.roi_traces[p][k - self.rois_smooth_window:k])
    #             #
    #             self.roi_traces[p] = smooth
    #
    #         # ensemble #2
    #         for p in range(2, 4, 1):
    #             smooth = np.zeros(self.roi_traces[p].shape)
    #             for k in trange(self.rois_smooth_window, self.roi_traces[p].shape[0], 1):
    #                 smooth[k] = self.smooth_ca_time_series(self.roi_traces[p][k - self.rois_smooth_window:k])
    #             #
    #             self.roi_traces[p] = smooth
    #
    #     #
    #     self.roi_f0s = []
    #     self.dff0 = []
    #     for k in range(len(self.roi_traces)):
    #         f0, dff0 = self.compute_dff0(self.roi_traces[k])
    #         self.roi_f0s.append(f0)
    #         self.dff0.append(dff0)
    #
    #     # compute ensembles using the smooth + baseline removed values
    #     E1 = self.dff0[0] - self.dff0[1]
    #     E2 = self.dff0[2] - self.dff0[3]
    #
    #     # initialize the max and min values
    #     max_E1 = np.max(E1)
    #     max_E2 = np.max(E2)
    #
    #     print(
    #         "TODO: Normalize the peaks of the 2 Ensembles so the mice don't learn to use one esnemble against the other!!!!")
    #     low = np.nan
    #     high = min(max_E1, max_E2) * 3
    #
    #     print("low, high: ", low, high)
    #     # difference between ensemble
    #     diff = np.abs(E1 - E2)
    #
    #     #
    #     self.n_sec_recording = int(diff.shape[0] / self.sample_rate)
    #     self.n_rewards_random = self.n_sec_recording // self.sample_rate
    #     self.n_rewards_default = int(self.n_rewards_random * 0.3)
    #     print("nsec recording: ", self.n_sec_recording,
    #           "max # of random rewards (i.e. every 30sec) ", self.n_rewards_random,
    #           "# of rewards for 30% of the time: ", self.n_rewards_default)
    #
    #     # loop over time series decreasing the rewards until we hit the random #
    #     n_rewards = 0
    #     stepper = 0.95
    #     while n_rewards < self.n_rewards_default:
    #
    #         # run inside while loop for eveyr setting of low and high until we hit
    #         #   exact number of random rewards
    #         k = 0
    #         n_rewards = 0
    #         reward_times = []
    #         while k < diff.shape[0]:
    #
    #             temp_diff = diff[k]
    #
    #             if temp_diff >= high:
    #                 # high reward state reached
    #                 n_rewards += 1
    #                 reward_times.append([k, 1])
    #                 k += int(self.post_reward_lockout * self.sample_rate)
    #             else:
    #                 k += 1
    #
    #         # print ("Reard times: ", reward_times)
    #         # check exit condition otherwise decrase thresholds
    #         # if len(reward_times) > 1:
    #         # 	rewarded_times = np.vstack(reward_times)
    #         #	high *= stepper
    #         # else:
    #         high *= stepper
    #
    #     print("updated rwards #: ", n_rewards, low, high)
    #
    #     self.reward_times = np.vstack(reward_times)
    #
    #     self.low = np.nan
    #     self.high = high
    #     self.E1 = E1
    #     self.E2 = E2
    #     self.diff = diff


    def find_reward_thresholds_high_realtime(self):

        # initialize the max and min values
        #
        n_sec_recording = int(self.ensemble1_traces[0].shape[0] / self.sample_rate)
        n_rewards_random = n_sec_recording // self.sample_rate
        print("nsec recording: ", n_sec_recording,
              "max # of random rewards (i.e. every 30sec) ", n_rewards_random)
        n_rewards_random = int(n_rewards_random * self.reward_rate)
        print(" @30% reward: ", n_rewards_random)
        self.n_rewards_default = n_rewards_random



        # take a stab at high value:
        high = np.max(self.ensemble1_traces[0])
        print (" high guess: ", high)
        #high = 12
        #
        # loop over time series decreasing the rewards until we hit the random #
        stepper = 0.95
        self.fps = 30
        n_rewards = 0
        from tqdm.notebook import tqdm
        for qq in range(200):

            # run inside while loop for eveyr setting of low and high until we hit
            #   exact number of random rewards
            n_rewards = 0
            last_reward = 0
            reward_times = []

            E1_1= np.zeros(self.ensemble1_traces[0].shape[0])
            E1_2= np.zeros(self.ensemble1_traces[1].shape[0])
            E2_1= np.zeros(self.ensemble2_traces[0].shape[0])
            E2_2= np.zeros(self.ensemble2_traces[1].shape[0])

            #
             #
            #p_bar = tqdm(np.arange(self.ensemble1_traces[0].shape[0]), desc='loop',position=0, leave=True)
            counter=-1

            #for k in tqdm(np.arange(5,self.ensemble1_traces[0].shape[0],1), desc='loop'):
            for k in range(5, self.ensemble1_traces[0].shape[0], 1):

                #
                if True:
                    self.E1_1 = smooth_ca_time_series4(self.ensemble1_traces[0][k - self.rois_smooth_window:k])
                    self.E1_2 = smooth_ca_time_series4(self.ensemble1_traces[1][k - self.rois_smooth_window:k])
                    self.E2_1 = smooth_ca_time_series4(self.ensemble2_traces[0][k - self.rois_smooth_window:k])
                    self.E2_2 = smooth_ca_time_series4(self.ensemble2_traces[1][k - self.rois_smooth_window:k])

                else:
                    self.E1_1 = self.ensemble1_traces[0][k]
                    self.E1_2 = self.ensemble1_traces[1][k]
                    self.E2_1 = self.ensemble2_traces[0][k]
                    self.E2_2 = self.ensemble2_traces[1][k]

                # remove F0 baseline
                # TODO: this is not quite right as we skip over values all the time here
                if False:
                    if k % (self.fps*120)==0:
                        for p in range(2):
                            E1_f0s[p] = get_mode(self.ensemble1_traces[p][k-self.fps*120:k])
                            E2_f0s[p] = get_mode(self.ensemble2_traces[p][k-self.fps*120:k])

                #
                if False:
                    self.E1_1 = (self.E1_1) / E1_f0s[0]
                    self.E1_2 = (self.E1_2) / E1_f0s[1]
                    self.E2_1 = (self.E2_1) / E2_f0s[0]
                    self.E2_2 = (self.E2_2) / E2_f0s[1]

                #print (self.E1_1,self.E1_2,self.E2_1,self.E2_2)

                # save data arrays for later visualization
                E1_1[k] = self.E1_1
                E1_2[k] = self.E1_2
                E2_1[k] = self.E2_1
                E2_2[k] = self.E2_2

                #
                self.E1 = self.E1_1+self.E1_2
                self.E2 = self.E2_1+self.E2_2

                temp_diff = self.E1-self.E2
                #print ("k: ", k, " tempdiff: ", temp_diff)

                #
                counter-=1
                if counter>0:
                    continue

                if temp_diff >= high:
                    # high reward state reached
                    n_rewards += 1
                    reward_times.append([k, 1])
                    last_reward = k

                    # lock out rewards for some time;
                    #k += int(self.post_reward_lockout * self.sample_rate)
                    counter = int(self.post_reward_lockout * self.sample_rate)

                elif (k-last_reward)>= int(self.trial_time * self.sample_rate):

                    # lock out rewards for some time;
                    #print (" missed reward... at k: ", k, " adding frames: ", int(self.post_missed_reward_lockout * self.fps))
                    #k += int(self.post_missed_reward_lockout * self.fps)
                    counter = int(self.post_missed_reward_lockout * self.sample_rate)
                    last_reward = k


                #if k % 10000==0:
                #print ("k : ", k)


            print("updated rewards #: ", n_rewards, high)

            #
            if n_rewards >= n_rewards_random:
                break

            # check exit condition otherwise decrase thresholds
            high *= stepper
            #low *= stepper

        #
        print("updated rewards #: ", n_rewards, high)

        #
        self.reward_times = np.vstack(reward_times)
        self.high = high
        self.ensemble1_traces_smooth= []
        self.ensemble2_traces_smooth= []
        self.ensemble1_traces_smooth.append(E1_1)
        self.ensemble1_traces_smooth.append(E1_2)
        self.ensemble2_traces_smooth.append(E2_1)
        self.ensemble2_traces_smooth.append(E2_2)

        #
        self.E1 = E1_1 + E1_2
        self.E2 = E2_1 + E2_2
        self.diff = self.E1-self.E2
        self.low = -high
        self.high = high


    #
    def find_reward_thresholds_high(self):

        #
        print("COMPUTED # of roi traces: ", len(self.roi_traces))
        # run smoothing on each ensemble
        self.ensemble1_traces_smooth=[]
        for p in range(len(self.ensemble1)):

            if self.smooth_diff_function_flag:

                # print ("cell id: ", self.ensemble1[p])
                smooth = np.zeros(self.ensemble1_traces[p].shape)

                # smooth each time point based on history/etc... this is how the online BMI does things
                for k in trange(self.rois_smooth_window, self.ensemble1_traces[p].shape[0], 1):
                    smooth[k] = smooth_ca_time_series4(self.ensemble1_traces[p][k - self.rois_smooth_window:k])

                #
                self.ensemble1_traces_smooth.append(smooth)

            else:
                self.ensemble1_traces_smooth.append(self.ensemble1_traces[p])

        # ensemble #2
        self.ensemble2_traces_smooth=[]
        for p in range(len(self.ensemble2)):

            if self.smooth_diff_function_flag:

                # print ("cell id: ", self.ensemble1[p])
                smooth = np.zeros(self.ensemble2_traces[p].shape)

                # smooth each time point based on history/etc... this is how the online BMI does things
                for k in trange(self.rois_smooth_window, self.ensemble2_traces[p].shape[0], 1):
                    smooth[k] = smooth_ca_time_series4(self.ensemble2_traces[p][k - self.rois_smooth_window:k])

                #
                self.ensemble2_traces_smooth.append(smooth)

            else:
                self.ensemble2_traces_smooth.append(self.ensemble2_traces[p])

        # remove F0 baseline
        for p in range(len(self.ensemble1_traces_smooth)):
            self.ensemble1_traces_smooth[p] -= np.median(self.ensemble1_traces_smooth[p])

        #
        for p in range(len(self.ensemble2_traces_smooth)):
            self.ensemble2_traces_smooth[p] -= np.median(self.ensemble2_traces_smooth[p])


        #
        E1 = np.sum(self.ensemble1_traces_smooth, axis=0)
        E2 = np.sum(self.ensemble2_traces_smooth, axis=0)


        # initialize the max and min values
        max_E1 = np.max(E1)
        max_E2 = np.max(E2)
        low = -max_E2
        high = max_E1

        #
        print("low, high: ", low, high)
        # difference between ensemble
        diff = E1 - E2

        #
        n_sec_recording = int(diff.shape[0] / self.sample_rate)
        n_rewards_random = n_sec_recording // self.sample_rate
        print("nsec recording: ", n_sec_recording,
              "max # of random rewards (i.e. every 30sec) ", n_rewards_random)
        n_rewards_random = int(n_rewards_random * self.reward_rate)
        print(" @30% reward: ", n_rewards_random)
        self.n_rewards_default = n_rewards_random

        # loop over time series decreasing the rewards until we hit the random #
        n_rewards = 0
        stepper = 0.95
        while n_rewards < n_rewards_random:

            # run inside while loop for eveyr setting of low and high until we hit
            #   exact number of random rewards
            k = 0
            n_rewards = 0
            reward_times = []
            while k < diff.shape[0]:

                temp_diff = diff[k]

                #
                if temp_diff >= high:
                    # high reward state reached
                    n_rewards += 1
                    reward_times.append([k, 1])

                    # lock out rewards for some time;
                    k += int(self.post_reward_lockout * self.sample_rate)
                else:
                    k += 1

            # check exit condition otherwise decrase thresholds
            high *= stepper
            low *= stepper

        #
        print("updated rewards #: ", n_rewards, low, high)

        #
        self.reward_times = np.vstack(reward_times)
        self.low = low
        self.high = high
        self.E1 = E1
        self.E2 = E2
        self.diff = diff

    #
    def plot_rewarded_ensembles(self):

        #
        plt.figure()
        ####################################################
        ################# VISUALIZE ROIS ###################
        ####################################################
        ax=plt.subplot(2,1,1)
        plt.title("ROIs")
        t = np.arange(self.diff.shape[0]) / self.sample_rate
        plt.plot([t[0], t[-1]], [self.low, self.low], '--', c='grey')#, label='Low threshold')
        plt.plot([t[0], t[-1]], [self.high, self.high], '--', c='grey')#, label='high threshold')


        # show rois
        #clrs = ['lightblue','darkblue','lightcoral','red']
        #names = ["Roi #1","Roi #2","Roi #3","Roi #4",]
        # ensembel 1
        for k in range(len(self.ensemble1_traces_smooth)):
            plt.plot(t, self.ensemble1_traces_smooth[k],
                             c='blue', alpha=.8)

        #
        for k in range(len(self.ensemble2_traces_smooth)):
            plt.plot(t, self.ensemble2_traces_smooth[k],
                             c='red', alpha=.8)

        #
        plt.plot(t, self.diff, c='black', alpha=1, label='Global ensemble state (i.e. E1-E2)')
        plt.plot([t[0], t[-1]], [0, 0], '--', c='black', linewidth=1, alpha=.5)

        ymaxes = np.max(np.abs(self.diff))

        # show locations of rewards
        for k in range(len(self.reward_times)):
            temp = self.reward_times[k]
            plt.plot([t[temp[0]], t[temp[0]]], [-ymaxes, ymaxes], '--', c='blue')

        # replot two random rewards just to make nice legend
        idx1 = np.where(self.reward_times[:, 1] == 1)[0].shape[0]

        #
        plt.plot([t[temp[0]], t[temp[0]]], [-ymaxes, ymaxes], '--', c='blue', label='E1 rewarded # ' + str(idx1), )
        plt.legend()
        plt.xlim(t[0],t[-1])

        ####################################################
        ################## VISUALIZE ENSEMBELS #############
        ####################################################
        ax=plt.subplot(2,1,2)
        plt.title("ENSEMBLES")
        plt.plot([t[0], t[-1]], [self.low, self.low], '--', c='grey')#, label='Low threshold')
        plt.plot([t[0], t[-1]], [self.high, self.high], '--', c='grey')#', label='high threshold')
        plt.plot(t, self.E1, c='darkblue', alpha=1, label='E1')
        plt.plot(t, self.E2, c='darkred', alpha=1, label='E2')
        plt.xlim(t[0],t[-1])
        #
        plt.plot(t, self.diff, c='black', alpha=1, label='Global ensemble state (i.e. E1-E2)')
        plt.plot([t[0], t[-1]], [0, 0], '--', c='black', linewidth=1, alpha=.5)

        ymaxes = np.max(np.abs(self.diff))
        print (" PROCESSING...")
        #
        for k in range(len(self.reward_times)):
            temp = self.reward_times[k]
            plt.plot([t[temp[0]], t[temp[0]]], [-ymaxes, ymaxes], '--', c='blue')

        # replot two random rewards just to make nice legend
        idx1 = np.where(self.reward_times[:, 1] == 1)[0].shape[0]

        #
        plt.plot([t[temp[0]], t[temp[0]]], [-ymaxes, ymaxes], '--', c='blue', label='E1 rewarded # ' + str(idx1), )
        plt.legend()

        #
        plt.suptitle("Rec duration: " + str(int(t[-1])) + " sec " +
                  "\n expected # of random rewards: " + str(int(t[-1] / 30)) +
                  "\n actual # of provided rewards: " + str(self.reward_times.shape[0]))
        plt.xlabel("Time (sec)", fontsize=20)
        plt.show()


def get_binary_std_map(std,
                       vmax=1500):
    #
    fig = plt.figure()

    sigma = 1.5

    #
    # ax=plt.subplot(111)
    plt.title("std map")
    live_image_vmin = 0
    live_image_vmax = vmax

    #
    image_obj = plt.imshow(std,
                           vmin=live_image_vmin,
                           vmax=live_image_vmax,
                           interpolation='none'
                           )
    plt.colorbar()

    axmin = fig.add_axes([0.05, 0.90, 0.1, 0.03])
    axmax = fig.add_axes([0.05, 0.93, 0.1, 0.03])

    #
    smin = Slider(axmin, 'Min', 0, live_image_vmax, valinit=live_image_vmin)
    smax = Slider(axmax, 'Max', 0, live_image_vmax, valinit=live_image_vmax)

    #
    def update_clim1(val):
        if smin.val < smax.val:
            image_obj.set_clim([smin.val,
                                smax.val])

            #
            #idx = np.where(std < 10)
            #std[idx] = np.nan

            res = scipy.ndimage.gaussian_filter(std, sigma=sigma)
            image_obj.set_data(res)
        else:
            smin.val = smax.val - 1

    #
    smin.on_changed(update_clim1)
    smax.on_changed(update_clim1)

    #
    # plt.show(block=True)

    return smin, smax


def get_img_std(smin, smax, std_map, bmi_c):
    #
    print("max proj values (vmin, vmax): ", smin.val, smax.val)

    img_std = std_map.copy()
    idx = np.where(img_std < smin.val)
    idx2 = np.where(img_std >= smin.val)

    img_std[idx] = 0
    img_std[idx2] = 1
    sigma = 1.5
    img_std = scipy.ndimage.gaussian_filter(img_std, sigma=sigma)

    bmi_c.vmin = smin.val;
    bmi_c.vmax = smax.val

    return bmi_c, img_std


#
def get_rois_stardist2d(img,
                        min_size,
                        max_size):
    # prints a list of available models
    print(StarDist2D.from_pretrained())

    # creates a pretrained model
    model = StarDist2D.from_pretrained('2D_versatile_fluo')

    # img = normalize(img[16], 1,99.8, axis=axis_norm)
    labels, details = model.predict_instances(img)

    #######################################
    # min_size_roi = 15
    # max_size_roi = 700
    # bmi_c.sigma = 0.1

    labels = labels.astype('float32')

    # remove very small and very large ROIs
    # min_size = min_size_roi
    # max_size = max_size_roi
    roi_centres = []
    footprints = []
    for k in tqdm(np.unique(labels), desc='looping over cells'):
        idx = np.where(labels == k)

        if idx[0].shape[0] < min_size or idx[0].shape[0] > max_size:
            labels[idx] = np.nan
            img[idx] = 0
        else:

            roi_centres.append([np.median(idx[0]),
                                np.median(idx[1])])
            footprints.append(idx)

    roi_centres = np.vstack(roi_centres)

    plt.figure(figsize=(8, 8))
    plt.imshow(img if img.ndim == 2 else img[..., 0], clim=(0, 1), cmap='gray')
    plt.imshow(labels, cmap='viridis', alpha=0.5)
    plt.axis('off');

    plt.show()

    #

    return roi_centres, footprints


def save_calibration_data_new_day(bmi_c, text=''):
    # save all data to disk
    # also add the tone values here as well that will be used for the experiment
    bmi_c.low_freq = 2000
    bmi_c.high_freq = 16000

    #

    # get ensemble f0 baselines
    ensemble1_f0s = []
    for k in bmi_c.ensemble1:
        # get footprints
        ensemble1_f0s.append(bmi_c.roi_f0s[k])

    # get ensemble f0 baselines
    ensemble2_f0s = []
    for k in bmi_c.ensemble2:
        # get footprints
        ensemble2_f0s.append(bmi_c.roi_f0s[k])


    # save individual pixels of each cell - currently implemented in BMI

    idx = bmi_c.fname.index('/calibration')
    fname_out = bmi_c.fname[:idx]+'/rois_pixels_and_thresholds.npz'
    #
    # if text == '':
    #     fname_out = os.path.join(os.path.split(os.path.split(bmi_c.fname)[0])[0],
    #                              'rois_pixels_and_thresholds.npz')
    # else:
    #     fname_out = os.path.join(os.path.split(os.path.split(bmi_c.fname)[0])[0],
    #                              'rois_pixels_and_thresholds_' + text + '.npz')

    np.savez(fname_out,

             #
             f0_allcells=bmi_c.roi_f0s,

             #
             ensemble1_footprints=bmi_c.ensemble1_footprints,
             ensemble1_contours=bmi_c.ensemble1_contours,
             ensemble1_f0s=ensemble1_f0s,

             #
             ensemble2_footprints=bmi_c.ensemble2_footprints,
             ensemble2_contours=bmi_c.ensemble2_contours,
             ensemble2_f0s=ensemble2_f0s,

             #
             reward_rate=bmi_c.reward_rate,
             reward_rate_scaling_factor=bmi_c.reward_rate_scaling_factor,

             #
             contours_all_cells=bmi_c.contours_all_cells,
             # cell_centres = np.int32(bmi_c.rois)[both],
             cell_ids=bmi_c.both,
             # all_rois = np.int32(bmi_c.rois),
             low_threshold=bmi_c.low,
             high_threshold=bmi_c.high,
             low_freq=bmi_c.low_freq,
             high_freq=bmi_c.high_freq,
             all_roi_traces_submsampled=bmi_c.roi_traces,

             #
             sample_rate=bmi_c.sample_rate,
             post_reward_lockout=bmi_c.post_reward_lockout,
             balance_ensemble_rewards_flag=bmi_c.balance_ensemble_rewards_flag,
             rois_smooth_window=bmi_c.rois_smooth_window,
             smooth_diff_function_flag=bmi_c.smooth_diff_function_flag,
             calibration_template=bmi_c.std_map,
             footprints=bmi_c.footprints,
             roi_traces=bmi_c.roi_traces,
             roi_f0s=bmi_c.roi_f0s

             )

    # also save the entire object as a pickle
    try:
        file_pi = open(os.path.join(os.path.split(fname_out)[0], "bmi_c.obj"), 'wb')
        bmi_c.data = None
        pickle.dump(bmi_c, file_pi)
    except:
        print(" couldn't save bmi_c.object .... TO FIX!")
    print("Done...")

    #return bmi_c

def save_calibration_data(bmi_c, text=''):

    # save all data to disk
    # also add the tone values here as well that will be used for the experiment
    bmi_c.low_freq = 2000
    bmi_c.high_freq = 16000

    #
    ensemble1_footprints = []
    ensemble1_contours = []
    for k in bmi_c.ensemble1:

        # get footprints
        temp = bmi_c.footprints[k]
        temp1 = temp[0]
        temp2 = temp[1]
        temp = np.vstack((temp1,temp2))
        ensemble1_footprints.append(temp.T)

        # get contours
        ensemble1_contours.append(bmi_c.compute_contour_map(bmi_c.std_map, [k]))

    # get ensembel 2 footprints/contours
    ensemble2_footprints = []
    ensemble2_contours = []
    for k in bmi_c.ensemble2:
        # get footprints
        temp = bmi_c.footprints[k]
        temp1 = temp[0]
        temp2 = temp[1]
        temp = np.vstack((temp1,temp2))
        ensemble2_footprints.append(temp.T)

        # get contours
        ensemble2_contours.append(bmi_c.compute_contour_map(bmi_c.std_map, [k]))

    # get ensemble f0 baselines
    ensemble1_f0s = []
    for k in bmi_c.ensemble1:
        # get footprints
        ensemble1_f0s.append(bmi_c.roi_f0s[k])

    # get ensemble f0 baselines
    ensemble2_f0s = []
    for k in bmi_c.ensemble2:
        # get footprints
        ensemble2_f0s.append(bmi_c.roi_f0s[k])

    # also grab contours of cells; both contains all cell ids
    contours_all_cells = []
    for k in range(len(bmi_c.footprints)):
        contours_all_cells.append(bmi_c.compute_contour_map(bmi_c.std_map, [k]))
    bmi_c.contours_all_cells = np.array(contours_all_cells, dtype='object')

    # save individual pixels of each cell - currently implemented in BMI
    if text=='':
        fname_out = os.path.join(os.path.split(os.path.split(bmi_c.fname)[0])[0],
                            'rois_pixels_and_thresholds.npz')
    else:
        fname_out = os.path.join(os.path.split(os.path.split(bmi_c.fname)[0])[0],
                                 'rois_pixels_and_thresholds_'+text+'.npz')

    np.savez(fname_out,

                #
                f0_allcells = bmi_c.roi_f0s,

                #
                ensemble1_footprints = ensemble1_footprints,
                ensemble1_contours = ensemble1_contours,
                ensemble1_f0s = ensemble1_f0s,

                #
                ensemble2_footprints = ensemble2_footprints,
                ensemble2_contours = ensemble2_contours,
                ensemble2_f0s = ensemble2_f0s,

                #
                reward_rate = bmi_c.reward_rate,
                reward_rate_scaling_factor = bmi_c.reward_rate_scaling_factor,

                #
                contours_all_cells = bmi_c.contours_all_cells,
                #cell_centres = np.int32(bmi_c.rois)[both],
                cell_ids = bmi_c.both,
                #all_rois = np.int32(bmi_c.rois),
                low_threshold = bmi_c.low,
                high_threshold = bmi_c.high,
                low_freq = bmi_c.low_freq,
                high_freq = bmi_c.high_freq,
                all_roi_traces_submsampled = bmi_c.roi_traces,

                #
                sample_rate = bmi_c.sample_rate,
                post_reward_lockout = bmi_c.post_reward_lockout,
                balance_ensemble_rewards_flag = bmi_c.balance_ensemble_rewards_flag,
                rois_smooth_window = bmi_c.rois_smooth_window,
                smooth_diff_function_flag = bmi_c.smooth_diff_function_flag,
                calibration_template = bmi_c.std_map,
                footprints = bmi_c.footprints,
                roi_traces = bmi_c.roi_traces,
                roi_f0s = bmi_c.roi_f0s

            )

    # also save the entire object as a pickle
    # file_pi = open(os.path.join(os.path.split(fname_out)[0], "bmi_c.json"), 'wb')
    # bmi_c.data=None
    # file_pi.write(bmi_c)
    # file_pi.close()

    try:
        file_pi = open(os.path.join(os.path.split(fname_out)[0], "bmi_c.obj"), 'wb')
        bmi_c.data=None
        pickle.dump(bmi_c, file_pi)
    except:
        print (" couldn't save bmi_c.object .... TO FIX!")
    print ("Done...")

    return bmi_c

def compute_roi_traces_f0_alignment(fname,
                                    footprints,
                                    cell_ids,
                                    subsample,
                                    ):

    #
    #cell_ids = np.arange(len(footprints))

    #
    data = np.memmap(fname, dtype='uint16', mode='r')
    data = data.reshape(-1, 512, 512) #.transpose(0,2,1)

    #####################################################
    ################ COMPUTE ROI TRACES #################
    #####################################################
    #

    print (" Computing traces; for cells:L ", cell_ids)
    roi_traces = np.zeros((len(footprints), data.shape[0]//subsample))

    # loop over each frame
    for p in trange(0, data.shape[0], subsample,
                    desc='computing roi traces for SNR indexing'):

        # grab frame
        frame = data[p]

        # loop over ROIS
        ctr = 0
        for k in cell_ids:
            # grab roi
            temp = frame[footprints[k]]

            # normalize by surface area so that cells don't look way different because of footprint size
            if True:
                #print (footprints[k].shape[0])
                temp = temp / footprints[k][0].shape[0]

            # add pixel values inside roi
            temp = np.nansum(temp)

            # save
            roi_traces[k,p//subsample] = temp
            ctr += 1

    ###########################################################
    ################### COMPUTE F0 AND SNR ####################
    ###########################################################
    # compute the baseline f0 of the cells in order to be able to offset it in the BMI
    # TODO: this is important; it functions as a rough DFF method
    #    TODO: we may wish to implement a more complex version of this
    roi_f0s = np.zeros(roi_traces.shape[0], dtype=np.float32)
    for k in cell_ids:

        #
        roi_f0s[k] = np.nanmedian(roi_traces[k])
        print (k, "F0: ", roi_f0s[k])

    #
    return roi_f0s, roi_traces


    #

def align_to_prev_day(bmi_c):
    # load contours
    contours = bmi_c.align_data['contours_all_cells']
    print("# cells from previous day: ", contours.shape)

    # load footprints
    raw_footprints = bmi_c.align_data['footprints']
    bmi_c.footprints = []
    for k in range(len(raw_footprints)):
        #
        temp = raw_footprints[k]
        temp1 = temp[0]
        temp2 = temp[1]
        temp = np.vstack((temp1, temp2)).T
        temp = temp[:, 0], temp[:, 1]

        #
        bmi_c.footprints.append(temp)

    # load ensemble cell ids:
    cell_ids = np.int32(bmi_c.align_data['cell_ids'])
    print("original cell ids: ", cell_ids)

    bmi_c.ensemble1 = [cell_ids[0], cell_ids[1]]
    bmi_c.ensemble2 = [cell_ids[2], cell_ids[3]]

    # load original footprint f0s
    if False:
        bmi_c.roi_f0s = []
        for k in range(len(temp)):
            bmi_c.roi_f0s.append(0)

        ensemble1_f0s = data['ensemble1_f0s']
        bmi_c.rois_f0s[cell_ids[0]] = ensemble1_f0s[0]
        bmi_c.rois_f0s[cell_ids[1]] = ensemble1_f0s[1]

        ensemble2_f0s = data['ensemble2_f0s']
        bmi_c.rois_f0s[cell_ids[2]] = ensemble2_f0s[0]
        bmi_c.rois_f0s[cell_ids[3]] = ensemble2_f0s[1]

    # recompute f0s for current session
    else:
        subsample = 10
        print("recomputing rois for ensmbel cells...")
        bmi_c.roi_f0s, bmi_c.roi_traces = compute_roi_traces_f0_alignment(bmi_c.fname,
                                                                          bmi_c.footprints,
                                                                          cell_ids,
                                                                          subsample)

    ###################################
    ###################################
    ###################################
    if False:
        plt.figure()
        for k in range(2):
            plt.plot(bmi_c.roi_traces[bmi_c.ensemble1[k]] + k * 500,
                     c='blue')

        for k in range(2):
            plt.plot(bmi_c.roi_traces[bmi_c.ensemble2[k]] + 1000 + k * 500,
                     c='red',
                     )

        plt.show()

        plt.figure()
        plt.imshow(bmi_c.template)
        for k in range(len(contours)):
            for p in range(len(contours[k]) - 1):
                plt.plot([contours[k][p][0], contours[k][p + 1][0]],
                         [contours[k][p][1], contours[k][p + 1][1]],
                         c='white')

        plt.show()

    # save ensemble rois
    bmi_c.both = np.hstack((bmi_c.ensemble1, bmi_c.ensemble2))
    print("all cells:", bmi_c.both)


    return bmi_c


def get_footprints_from_day0(bmi_c):


    ########################################################
    ########################################################
    ########################################################
    # initialize calcium object and load suite2p data
    data_dir = os.path.split(bmi_c.fname_day0)[0]
    c = bmi_c.calcium.Calcium()
    c.verbose = True  # outputs additional information during processing
    c.recompute_binarization = False  # recomputes binarization and other processing steps; False: loads from previous saved locations if avialable
    c.data_dir = data_dir
    c.load_suite2p()

    # this loads the suite2p footprints
    c.load_footprints()
    print("# of footprints; ", len(c.footprints))

    #
    ########################################################
    ########################################################
    ########################################################
    idx = bmi_c.fname_day0.index('calibration')
    fname = os.path.join(bmi_c.fname_day0[:idx-1],'rois_pixels_and_thresholds_day0.npz')

    print ("to load fname: ", fname)

    data = np.load(fname, allow_pickle = True)
    #
    bmi_c.footprints = data['footprints']
    bmi_c.ensemble1_footprints = data['ensemble1_footprints']
    bmi_c.ensemble2_footprints = data['ensemble2_footprints']
    bmi_c.ensemble1_contours = data['ensemble1_contours']
    bmi_c.ensemble2_contours = data['ensemble2_contours']
    bmi_c.contours_all_cells = data['contours_all_cells']


    #
    bmi_c.roi_traces = data['roi_traces']
    bmi_c.roi_f0s = data['roi_f0s']
    bmi_c.ensemble_ids = data['cell_ids']
    bmi_c.ensemble1 = bmi_c.ensemble_ids[:2]
    bmi_c.ensemble2 = bmi_c.ensemble_ids[2:]
    print ("ENSMBEL CELL IDS: ", bmi_c.ensemble_ids)

    #
    plt.figure()
    plt.imshow(bmi_c.std_map)
    for k in range(len(c.footprints)):
        plt.plot(c.contours[k][:, 0], c.contours[k][:, 1])

        #
        footprint = c.footprints[k].T
        #idx = np.where(footprint >0)
        #bmi_c.footprints.append(idx)

        # print (temp)
        idx = np.where(footprint <= 0)
        idx2 = np.where(footprint > 0)
        footprint[idx] = np.nan
        footprint[idx2] = 1

        # print (np.nanmax(temp), np.nanmin(temp))
        plt.imshow(footprint,
                   vmin=0,
                   vmax=1)

    plt.xlim(0, 512)
    plt.ylim(0, 512)
    plt.colorbar()
    plt.show()

    return bmi_c


def get_footprints_from_suite2p(bmi_c):
    data_dir = os.path.split(bmi_c.fname)[0]

    # initialize calcium object and load suite2p data
    c = bmi_c.calcium.Calcium()
    c.verbose = True  # outputs additional information during processing
    c.recompute_binarization = False  # recomputes binarization and other processing steps; False: loads from previous saved locations if avialable
    c.data_dir = data_dir
    c.load_suite2p()

    # this loads the suite2p footprints
    c.load_footprints()
    print("# of footprints; ", len(c.footprints))

    bmi_c.footprints = []

    #
    plt.figure()
    plt.imshow(bmi_c.std_map)
    for k in range(len(c.footprints)):
        plt.plot(c.contours[k][:, 0], c.contours[k][:, 1])

        #
        footprint = c.footprints[k].T
        idx = np.where(footprint >0)
        bmi_c.footprints.append(idx)

        # print (temp)
        idx = np.where(footprint <= 0)
        idx2 = np.where(footprint > 0)
        footprint[idx] = np.nan
        footprint[idx2] = 1

        # print (np.nanmax(temp), np.nanmin(temp))
        plt.imshow(footprint,
                   vmin=0,
                   vmax=1)

    plt.xlim(0, 512)
    plt.ylim(0, 512)
    plt.colorbar()
    plt.show()

    return bmi_c