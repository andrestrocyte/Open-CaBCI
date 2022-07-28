import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange, tqdm

from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import signal
import scipy
import scipy.ndimage
import cv2
from matplotlib.widgets import Slider, Button, RadioButtons

from stardist.models import StarDist2D
from utils.utils import smooth_ca_time_series, compute_dff0, compute_dff0_with_reference


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
            sigma = 1
            order = 0
            print(" gaussian filter width: ", sigma, ", order: ", order)
            data_sparse = scipy.ndimage.gaussian_filter(data_sparse,
                                                        sigma,
                                                        order)

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

        std = np.std(data_sparse, axis=0)

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
            # temp = temp.astype('uint8')

            #
            contour, _ = cv2.findContours(temp,
                                          cv2.RETR_TREE,
                                          cv2.CHAIN_APPROX_SIMPLE)
            contour = contour[0].squeeze()
            contour = np.vstack((contour, contour[0]))

            #
            contour_array.append(contour)

        return contour_array

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
        for p in cell_ids:
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
                         c='white')
            #
            z = np.vstack(footprints[p]).T
            plt.text(np.median(z[:, 1]), np.median(z[:, 0]), str(p), c='red')

        plt.show()

    #
    def compute_and_plot_traces2_datafile(self, data, std_map, cell_ids=None, fig=None):
        """ Same as below but visualize every single frame
		"""

        #
        clrs = ['blue', 'green', 'red', 'orange']

        #
        if cell_ids is None:
            cell_ids = np.arange(len(self.footprints))
        print("plotting cells: ", cell_ids)

        #####################################################
        plt.figure()
        ax = plt.subplot(111)
        ax.tick_params(axis='both', which='both', labelsize=20)
        plt.ylabel("Neuron ID ", fontsize=20)

        #
        new_plot = False
        print(cell_ids)
        self.show_contour_map(std_map,
                              self.footprints,
                              cell_ids, new_plot)

        plt.show()

        return
        #####################################################
        plt.figure()
        ax = plt.subplot(111)

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
        self.roi_traces = roi_traces

        #
        t = np.arange(0, data.shape[0], self.trace_subsample) / 30.
        ctr = 0

        # save the baselin of the cells in order to be able to offset it in the BMI
        # TODO: this is important; it functions as a rough DFF method
        #    TODO: we may wish to implement a more complex version of this
        self.roi_f0s = np.zeros(len(roi_traces), dtype=np.float32)
        for k in range(len(roi_traces)):
            temp = roi_traces[k]
            self.roi_f0s[k] = np.median(temp)
            temp = temp - self.roi_f0s[k]
            plt.plot(t, temp + ctr * self.scale,
                     linewidth=2,
                     c=clrs[k])

            # also plot baseline
            baseline = np.median(temp[:5000] + ctr * self.scale)
            plt.plot([t[0], t[-1]], [baseline, baseline], '--',
                     linewidth=4,
                     c='black')
            ctr += 1
        #
        labels = cell_ids
        labels_old = np.arange(0, ctr * self.scale, self.scale)

        #
        plt.yticks(labels_old, labels, fontsize=10)
        plt.xlabel("Time (sec)", fontsize=20)

        plt.show()

    def reorder_cells_by_snr_or_f0(self,
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

        ###########################################################
        ################### COMPUTE F0 AND SNR ####################
        ###########################################################
        # compute the baseline f0 of the cells in order to be able to offset it in the BMI
        # TODO: this is important; it functions as a rough DFF method
        #    TODO: we may wish to implement a more complex version of this
        self.roi_f0s = np.zeros(self.roi_traces.shape[0], dtype=np.float32)
        self.roi_snrs = np.zeros(self.roi_traces.shape[0], dtype=np.float32)
        for k in range(self.roi_traces.shape[0]):

            #
            f0 = np.median(self.roi_traces[k])

            #
            self.roi_f0s[k] = f0

            #
            self.roi_snrs[k] = np.max(self.roi_traces[k]/f0)

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
        for k in range(idx.shape[0]):
            self.footprints_temp.append(self.footprints[idx[k]])

        self.footprints = self.footprints_temp

    #

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
        self.roi_traces_fullres = []
        ctr=0
        for k in range(len(roi_traces)):
            temp = roi_traces[k]
            temp = temp - self.roi_f0s[k]

            # we update the selected traces time dynamics
            self.roi_traces_fullres.append(temp)

            #
            t = np.linspace(0, data.shape[0], temp.shape[0]) / 30.
            plt.plot(t, temp + ctr * self.scale)

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
        ax = plt.subplot(121)
        ax.tick_params(axis='both', which='both', labelsize=20)
        plt.ylabel("Neuron ID ", fontsize=20)

        #
        self.roi_f0s = np.zeros(self.roi_traces.shape[0], dtype=np.float32)
        ctr = 0
        for k in range(self.roi_traces.shape[0]):
            temp = self.roi_traces[k]
            temp = temp - self.roi_f0s[k]


            # each cell might have different time signatures
            t = np.linspace(0, data.shape[0], temp.shape[0]) / 30.

            plt.plot(t, temp + ctr * self.scale)

            ctr += 1

        ###########################################################
        ################### FINISH UP LABELS ETC ##################
        ###########################################################
        labels = cell_ids
        labels_old = np.arange(0, ctr * self.scale, self.scale)

        #
        plt.yticks(labels_old, labels, fontsize=10)
        plt.xlabel("Time (sec)", fontsize=20)

        ###########################################################
        ################### PLOT IMAGE OF [CA] ####################
        ###########################################################
        plt.subplot(122)
        new_plot = False
        self.show_contour_map(std_map,
                              self.footprints,
                              cell_ids,
                              new_plot)

        plt.show()

    #
    def show_traces_ids(self, ids):

        #
        fig = plt.figure()

        #
        plt.title("Cell Ids: " + str(ids))
        #
        t = np.arange(0, self.roi_traces[0].shape[0], 1) / 30. * self.trace_subsample
        ctr = 0
        for k in ids:
            temp = self.roi_traces[k]
            temp = temp - np.median(temp)
            plt.plot(t, temp + ctr * self.scale)

            ctr += 1

        labels = np.arange(len(ids))
        labels_old = np.arange(0, ctr * self.scale, self.scale)

        #
        plt.yticks(labels_old, labels)
        plt.xlabel("Time (sec)")

        plt.show()

    #
    def find_reward_thresholds_absolute(self, normalize_peaks=True):
        '''  Computes the aboslute |E1-E2|
		     and rewards anytime the ensembel goes above this value
		     Note that self.roi_traces contains only the 4 neurons from the ensembes selected
		     now
		     - TODO: change this for the high and high_low functions also

		'''

        # TODO: refactor this part and send it to the BMI session code

        # run smoothing on each ensemble
        if self.smooth_diff_function_flag:

            # ensemble #1
            for p in range(2):
                smooth = np.zeros(self.roi_traces[p].shape)
                for k in trange(self.rois_smooth_window, self.roi_traces[p].shape[0], 1):
                    smooth[k] = self.smooth_ca_time_series(self.roi_traces[p][k - self.rois_smooth_window:k])
                #
                self.roi_traces[p] = smooth

            # ensemble #2
            for p in range(2, 4, 1):
                smooth = np.zeros(self.roi_traces[p].shape)
                for k in trange(self.rois_smooth_window, self.roi_traces[p].shape[0], 1):
                    smooth[k] = self.smooth_ca_time_series(self.roi_traces[p][k - self.rois_smooth_window:k])
                #
                self.roi_traces[p] = smooth

        #
        self.roi_f0s = []
        self.dff0 = []
        for k in range(len(self.roi_traces)):
            f0, dff0 = self.compute_dff0(self.roi_traces[k])
            self.roi_f0s.append(f0)
            self.dff0.append(dff0)

        # compute ensembles using the smooth + baseline removed values
        E1 = self.dff0[0] - self.dff0[1]
        E2 = self.dff0[2] - self.dff0[3]

        # initialize the max and min values
        max_E1 = np.max(E1)
        max_E2 = np.max(E2)

        print(
            "TODO: Normalize the peaks of the 2 Ensembles so the mice don't learn to use one esnemble against the other!!!!")
        low = np.nan
        high = min(max_E1, max_E2) * 3

        print("low, high: ", low, high)
        # difference between ensemble
        diff = np.abs(E1 - E2)

        #
        self.n_sec_recording = int(diff.shape[0] / self.sample_rate)
        self.n_rewards_random = self.n_sec_recording // self.sample_rate
        self.n_rewards_default = int(self.n_rewards_random * 0.3)
        print("nsec recording: ", self.n_sec_recording,
              "max # of random rewards (i.e. every 30sec) ", self.n_rewards_random,
              "# of rewards for 30% of the time: ", self.n_rewards_default)

        # loop over time series decreasing the rewards until we hit the random #
        n_rewards = 0
        stepper = 0.95
        while n_rewards < self.n_rewards_default:

            # run inside while loop for eveyr setting of low and high until we hit
            #   exact number of random rewards
            k = 0
            n_rewards = 0
            reward_times = []
            while k < diff.shape[0]:

                temp_diff = diff[k]

                if temp_diff >= high:
                    # high reward state reached
                    n_rewards += 1
                    reward_times.append([k, 1])
                    k += int(self.post_reward_lockout * self.sample_rate)
                else:
                    k += 1

            # print ("Reard times: ", reward_times)
            # check exit condition otherwise decrase thresholds
            # if len(reward_times) > 1:
            # 	rewarded_times = np.vstack(reward_times)
            #	high *= stepper
            # else:
            high *= stepper

        print("updated rwards #: ", n_rewards, low, high)

        self.reward_times = np.vstack(reward_times)

        self.low = np.nan
        self.high = high
        self.E1 = E1
        self.E2 = E2
        self.diff = diff

    #
    def find_reward_thresholds_high(self):

        #
        print("COMPUTED # of roi traces: ", len(self.roi_traces))

        # run smoothing on each ensemble
        if self.smooth_diff_function_flag:

            # ensemble #1
            for p in range(4):
                # print ("cell id: ", self.ensemble1[p])
                smooth = np.zeros(self.roi_traces_fullres[p].shape)
                for k in trange(self.rois_smooth_window, self.roi_traces_fullres[p].shape[0], 1):
                    smooth[k] = smooth_ca_time_series(self.roi_traces_fullres[p][k - self.rois_smooth_window:k])
                #
                self.roi_traces_fullres[p] = smooth

        # get baseline f0 after smoothing
        self.roi_f0s = []
        for k in range(4):
            self.roi_f0s.append(np.median(self.roi_traces_fullres[k], axis=0))

        # detrend traces and make ensembles
        temp0 = (self.roi_traces_fullres[0] - self.roi_f0s[0]) / self.roi_f0s[0]
        temp1 = (self.roi_traces_fullres[1] - self.roi_f0s[1]) / self.roi_f0s[1]
        E1 = temp0 + temp1

        #
        temp2 = (self.roi_traces_fullres[2] - self.roi_f0s[2]) / self.roi_f0s[2]
        temp3 = (self.roi_traces_fullres[3] - self.roi_f0s[3]) / self.roi_f0s[3]
        E2 = temp2 + temp3

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
        n_rewards_random = int(n_rewards_random * 0.3)
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
    def find_reward_thresholds_low_and_high(self):

        # run smoothing on each ensemble
        if self.smooth_diff_function_flag:

            for p in range(2):
                smooth = np.zeros(self.roi_traces[self.ensemble1[p]].shape)
                for k in trange(self.rois_smooth_window, self.roi_traces[self.ensemble1[p]].shape[0], 1):
                    smooth[k] = smooth_ca_time_series(self.roi_traces[self.ensemble1[p]][k - self.rois_smooth_window:k])
                #
                self.roi_traces[self.ensemble1[p]] = smooth

            for p in range(2):
                smooth = np.zeros(self.roi_traces[self.ensemble2[p]].shape)
                for k in trange(self.rois_smooth_window, self.roi_traces[self.ensemble2[p]].shape[0], 1):
                    smooth[k] = smooth_ca_time_series(self.roi_traces[self.ensemble2[p]][k - self.rois_smooth_window:k])
                #
                self.roi_traces[self.ensemble2[p]] = smooth

        # detrend traces and make ensembles
        temp0 = self.roi_traces[self.ensemble1[0]] - np.median(self.roi_traces[self.ensemble1[0]], axis=0)
        temp1 = self.roi_traces[self.ensemble1[1]] - np.median(self.roi_traces[self.ensemble1[1]], axis=0)
        E1 = temp0 + temp1

        #
        temp2 = self.roi_traces[self.ensemble2[0]] - np.median(self.roi_traces[self.ensemble2[0]], axis=0)
        temp3 = self.roi_traces[self.ensemble2[1]] - np.median(self.roi_traces[self.ensemble2[1]], axis=0)
        E2 = temp2 + temp3

        # initialize the max and min values
        max_E1 = np.max(E1)
        max_E2 = np.max(E2)
        low = -max_E1
        high = max_E2

        print("low, high: ", low, high)
        # difference between ensemble
        diff = E1 - E2

        #
        n_sec_recording = int(diff.shape[0] / self.sample_rate)
        n_rewards_random = n_sec_recording // self.sample_rate
        print("nsec recording: ", n_sec_recording,
              "max # of random rewards (i.e. every 30sec) ", n_rewards_random)

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
                if temp_diff <= low:
                    # low reward state reached
                    n_rewards += 1
                    reward_times.append([k, 0])
                    k += int(self.post_reward_lockout * self.sample_rate)
                elif temp_diff >= high:
                    # high reward state reached
                    n_rewards += 1
                    reward_times.append([k, 1])
                    k += int(self.post_reward_lockout * self.sample_rate)
                else:
                    k += 1

            # print ("Reard times: ", reward_times)
            # check exit condition otherwise decrase thresholds
            if len(reward_times) > 1:
                rewarded_times = np.vstack(reward_times)
                if self.balance_ensemble_rewards_flag:
                    idx_E1 = np.where(rewarded_times[:, 1] == 0)[0].shape[0]
                    idx_E2 = np.where(rewarded_times[:, 1] == 1)[0].shape[0]
                    if idx_E1 <= idx_E2:
                        low *= stepper
                    else:
                        high *= stepper
                else:
                    low *= stepper
                    high *= stepper
            else:
                low *= stepper
                high *= stepper

        print("updated rwards #: ", n_rewards, low, high)

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

        t = np.arange(self.diff.shape[0]) / self.sample_rate
        plt.plot([t[0], t[-1]], [self.low, self.low], '--', c='grey')
        plt.plot([t[0], t[-1]], [self.high, self.high], '--', c='grey')
        plt.plot(t, self.E1, c='blue', alpha=.2, label='E1')
        plt.plot(t, self.E2, c='red', alpha=.2, label='E2')
        plt.plot(t, self.diff, c='black', alpha=.8, label='Difference')
        plt.plot([t[0], t[-1]], [0, 0], c='black', linewidth=3)

        ymaxes = np.max(np.abs(self.diff))

        #
        for k in range(len(self.reward_times)):
            temp = self.reward_times[k]

            # if temp[1] == 0:
            #    plt.plot([t[temp[0]], t[temp[0]]], [-ymaxes, ymaxes], '--', c='red')
            # else:
            plt.plot([t[temp[0]], t[temp[0]]], [-ymaxes, ymaxes], '--', c='blue')

        # replot two random rewards just to make nice legend
        idx1 = np.where(self.reward_times[:, 1] == 1)[0].shape[0]
        # idx2 = np.where(self.reward_times[:, 1] == 1)[0].shape[0]

        #
        plt.plot([t[temp[0]], t[temp[0]]], [-ymaxes, ymaxes], '--', c='blue', label='E1 rewarded # ' + str(idx1), )
        # plt.plot([t[temp[0]], t[temp[0]]], [-ymaxes, ymaxes], '--', c='red', label='E2 rewarded # ' + str(idx2), )
        plt.legend()

        #
        plt.title("Rec duration: " + str(int(t[-1])) + " sec " +
                  "\n expected # of random rewards: " + str(int(t[-1] / 30)) +
                  "\n actual # of provided rewards: " + str(self.reward_times.shape[0]))
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
            idx = np.where(std < 10)
            std[idx] = np.nan

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

    rois = np.vstack(roi_centres)

    plt.figure(figsize=(8, 8))
    plt.imshow(img if img.ndim == 2 else img[..., 0], clim=(0, 1), cmap='gray')
    plt.imshow(labels, cmap='viridis', alpha=0.5)
    plt.axis('off');

    plt.show()

    #

    return rois, footprints
