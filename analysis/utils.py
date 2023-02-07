import numpy as np
# Visualisation
import matplotlib.pyplot as plt
import numpy
import os
import pandas as pd

import scipy.ndimage
#from matplotlib_scalebar.scalebar import ScaleBar


class ProcessSession():

    def __init__(self,
                 root_dir,
                 animal_id,
                 session_id):

        #
        self.root_dir = root_dir
        self.animal_id = animal_id
        self.session_id = session_id

        #
        self.sample_rate = 30
        print ("sample rate: ", self.sample_rate)

        #
        self.save_dir = os.path.join(self.root_dir,
                                     self.animal_id,
                                     self.session_id,
                                     'results')

        #
        if os.path.exists(self.save_dir)==False:
            os.mkdir(self.save_dir)

    def load_data(self):

        #
        fname = os.path.join(self.root_dir,
                             self.animal_id,
                             self.session_id,
                             'data', 'results.npz')
        #
        data = np.load(fname, allow_pickle=True)

        fname_dict = os.path.join(self.root_dir,
                                  self.animal_id,
                                  self.session_id,
                                  'data', 'results.xlsx')

        #
        df = pd.read_excel(fname_dict)
        D = df.iloc[:, 1:].values

        #
        self.white_noise = D[:, 2]
        self.high_threshold = D[:, 1]
        post_reward = D[:, 3]

        #
        self.reward_times = np.int32(data['rewarded_times_abs'][:, 1])
        print("reward times: ", self.reward_times.shape)

        #
        self.abs_times = data['abs_times']
        print("abs times: ", self.abs_times.shape, self.abs_times)

        self.ttl_times = data['ttl_times']
        print("ttl times: ", self.ttl_times.shape, self.ttl_times[0], self.ttl_times[-1], " total rec time sec: ",
              self.ttl_times[-1] - self.ttl_times[0])

        self.ttl_comp = data['ttl_n_computed']
        print("ttl computed: ", self.ttl_comp.shape, self.ttl_comp)

        #
        self.ttl_det = data['ttl_n_detected']
        print("ttl detected: ", self.ttl_det.shape)

        self.lick_detector = data['lick_detector_abstime']
        print("lick detector: ", self.lick_detector.shape)
        # lick_detector -= lick_detector[0]
        idx = np.where(self.lick_detector > 3)[0]
        self.lick_times = self.abs_times[idx] - self.ttl_times[0]
        print("lick times: ", self.lick_times)
        # licks = result[0]/(30*33.425)

        #
        self.ttl_times -= self.ttl_times[0]

        #
        self.E = data['ensemble_diff_array']
        print(self.E.shape)

        #
        self.E1 = data["rois_traces_smooth1"]
        self.E2 = data["rois_traces_smooth2"]
        self.E1[:, :10] = 0
        self.E2[:, :10] = 0
        print("E1 , E2, ", self.E1.shape, self.E2.shape)

        #
        self.rec_len_mins = self.E1.shape[1]/self.sample_rate/60.

        #
        print ("Recording length (mins): ", self.rec_len_mins)


        print("DONE...")

    def show_session_traces(self):

        #
        ########################################################
        ########################################################
        ########################################################
        plt.figure(figsize=(40,20))
        ax = plt.subplot(1, 1, 1)
        #mng = plt.get_current_fig_manager()
        #mng.resize(*mng.window.maxsize())
        #
        scale = 2
        alpha = .7
        plt.plot(self.ttl_times, self.E1[0, self.ttl_comp], c='blue', label='roi1', alpha=alpha)
        plt.plot(self.ttl_times, self.E1[1, self.ttl_comp] + scale, c='lightblue', label='roi2', alpha=alpha)
        plt.plot(self.ttl_times, self.E2[0, self.ttl_comp] + scale * 2, c='red', label='roi3', alpha=alpha)
        plt.plot(self.ttl_times, self.E2[1, self.ttl_comp] + scale * 3, c='pink', label='roi4', alpha=alpha)

        # total ensemble state
        ctr = 6
        plt.plot(self.ttl_times, self.E[self.ttl_comp] + scale * ctr, c='black', label='Ensemble state', alpha=.3)

        # plot rewarded times
        plt.scatter(self.ttl_times[self.reward_times],
                    self.high_threshold[self.reward_times] + scale * ctr, s=25, c='green', label='reward times')

        # add lines
        for k in range(self.reward_times.shape[0]):
            plt.plot([self.ttl_times[self.reward_times[k]], self.ttl_times[self.reward_times[k]]],
                     [0, self.high_threshold[self.reward_times[k]] + scale * ctr], c='green', alpha=.2)

        # plot reward threshold
        plt.plot(self.ttl_times,
                 self.high_threshold + scale * ctr, '--', c='lightgreen', label='threshold')

        # plot white noise
        idx = np.where(self.white_noise)[0]
        ax.scatter(
            self.ttl_times[idx],
            self.ttl_times[idx] * 0 + scale * ctr,
            color='brown',
            alpha=1,
            label='white-nose',
        )

        # lick times
        ctr += .4
        plt.scatter(self.lick_times, self.lick_times * 0 + scale * ctr, alpha=0.8, c='orange', label='lick detector')

        #
        plt.legend()
        plt.xlabel("Time (sec)")
        plt.xlim(self.ttl_times[0], self.ttl_times[-1])
        plt.suptitle(self.animal_id + " " + self.session_id)


        plt.savefig(os.path.join(self.save_dir,'session.png'),dpi=200)
        plt.show()


    #
    def compute_ensemble_correlations(self):

        #
        from scipy import stats
        self.E1_corr = stats.pearsonr(self.E1[0], self.E1[1])
        self.E2_corr = stats.pearsonr(self.E2[0], self.E2[1])

        print("pearson correlation E1 cells; ", self.E1_corr[0])
        print("pearson correlation E2 cells; ", self.E2_corr[0])

    #
    def compute_correlograms_reward_vs_licking(self):

        from correlograms_phy import correlograms

        # TODO: note all times are given in seconds

        # rewarded times
        spikes1 = self.ttl_times[self.reward_times]

        # lick times
        spikes2 = self.lick_times

        spike_times = np.hstack((spikes1, spikes2))

        idx = np.argsort(spike_times)
        spike_times = spike_times[idx]

        spike_clusters = np.int32(np.hstack((np.zeros(spikes1.shape[0]),
                                             np.zeros(spikes2.shape[0]) + 1)))
        #
        spike_clusters = np.int32(spike_clusters[idx])

        soft_assignment = np.ones(spike_times.shape[0])

        corr = correlograms(spike_times,
                            spike_clusters,
                            soft_assignment,
                            cluster_ids=np.arange(2),
                            sample_rate=1,
                            bin_size=1,
                            window_size=self.window_size)

        plt.figure()
        titles = ['reward', 'lick']
        t = np.arange(corr.shape[2]) - corr.shape[2] // 2
        for k in range(2):
            for p in range(k,2,1):
                plt.subplot(2, 2, k*2+p + 1)
                plt.plot(t, corr[k, p], label=titles[k] + " vs " + titles[p])
                plt.legend()
                plt.xlabel("Time (sec)")
                plt.xlim(t[0], t[-1])
                plt.ylim(bottom=0)
                plt.plot([0,0],
                         [0,np.max(corr[k,p]) ],
                         '--',c='grey')
        plt.show()

        np.save(os.path.join(self.save_dir, 'reward_vs_licking.npy'), corr)
        plt.savefig(os.path.join(self.save_dir,'reward_vs_licking.png'),dpi=200)

        print("DONE...")


    def compute_intra_session_inter_burst_interval(self):

        #names = ['roi1','roi2','roi3','roi4']
        names = ["roi1", "roi2", "roi3", "roi4"]
        clrs = ['blue','lightblue','red','pink']

        plt.figure()
        burst_array = []
        isi_array = []
        for k in range(4):
            temp = self.F_upphase_bin[k]
            diffs = temp[1:]-temp[:-1]
            # detect exactly when the upphase goes on
            bursts = np.where(diffs==1)[0]/float(self.sample_rate)/60.

            # detect interburst interval
            idx_b = bursts[1:]-bursts[:-1]

            # do a histogram over the ISI-bursts
            y = np.histogram(idx_b,
                             bins=np.arange(0, self.isi_width, self.isi_bin_width))
            plt.plot(y[1][:-1]+self.isi_bin_width/2.,
                     y[0],
                     label=names[k],
                     linewidth=5,
                     color=clrs[k])
            isi_array.append(y[0])

        plt.xlim(y[1][0],y[1][-1])
        plt.legend()


        plt.ylabel("# of bursts")
        #
        plt.suptitle(self.animal_id + " -- " + self.session_id)
        plt.xlabel("Time (mins)")
        plt.show()

        np.save(os.path.join(self.save_dir, 'cell_isis.npy'), isi_array)
        plt.savefig(os.path.join(self.save_dir,'cell_isis.png'),dpi=200)




    def compute_intra_session_cell_burst_histogram_v2(self):

        #names = ['roi1','roi2','roi3','roi4']
        names = ["roi1", "roi2", "roi3", "roi4"]
        clrs = ['blue','lightblue','red','pink']

        plt.figure()
        burst_array = []
        for k in range(4):
            temp = self.F_upphase_bin[k]
            diffs = temp[1:]-temp[:-1]
            bursts = np.where(diffs==1)[0]/float(self.sample_rate)/60.

            y = np.histogram(bursts,
                             bins=np.arange(0, self.rec_len_mins+self.bin_width, self.bin_width))
            plt.plot(y[1][:-1]+self.bin_width/2.,
                     y[0],
                     label=names[k],
                     linewidth=5,
                     color=clrs[k])
            burst_array.append(y[0])

        plt.xlim(y[1][0],y[1][-1])
        plt.legend()

        plt.ylabel("# of bursts")
        #
        plt.suptitle(self.animal_id + " -- " + self.session_id)
        plt.xlabel("Time (mins)")
        plt.show()

        np.save(os.path.join(self.save_dir, 'cell_burst_histogram_v2.npy'), burst_array)
        plt.savefig(os.path.join(self.save_dir,'cell_burst_histogram_v2.png'),dpi=200)

    def compute_intra_session_cell_burst_histogram(self):

        plt.figure()
        names = ['roi1','roi2','roi3','roi4']
        clrs = ['blue','red']
        burst_array = []
        for k in range(4):
            plt.subplot(4,1,k+1)
            temp = self.F_upphase_bin[k]
            diffs = temp[1:]-temp[:-1]
            bursts = np.where(diffs==1)[0]/float(self.sample_rate)/60.

            y = np.histogram(bursts, bins=np.arange(0, 65, self.bin_width))

            plt.bar(y[1][:-1], y[0], self.bin_width * 0.9, label=names[k],
                    color=clrs[k//2])

            burst_array.append(y[0])
            if k!=3:
                plt.xticks([])
            plt.ylabel("# of bursts")
            plt.legend()

        #
        plt.suptitle(self.animal_id + " -- " + self.session_id)
        plt.xlabel("Time (mins)")
        plt.show()

        np.save(os.path.join(self.save_dir, 'cell_burst_histogram.npy'), burst_array)
        plt.savefig(os.path.join(self.save_dir,'cell_burst_histogram.png'),dpi=200)


    #
    def compute_intra_session_reward_histogram(self):

        rs = self.reward_times / self.sample_rate/60.

        y = np.histogram(rs,
                        bins = np.arange(0, self.rec_len_mins + self.bin_width, self.bin_width))

        #
        xx = y[1][:-1]+self.bin_width/2.
        yy = y[0]


        #


        from scipy import stats
        res = stats.pearsonr(xx,yy)
        print ("Perason corr: ", res)

        #

        #
        plt.figure()
        plt.plot(np.unique(xx), np.poly1d(np.polyfit(xx, yy, 1))(np.unique(xx)),
                 '--')
        plt.scatter(xx,yy, label = "pcorr: "+str(round(res[0],2))+ ", pval: "+str(round(res[1],5)))
        plt.bar(xx,yy,self.bin_width*0.9, alpha=.5)
        plt.ylim(bottom=0)
        plt.xlim(y[1][0],y[1][-1])
        plt.legend()

        plt.xlabel("Time (mins)")
        plt.ylabel("# of rewards")
        plt.title(self.animal_id +  " -- " + self.session_id)
        plt.show()

        #
        plt.savefig(os.path.join(self.save_dir,'intra_session_reward.png'),dpi=200)
        np.save(os.path.join(self.save_dir,'intra_session_reward.npy'),y)

    #
    def compute_correlograms_ensembles_upphase(self):

        from correlograms_phy import correlograms

        self.sample_rate = 30  # in Hz
        self.bin_size = 1  # in seconds

        # rewarded times
        std_threshold = 5

        #
        self.spikes_E10 = np.where(self.F_upphase_bin[0] == 1)[0]

        #
        self.spikes_E11 = np.where(self.F_upphase_bin[1] == 1)[0]

        #
        self.spikes_E20 = np.where(self.F_upphase_bin[2] == 1)[0]

        #
        self.spikes_E21 = np.where(self.F_upphase_bin[3] == 1)[0]


        # MAKE SPIKE TIMES

        #self.E1_spikes = self.ttl_times[self.reward_times]
        spike_times = np.hstack((
            self.spikes_E10,
            self.spikes_E11,
            self.spikes_E20,
            self.spikes_E21))


        # sort them for the correlogram function below
        idx = np.argsort(spike_times)
        spike_times = spike_times[idx]

        spike_times = spike_times/self.sample_rate
        print ("spike times: ", spike_times)

        # MAKE SPIKE CLUSTERS
        spike_clusters = np.int32(np.hstack((
            np.zeros(self.spikes_E10.shape[0]),
            np.zeros(self.spikes_E11.shape[0]) + 1,
            np.zeros(self.spikes_E20.shape[0]) + 2,
            np.zeros(self.spikes_E21.shape[0]) + 3)
        ))

        spike_clusters = np.int32(spike_clusters[idx])

        # THIS IS NOT REQUIRED ?!
        soft_assignment = np.ones(spike_times.shape[0])

        # RUN FUNCTION
        corr = correlograms(spike_times,
                            spike_clusters,
                            soft_assignment,
                            cluster_ids=np.arange(4),
                            sample_rate=self.sample_rate,
                            bin_size=self.bin_size,
                            window_size=self.corr_window)

        print(corr.shape)
        plt.figure(figsize=(15,10))
        titles = ['roi1', 'roi2', 'roi3', 'roi4']
        t = np.arange(corr.shape[2]) - corr.shape[2] // 2
        t = t*self.bin_size

        #
        for k in range(4):
            for p in range(k,4,1):
                plt.subplot(4, 4, k*4 + p+1)
                plt.plot(t, corr[k, p], label=titles[k] + " vs " + titles[p])
                plt.legend()
                plt.xlabel("Time (sec)")
                plt.xlim(t[0], t[-1])
                plt.ylim(bottom=0)
        plt.show()

        plt.savefig(os.path.join(self.save_dir,'correlograms_upphase.png'),dpi=200)
        np.save(os.path.join(self.save_dir,'correlograms_upphase.npy'),corr)


        print("DONE...")


    #
    def compute_correlograms_ensembles_fluorescence(self):

        from scipy import stats

        self.sample_rate = 30  # in Hz
        self.bin_size = 1  # in seconds

        # rewarded times
        std_threshold = 5
        names = ["roi1", "roi2","roi3","roi4",]

        #
        plt.figure()
        t=np.arange(-self.window, self.window, 1)
        cc_array = []
        for k in range(4):
            cc_array.append([])
            for p in range(4):
                cc_array[k].append([])
        for k in range(4):
            for p in range(k,4,1):
                t1 = self.F_filtered[k]
                t2 = self.F_filtered[p]

                #
                cc = []
                for z in range(-self.window, self.window,1):
                    cc.append(stats.pearsonr(np.roll(t1,z), t2)[0])

                cc_array[k][p]=cc
                #
                #print (k*4+p)
                plt.subplot(4,4,k*4+p+1)
                plt.plot(t, cc)

                plt.title(names[k] + " vs " +names[p])
                plt.xlim(t[0],t[-1])
                if p!=k:
                    plt.xticks([])
                else:
                    plt.xlabel("Time (sec)")
                plt.ylim(bottom=0)
                plt.plot([0,0],
                         [0,np.max(cc) ],
                         '--',c='grey')


        plt.suptitle(self.animal_id + " " + self.session_id + " Raw fluorescence based xcorrelation")
        plt.show()
        plt.savefig(os.path.join(self.save_dir,'correlograms_fluorescence.png'),dpi=200)
        np.save(os.path.join(self.save_dir,'correlograms_fluorescence.npy'),cc_array, allow_pickle=True)

        print("DONE...")

    #
    def binarize_ensembles(self):

        from calcium import Calcium

        c = Calcium()
        c.data_dir = os.path.join(self.root_dir,
                                  self.animal_id,
                                  self.session_id)

        # c.detrend_model_order = 1
        c.save_python = True
        c.save_matlab = False
        c.recompute_binarization = True
        c.sample_rate = 30

        #
        c.min_width_event_onphase = c.sample_rate // 2  # set minimum withd of an onphase event; default: 0.5 seconds
        c.min_width_event_upphase = c.sample_rate // 4  # set minimum width of upphase event; default: 0.25 seconds

        ############# PARAMTERS TO TWEAK ##############
        #     1. Cutoff for calling somthing a spike:
        #        This is stored in: std_Fluorescence_onphase/uppohase: defaults: 1.5
        #                                        higher -> less events; lower -> more events
        #                                        start at default and increase if data is very noisy and getting too many noise-events
        c.min_thresh_std_onphase = 2.5              # set the minimum thrshold for onphase detection; defatul 2.5
        c.min_thresh_std_upphase = self.std_upphase  # set the minimum thershold for uppohase detection; default: 2.5

        #     2. Filter of [Ca] data which smooths the data significantly more and decreases number of binarzied events within a multi-second [Ca] event
        #        This is stored in high_cutoff: default 0.5 to 1.0
        #        The lower we set it the smoother our [Ca] traces and less "choppy" the binarized traces (but we loose some temporal precision)
        c.high_cutoff = 0.5

        #     3. Removing bleaching and drift artifacts using polynomial fits
        #        This is stored in detrend_model_order
        c.detrend_model_order = 1  # 1-5 polynomial fit

        #
        traces = np.vstack((self.E1[0],
                            self.E1[1],
                            self.E2[0],
                            self.E2[1])).astype('float32')

        #
        c.F = traces
        c.binarize_fluorescence()
        c.load_binarization()

        print("Binarized traces: ", c.F_upphase_bin.shape)

        self.F_upphase_bin = c.F_upphase_bin
        self.F_filtered = c.F_filtered

        ################################################
        ############### SIMPLE VIS TEST ################
        ################################################
        plt.figure(figsize=(15,10))
        Ensembles = [
            self.E1[0],
            self.E1[1],
            self.E2[0],
            self.E2[1],
                     ]
        names = ["roi1", "roi2","roi3","roi4",]
        clrs=['blue','red']
        for k in range(4):
            plt.subplot(4,1,k+1)
            t=np.arange(self.F_filtered.shape[1])/30.
            plt.plot(t,self.F_upphase_bin[k], c=clrs[k//2],alpha=.5)
            plt.plot(t,self.F_filtered[k],c='black',alpha=.5,label=names[k])
            plt.legend()
            plt.xlim(t[0],t[-1])

        plt.suptitle("Using STD for upphase detection: "+str(self.std_upphase))
        plt.show()

        plt.savefig(os.path.join(self.save_dir,'binarized_traces.png'),dpi=200)
        np.save(os.path.join(self.save_dir,'binarized_traces.npy'),self.F_upphase_bin)

