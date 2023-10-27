import numpy as np
# Visualisation
import matplotlib.pyplot as plt
import numpy
import os
import pandas as pd
from tqdm import tqdm, trange
import yaml

import scipy.ndimage
#from matplotlib_scalebar.scalebar import ScaleBar


def smooth_ca_time_series4(diff):
	#
	''' This returns the last value. i.e. no filter

	'''

	temp = (diff[-1]*0.4+
			diff[-2] * 0.25 +
			diff[-3] * 0.15 +
			diff[-4] * 0.10 +
			diff[-5] * 0.10)

	return temp

#
class ProcessCohort():

    def __init__(self, root_dir, animal_ids, cohort_name):

        #
        self.root_dir = root_dir
        self.animal_ids = animal_ids
        self.cohort_name = cohort_name

    def load_animals(self):

        for animal_id in self.animal_ids:
            sessions = np.load(os.path.join(
                                self.root_dir,
                                animal_id,
                                'session_names.npy'
                                ))
            print (animal_id, sessions)

    #
    def cohort_hit_rate(self):

        #
        hit_rates = []
        rec_types = []
        for animal_id in self.animal_ids:


            S = ProcessSession(self.root_dir,
                                 animal_id)
            
            #
            rec_types.append(S.rec_type)
            
            #
            hit_array = []        
            for session_id in S.session_ids[1:]:
                d = np.load(os.path.join(self.root_dir,
                                            animal_id,
                                            session_id,
                                            'results/intra_session_reward_hits_per_minute.npy'))
                hit_array.append(np.nanmean(d))
            
            #
            hit_array = hit_array[:8]

            #
            hit_rates.append(hit_array)


        # plot each line from hit_rates
        plt.figure(figsize=(14,7.5))
        ax1=plt.subplot(1,2,1)
        ax2=plt.subplot(1,2,2)

        line_styles = ['-', '--', ':', '-.', (0, (5, 2)), (0, (1, 1)), (0, (3, 1, 1, 1)), (0, (3, 2, 1, 2))]
        
        clrs = ['black','red']
        clrs_m1 = ['green','darkgreen','lightseagreen','blue','navy','royalblue','darkblue','lightblue']
        clrs_ca3 = ['red','darkred','lightcoral','orange','gold','pink']
        clrs = ['black','red', 'blue', 'green', 'orange', 'purple', 'pink', 'brown']
        ctr_m1 = 0
        ctr_ca3 = 0
        
        #
        ave_m1 = []
        ave_ca3 = []

        for k in range(len(hit_rates)):
            x = np.arange(len(hit_rates[k]))
            
            temp = hit_rates[k]

            # check if tepm has length 8 and if not add a nan
            while len(temp)<8:
                temp = np.hstack((temp,np.nan))
                            #
            if rec_types[k][0]=='m1':
                #
                ave_m1.append(temp)

                ax1.plot(x,hit_rates[k],
                        alpha=.7,
                        linewidth = 5,
                        linestyle = line_styles[ctr_m1],
                        c=clrs_m1[ctr_m1],
                        label = self.animal_ids[k] + " " + rec_types[k][0]
                        )
                ctr_m1+=1
            elif rec_types[k][0]=='ca3':
                ave_ca3.append(temp)
                
                ax2.plot(x,hit_rates[k],
                        alpha=.7,
                        linewidth = 5,
                        linestyle = line_styles[ctr_ca3],
                        c=clrs_ca3[ctr_ca3],
                        label = self.animal_ids[k] + " " + rec_types[k][0]
                        )
                ctr_ca3+=1
            else:
                print ("cant find rec type", rec_types[k][0])
        
        # plot the averages for ave_m1
        ave_m1 = np.vstack(ave_m1)
        mean = np.nanmean(ave_m1, axis=0)
        std = np.nanstd(ave_m1, axis=0)

        # plot the average and standard deviation as error lines
        ax1.plot(x,mean,
                alpha=1,
                linewidth = 5,
                linestyle = '-',
                c='black',
                label = "m1 average",
                )
        
        # plot error bars based on std
        ax1.fill_between(x, mean-std, mean+std, alpha=0.2, color='black')

        # same for ave_ca3
        ave_ca3 = np.vstack(ave_ca3)
        mean = np.nanmean(ave_ca3,axis=0)
        std = np.nanstd(ave_ca3, axis=0)
        
        # plot the average and standard deviation as error lines
        ax2.plot(x,mean,
                alpha=1,
                linewidth = 5,
                linestyle = '-',
                c='black',
                label = "ca3 average"
                )
        
        # plot error bars based on std
        ax2.fill_between(x, mean-std, mean+std, alpha=0.2, color='black')
        
        #
        ax1.legend()
        ax1.set_ylim(0.1,0.8)

        ax2.legend()
        ax2.set_ylim(0.1,0.8)
 
        # write xticks as session numbers
        xticks = []
        for k in range(8):
            xticks.append(str(k+1))
        ax1.set_xticks(x)
        ax1.set_xticklabels(xticks)
        ax1.set_xlabel("Session #", fontsize=20)
        ax1.set_ylabel("% hit rate", fontsize=20)
        ax1.set_title("M1", fontsize=20)

        # add shading horizontal at 25% of thickness 5%
        ax1.axhspan(0.225, 0.275, alpha=0.2, color='black')

        # increase font sizes
        ax1.tick_params(axis='both', which='major', labelsize=20)


        ax2.set_xticks(x)
        ax2.set_xticklabels(xticks)
        ax2.set_xlabel("Session #", fontsize=20)
        ax2.set_ylabel("% hit rate", fontsize=20)
        ax2.set_title("CA3", fontsize=20)

        # add shading horizontal at 25%
        ax2.axhspan(0.225, 0.275, alpha=0.2, color='black')

        # increase font sizes
        ax2.tick_params(axis='both', which='major', labelsize=20)

        # 

        plt.suptitle(self.cohort_name + " hit-rate", fontsize=20)

        plt.savefig(os.path.join(self.root_dir,
                                    'cohort_hit_rates.png'), dpi=200)
        

        # xticks=['early','late']
        # plt.xticks(x,xticks)
        # plt.ylabel("% hit rate")
        # plt.suptitle(self.cohort_name + " hit-rate")


        plt.savefig(os.path.join(self.root_dir,
                                 'cohort_hit_rates.png'), dpi=200)

        if self.show_plots:
            plt.show()
        else:
            plt.close()

        np.save(os.path.join(self.root_dir,
                            'cohort_hit_rates.npy'), hit_rates)

    #
    def cohort_hit_rate_early_vs_late_ca3_vs_m1(self):
        
        #
        hit_rates_m1 = []
        hit_rates_ca3 = []
        rec_types = []
        for animal_id in self.animal_ids:

            #
            S = ProcessSession(self.root_dir,
                                 animal_id)
            
            #
            rec_types.append(S.rec_type)
            
            #
            hit_array = []        
            for session_id in S.session_ids[1:9]:

                try:
                    hits_per_5min = np.load(os.path.join(self.root_dir,
                                            animal_id,
                                            session_id,
                                            'results',
                                            'intra_session_reward_hits_per_minute.npy'))

                #                    
                except:
                    print ('missing...')
                    hits_per_5min = np.zeros(10)+np.nan


                hit_array.append(hits_per_5min)
            
            #
            if S.rec_type[0]=='m1':
                hit_rates_m1.append(hit_array)
            elif S.rec_type[0]=='ca3':
                hit_rates_ca3.append(hit_array)


        # loop over animals now
        plt.figure(figsize=(25,10))
        plt.suptitle("Cohort hit rate first 2 sessions vs. last 2 sessions", fontsize=20)

        # grab the first 2 and last 2 sessions of each of the hit_rates_m1 sessions
        plt.subplot(121)
        plt.title("M1 mice")
        early_m1 = []
        late_m1 = []
        for k in range(len(hit_rates_m1)):
            early_m1.append(np.hstack((hit_rates_m1[k][0], hit_rates_m1[k][1])))
            late_m1.append(np.hstack((hit_rates_m1[k][-2], hit_rates_m1[k][-1])))

        # plot scatter plot of early and late
        plt.scatter(np.zeros(np.hstack(early_m1).shape[0]), np.hstack(early_m1),
                    edgecolor='black',
                    c='mediumturquoise', label="early")
        plt.scatter(np.zeros(np.hstack(late_m1).shape[0])+1, np.hstack(late_m1),
                    edgecolor='black',
                    c='royalblue',
                    label="late")
        
        # plot the means as bars
        plt.bar(0, np.mean(np.hstack(early_m1)),
                width=0.9,
                color='mediumturquoise', alpha=.5, edgecolor='black',linewidth=5)
        plt.bar(1, np.mean(np.hstack(late_m1)),
                width=0.9,
                color='royalblue', alpha=.5, edgecolor='black',linewidth=5)
        
        # do a 2 sample ks test on early and late distributions
        res_ks = scipy.stats.ks_2samp(np.hstack(early_m1), np.hstack(late_m1))

        # plot the ks test results as an addition to the legend
        plt.legend(title="ks test: "+str(round(res_ks[0],3))+
                    ", ks pval: "+str(round(res_ks[1],5)))
        
        ##################################################
        # grab the first 2 and last 2 sessions of each of the hit_rates_ca3 sessions
        plt.subplot(122)
        plt.title("CA3 mice")
        early_ca3 = []
        late_ca3 = []
        for k in range(len(hit_rates_ca3)):
            early_ca3.append(np.hstack((hit_rates_ca3[k][0], hit_rates_ca3[k][1])))
            late_ca3.append(np.hstack((hit_rates_ca3[k][-2], hit_rates_ca3[k][-1])))

        # plot scatter plot of early and late
        plt.scatter(np.zeros(np.hstack(early_ca3).shape[0]), np.hstack(early_ca3),
                    edgecolor='black',
                    c='mediumturquoise', label="early")
        plt.scatter(np.zeros(np.hstack(late_ca3).shape[0])+1, np.hstack(late_ca3),
                    edgecolor='black',
                    c='royalblue',
                    label="late")
        
        # plot the means as bars
        plt.bar(0, np.mean(np.hstack(early_ca3)),
                width=0.9,
                color='mediumturquoise', alpha=.5, edgecolor='black',linewidth=5)
        plt.bar(1, np.mean(np.hstack(late_ca3)),
                width=0.9,
                color='royalblue', alpha=.5, edgecolor='black',linewidth=5)
        
        # do a 2 sample ks test on early and late distributions
        res_ks = scipy.stats.ks_2samp(np.hstack(early_ca3), np.hstack(late_ca3))
        
        # plot the ks test results as an addition to the legend
        plt.legend(title="ks test: "+str(round(res_ks[0],3))+
                    ", ks pval: "+str(round(res_ks[1],5)))
        
        


    #
    def cohort_hit_rate_early_vs_late(self):

        #
        hit_rates = []
        rec_types = []
        for animal_id in self.animal_ids:


            S = ProcessSession(self.root_dir,
                                 animal_id)
            
            #
            rec_types.append(S.rec_type)
            
            #
            hit_array = []        
            for session_id in S.session_ids[1:9]:

                try:
                    hits_per_5min = np.load(os.path.join(self.root_dir,
                                            animal_id,
                                            session_id,
                                            'results',
                                            'intra_session_reward_hits_per_minute.npy'))

                #                    
                except:
                    print ('missing...')
                    hits_per_5min = np.zeros(10)+np.nan


                hit_array.append(hits_per_5min)
            
            #
            hit_array = hit_array

            #
            hit_rates.append(hit_array)

        #
        print ("# of hit rate arrays ", len(hit_rates))
        print ("example hit rate array ", hit_rates[0][0])

        # loop over animals now
        plt.figure(figsize=(25,10))
        plt.suptitle("Cohort hit rate first 2 sessions vs. last 2 sessions", fontsize=20)
        for k in range(len(hit_rates)):
            
            #
            plt.subplot(2,5,k+1)

            # plot title as animal_id and also rec type
            plt.title(self.animal_ids[k] + " " + rec_types[k][0])

            # plot yalbel as hit rate
            plt.ylabel("% hit rate", fontsize=10)


            # grab the first 2 and last 2 sessions
            early = np.hstack((hit_rates[k][0], hit_rates[k][1]))
            late = np.hstack((hit_rates[k][-2], hit_rates[k][-1]))

            # do a 2 sample ks test on early and late distributions
            res_ks = scipy.stats.ks_2samp(early, late)


            # plot early as scatter plot and mean as bar
            plt.scatter(np.zeros(early.shape[0]), early,
                        edgecolor='black',
                        c='mediumturquoise', label="early")
            plt.bar(np.zeros(1), np.mean(early),
                    width=0.9,
                    color='mediumturquoise', alpha=.5, edgecolor='black',linewidth=5)
            
            # plot late as scatter plot and mean as bar
            plt.scatter(np.zeros(late.shape[0])+1, late,
                        edgecolor='black',
                        c='royalblue', 
                        label="late")
            plt.bar(1, np.mean(late), width=0.9,
                    color='royalblue', alpha=.5, edgecolor='black',linewidth=5)
            
            # plot the ks test results as an addition to the legend
            plt.legend(title="ks test: "+str(round(res_ks[0],3))+
                        ", ks pval: "+str(round(res_ks[1],5)))

    #
    def cohort_rewards_per_minute(self):

        #
        n_mins_per_rec =50

        #
        hit_rates = []
        rec_types = []
        for animal_id in self.animal_ids:


            S = ProcessSession(self.root_dir,
                                 animal_id)
            
            #
            rec_types.append(S.rec_type)
            
            #
            hit_array = []        
            for session_id in S.session_ids[1:9]:

                try:
                    data = np.load(os.path.join(self.root_dir,
                                            animal_id,
                                            session_id,
                                            'data',
                                            'results.npz'), allow_pickle=True)
                    d = data['rewarded_times_abs'][:,1]
                except:
                    d = np.empty(0)


                hit_array.append(d.shape[0]/n_mins_per_rec)
            
            #
            hit_array = hit_array[:8]

            #
            hit_rates.append(hit_array)


        # plot each line from hit_rates
        plt.figure(figsize=(14,7.5))
        ax1=plt.subplot(1,2,1)
        ax2=plt.subplot(1,2,2)

        line_styles = ['-', '--', ':', '-.', (0, (5, 2)), (0, (1, 1)), (0, (3, 1, 1, 1)), (0, (3, 2, 1, 2))]
        
        clrs = ['black','red']
        clrs_m1 = ['green','darkgreen','lightseagreen','blue','navy','royalblue','darkblue','lightblue']
        clrs_ca3 = ['red','darkred','lightcoral','orange','gold','pink']
        clrs = ['black','red', 'blue', 'green', 'orange', 'purple', 'pink', 'brown']
        ctr_m1 = 0
        ctr_ca3 = 0
        
        #
        ave_m1 = []
        ave_ca3 = []

        for k in range(len(hit_rates)):
            x = np.arange(len(hit_rates[k]))
            
            temp = hit_rates[k]

            # check if tepm has length 8 and if not add a nan
            while len(temp)<8:
                temp = np.hstack((temp,np.nan))
                            #
            if rec_types[k][0]=='m1':
                #
                ave_m1.append(temp)

                ax1.plot(x,hit_rates[k],
                        alpha=.7,
                        linewidth = 5,
                        linestyle = line_styles[ctr_m1],
                        c=clrs_m1[ctr_m1],
                        label = self.animal_ids[k] + " " + rec_types[k][0]
                        )
                ctr_m1+=1
            elif rec_types[k][0]=='ca3':
                ave_ca3.append(temp)
                
                ax2.plot(x,hit_rates[k],
                        alpha=.7,
                        linewidth = 5,
                        linestyle = line_styles[ctr_ca3],
                        c=clrs_ca3[ctr_ca3],
                        label = self.animal_ids[k] + " " + rec_types[k][0]
                        )
                ctr_ca3+=1
            else:
                print ("cant find rec type", rec_types[k][0])
        
        # plot the averages for ave_m1
        ave_m1 = np.vstack(ave_m1)
        mean = np.nanmean(ave_m1, axis=0)
        std = np.nanstd(ave_m1, axis=0)

        # plot the average and standard deviation as error lines
        ax1.plot(x,mean,
                alpha=1,
                linewidth = 5,
                linestyle = '-',
                c='black',
                label = "m1 average",
                )
        
        # plot error bars based on std
        ax1.fill_between(x, mean-std, mean+std, alpha=0.2, color='black')

        # same for ave_ca3
        ave_ca3 = np.vstack(ave_ca3)
        mean = np.nanmean(ave_ca3,axis=0)
        std = np.nanstd(ave_ca3, axis=0)
        
        # plot the average and standard deviation as error lines
        ax2.plot(x,mean,
                alpha=1,
                linewidth = 5,
                linestyle = '-',
                c='black',
                label = "ca3 average"
                )
        
        # plot error bars based on std
        ax2.fill_between(x, mean-std, mean+std, alpha=0.2, color='black')
        
        #
        ax1.legend()
        #ax1.set_ylim(0.1,0.8)

        ax2.legend()
        #ax2.set_ylim(0.1,0.8)
 
        # write xticks as session numbers
        xticks = []
        for k in range(8):
            xticks.append(str(k+1))
        ax1.set_xticks(x)
        ax1.set_xticklabels(xticks)
        ax1.set_xlabel("Session #", fontsize=20)
        ax1.set_ylabel("# rewards per minute", fontsize=20)
        ax1.set_title("M1", fontsize=20)

        # increase font sizes
        ax1.tick_params(axis='both', which='major', labelsize=20)


        ax2.set_xticks(x)
        ax2.set_xticklabels(xticks)
        ax2.set_xlabel("Session #", fontsize=20)
        ax2.set_ylabel("# rewards per minute", fontsize=20)
        ax2.set_title("CA3", fontsize=20)

        # increase font sizes
        ax2.tick_params(axis='both', which='major', labelsize=20)

        # 

        plt.suptitle(self.cohort_name + " hit-rate", fontsize=20)

        plt.savefig(os.path.join(self.root_dir,
                                    'cohort_hit_rates.png'), dpi=200)


        plt.savefig(os.path.join(self.root_dir,
                                 'cohort_hit_rates.png'), dpi=200)

        if self.show_plots:
            plt.show()
        else:
            plt.close()

        np.save(os.path.join(self.root_dir,
                            'cohort_hit_rates.npy'), hit_rates)

    # #
    # def cohort_hit_rate_early_vs_late(self):

    #     #
    #     early_rates = []
    #     late_rates = []
    #     for animal_id in self.animal_ids:
    #         d = np.load(os.path.join(self.root_dir,
    #                                     animal_id, 'intra_session_reward_hits_per_minute'))
    #         early_rates.append(d[0].mean())
    #         late_rates.append(d[1].mean())

    #     early = np.hstack(early_rates)
    #     late = np.hstack(late_rates)

    #     #
    #     res_ttest = scipy.stats.ttest_ind(early, late)
    #     res_ks = scipy.stats.ks_2samp(early, late)

    #     #
    #     x = np.arange(2)
    #     #
    #     plt.bar(0,np.mean(early),width=0.9,
    #             color='mediumturquoise',alpha=1, edgecolor='black',linewidth=5)

    #     plt.scatter(np.zeros(early.shape[0]), early,
    #                 edgecolor='black',
    #                 c='mediumturquoise', label="ttest statistic: "+str(round(res_ttest[0],3))+
    #                 ", ttest pval: "+str(round(res_ttest[1],5)))

    #     ################################
    #     plt.bar(1, np.mean(late), width=0.9,
    #             color='royalblue', alpha=1, edgecolor='black',linewidth=5)
    #     plt.scatter(np.zeros(early.shape[0])+1, late,edgecolor='black',
    #                 c='royalblue', label="ks test: "+str(round(res_ks[0],3))+
    #                 ", ks pval: "+str(round(res_ks[1],5)))

    #     plt.legend()



    #     #
    #     xticks=['early','late']
    #     plt.xticks(x,xticks)
    #     plt.ylabel("% hit rate")
    #     plt.suptitle(self.cohort_name + " hit-rate")


    #     plt.savefig(os.path.join(self.root_dir,
    #                              'early_vs_late_cohort.png'), dpi=200)

    #     if self.show_plots:
    #         plt.show()
    #     else:
    #         plt.close()

    #     np.save(os.path.join(self.root_dir,
    #                         'early_vs_late_cohort.npy'), [early,late])

#
class ProcessSession():

    def __init__(self,
                 root_dir,
                 animal_id,
                 session_id=''):

        #
        self.root_dir = root_dir

        #
        self.animal_id = animal_id

        #
        self.sample_rate = 30

        ##
        self.save_dir = os.path.join(self.root_dir,
                                     self.animal_id,
                                     'results')

        # load yaml file
        import yaml
        with open(os.path.join(self.root_dir,
                               self.animal_id,
                               self.animal_id+'.yaml')) as file:
            doc = yaml.load(file, Loader=yaml.FullLoader)

        self.session_ids = np.array(doc['session_names'],dtype='str')

        #
        self.rec_type = np.array(doc['anatomy'],dtype='str')
        
        
        # default show plots
        self.show_plots = True

        #
        self.verbose = False

        #
        if os.path.exists(self.save_dir)==False:
            os.mkdir(self.save_dir)

    # 
    def plot_rewarded_ensembles(self):

        #
        plt.figure(figsize=(14,8))
        ####################################################
        ################# VISUALIZE ROIS ###################
        ####################################################
        ax=plt.subplot(2,1,1)
        plt.title("ROIs")
        t = np.arange(self.diff.shape[0]) / self.sample_rate
        #plt.plot([t[0], t[-1]], [self.low, self.low], '--', c='grey')#, label='Low threshold')
        plt.plot([t[0], t[-1]], [self.high, self.high], '--', c='grey')#, label='high threshold')


        # show rois
        #clrs = ['lightblue','darkblue','lightcoral','red']
        #names = ["Roi #1","Roi #2","Roi #3","Roi #4",]
        # ensembel 1
        clrs=['blue','lightblue']
        for k in range(len(self.ensemble1_traces_smooth)):
            plt.plot(t, self.ensemble1_traces_smooth[k],
                             c=clrs[k], alpha=.8)

        #
        clrs = ['red','pink']
        for k in range(len(self.ensemble2_traces_smooth)):
            plt.plot(t, self.ensemble2_traces_smooth[k],
                             c=clrs[k], alpha=.8)

        #
        #plt.plot(t, self.diff, c='black', alpha=1, label='Global ensemble state (i.e. E1-E2)')
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
        for k in range(len(self.white_noise)):
            #print(self.white_noise[k])
            try:
                ax.axvspan(t[self.white_noise[k][0]],
                       t[self.white_noise[k][1]],
                       alpha=0.2,
                       color='grey')
            except:
                pass
        ####################################################
        ################## VISUALIZE ENSEMBELS #############
        ####################################################
        ax=plt.subplot(2,1,2)
        plt.title("ENSEMBLES")
        #plt.plot([t[0], t[-1]], [self.low, self.low], '--', c='grey')#, label='Low threshold')
        plt.plot([t[0], t[-1]], [self.high, self.high], '--', c='grey')#', label='high threshold')
        #plt.plot(t, self.E1, c='darkblue', alpha=1, label='E1')
        #plt.plot(t, self.E2, c='darkred', alpha=1, label='E2')
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

        for k in range(len(self.white_noise)):
            #print(self.white_noise[k])
            try:
                ax.axvspan(t[self.white_noise[k][0]],
                       t[self.white_noise[k][1]],
                       alpha=0.2,
                       color='grey')
            except:
                pass
				
        #
        plt.suptitle("binning: "+str(self.binning_flag) + ", rec time: " + str(int(t[-1])) + " sec " +
                  "\n expected # of random rewards: " + str(int(t[-1] / 30)) +
                  "\n actual # of provided rewards: " + str(self.reward_times.shape[0]))
        plt.xlabel("Time (sec)", fontsize=20)
        
        #
        plt.savefig(os.path.join(self.root_dir,
                                 self.animal_id,
                                 self.session_id,
                                 'results',
                                 "calibration_threshold_plot.png"))
            

        
        plt.show()


    #
    def run_simulated_bmi(self):


        # initialize the max and min values
        #
        n_sec_recording = int(self.ensemble1_traces[0].shape[0] / self.sample_rate)
        n_rewards_random = n_sec_recording // self.sample_rate

        self.fps = 30
        n_rewards = 0
        last_reward = 0
        reward_times = []

        E1_1= np.zeros(self.ensemble1_traces[0].shape[0])
        E1_2= np.zeros(self.ensemble1_traces[1].shape[0])
        E2_1= np.zeros(self.ensemble2_traces[0].shape[0])
        E2_2= np.zeros(self.ensemble2_traces[1].shape[0])

        #
        counter=-1

        #
        post_reward_lockout=False
        binning_n_frames = int(self.binning_time * self.fps)
        binning_counter = binning_n_frames
        white_noise_starts = []
        white_noise_ends = []
        
        # keep track of trials
        trial_starts = []
        trial_ends = []
        trial_starts.append(0)

        # here we bin data
        self.binning_flag = True
        for k in range(binning_n_frames, self.ensemble1_traces[0].shape[0], 1):

            #
            if self.binning_flag:
                if False:
                    # smooth the data
                    if binning_counter<=0:
                        temp_now = np.mean(self.ensemble1_traces[0][k-binning_n_frames:k])
                        temp2 = []
                        # go to previous n time bins and grab data
                        for q in range(self.smoothing_n_bins-1,0,-1):
                            temp2.append(E1_1[k-q*binning_n_frames])
                        self.E1_1 = np.mean(np.hstack((temp2, temp_now)))

                        temp_now = np.mean(self.ensemble1_traces[1][k-binning_n_frames:k])
                        temp2 = []
                        # go to previous n time bins and grab data
                        for q in range(self.smoothing_n_bins-1,0,-1):
                            temp2.append(E1_2[k-q*binning_n_frames])
                        self.E1_2 = np.mean(np.hstack((temp2, temp_now)))

                        temp_now = np.mean(self.ensemble2_traces[0][k-binning_n_frames:k])
                        temp2 = []
                        # go to previous n time bins and grab data
                        for q in range(self.smoothing_n_bins-1,0,-1):
                            temp2.append(E2_1[k-q*binning_n_frames])
                        self.E2_1 = np.mean(np.hstack((temp2, temp_now)))

                        temp_now = np.mean(self.ensemble2_traces[1][k-binning_n_frames:k])
                        temp2 = []
                        # go to previous n time bins and grab data
                        for q in range(self.smoothing_n_bins-1,0,-1):
                            temp2.append(E2_2[k-q*binning_n_frames])
                        self.E2_2 = np.mean(np.hstack((temp2, temp_now)))


                        # reset counter
                        binning_counter = binning_n_frames
                    else:
                        self.E1_1 = E1_1[k-1]
                        self.E1_2 = E1_2[k-1]
                        self.E2_1 = E2_1[k-1]
                        self.E2_2 = E2_2[k-1]
                        binning_counter-=1
                else:
                    self.E1_1 = self.ensemble1_traces[0][k]
                    self.E1_2 = self.ensemble1_traces[1][k]
                    self.E2_1 = self.ensemble2_traces[0][k]
                    self.E2_2 = self.ensemble2_traces[1][k]


            

            # old way of smoothing
            else:
                self.E1_1 = smooth_ca_time_series4(self.ensemble1_traces[0][k - self.rois_smooth_window:k])
                self.E1_2 = smooth_ca_time_series4(self.ensemble1_traces[1][k - self.rois_smooth_window:k])
                self.E2_1 = smooth_ca_time_series4(self.ensemble2_traces[0][k - self.rois_smooth_window:k])
                self.E2_2 = smooth_ca_time_series4(self.ensemble2_traces[1][k - self.rois_smooth_window:k])


            # save data arrays for later visualization
            E1_1[k] = self.E1_1
            E1_2[k] = self.E1_2
            E2_1[k] = self.E2_1
            E2_2[k] = self.E2_2

            #
            self.E1 = self.E1_1+self.E1_2
            self.E2 = self.E2_1+self.E2_2

            temp_diff = self.E1-self.E2

            #
            counter-=1
            if counter>0:
                continue

            if post_reward_lockout:
                if temp_diff > (self.high*self.post_reward_lockout_baseline_min):
                    continue
                else:
                    post_reward_lockout=False


            ############ REWARD REACHED ###########
            if temp_diff >= self.high:
                # high reward state reached
                n_rewards += 1
                reward_times.append([k, 1])
                last_reward = k

                trial_ends.append(k)

                # lock out rewards for some time;
                #k += int(self.post_reward_lockout * self.sample_rate)
                counter = int(self.post_reward_lockout * self.sample_rate)

                #
                post_reward_lockout = True

                #
                trial_starts.append(k+counter)

            ########### WHITE NOISE PENALTY ############
            elif (k-last_reward)>= int(self.trial_time * self.sample_rate):

                last_reward = k
                # lock out rewards for some time;
                counter = int(self.post_missed_reward_lockout * self.sample_rate)
                trial_ends.append(k)
                #
                white_noise_starts.append(k)
                white_noise_ends.append(k+self.post_missed_reward_lockout*self.sample_rate)

                post_reward_lockout = True
                trial_starts.append(k+counter)

        #
        #print("updated rewards #: ", n_rewards, " for threshold: ", self.high)

        #
        self.reward_times = np.vstack(reward_times)
        self.white_noise = np.vstack((white_noise_starts, white_noise_ends)).T
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
        
        self.rewards_in_frame_time=self.reward_times

        #
        self.trial_starts = np.array(trial_starts)
        self.trial_ends = np.array(trial_ends)

        max_len = min(self.trial_starts.shape[0], self.trial_ends.shape[0])

        self.trials = np.vstack((self.trial_starts[:max_len], self.trial_ends[:max_len])).T
        #print ("SELF TRIALS: ", self.trials)
    
    #
    def simulate_session(self):

        # we first load the rois_traces_raw from the results file
        for session_id in self.during:

            d = np.load(os.path.join(self.root_dir,
                                    self.animal_id,
                                    session_id,
                                    'data',
                                    'results.npz')
                                    , allow_pickle=True
                                    )
            
            #
            self.ensemble1_traces = d['rois_traces_smooth1']
            self.ensemble2_traces = d['rois_traces_smooth2']

            # zero out first 10 seconds
            self.ensemble1_traces[:,:int(10*self.sample_rate)] = 0
            self.ensemble2_traces[:,:int(10*self.sample_rate)] = 0


            #
            self.high = d['high_threshold']

            # run the bmit in simulation
            self.run_simulated_bmi()

            # make hits per min array
            self.bin_width_mins = 5
            xx = np.arange(0,self.rec_len_mins,self.bin_width_mins)

            hits_per_bin = np.zeros(xx.shape[0])
            #misses_per_bin = np.zeros(xx.shape[0])
            n_trials_per_bin = np.zeros(xx.shape[0])

            #
            #print ("all rewards frame time: ", self.rewards_in_frame_time)
            #
            for k in range(self.rewards_in_frame_time.shape[0]):
                temp = self.rewards_in_frame_time[k]

                # find bin of the reward
                temp_mins = temp[0]/self.sample_rate/60.
                #print ("reward time in mins: ", temp_mins)
                idx = int(temp_mins/self.bin_width_mins)
                hits_per_bin[idx]+= 1

            # compute number of traisl per bin
            for k in range(self.trials.shape[0]):
                temp = self.trials[k]
                start_min = temp[0]/self.sample_rate/60.
                idx = int(start_min/self.bin_width_mins)

                n_trials_per_bin[idx]+=1

            yy = hits_per_bin/n_trials_per_bin
            self.save_dir = os.path.join(self.root_dir,
                                            self.animal_id,
                                            session_id,
                                            'results')
            np.save(os.path.join(self.save_dir,
                                 'intra_session_reward_hits_per_minute_simulated.npy'),yy)
            
            
            #
            from scipy import stats
            res = stats.pearsonr(xx,yy)
            if self.verbose:
                print ("Pearson corr: ", res)

            #
            plt.figure()
            plt.plot(np.unique(xx), np.poly1d(np.polyfit(xx, yy, 1))(np.unique(xx)),
                    '--')
            plt.scatter(xx,yy, label = "pcorr: "+str(round(res[0],2))+ ", pval: "+str(round(res[1],5)))
            plt.bar(xx,yy,self.bin_width_mins*0.9, alpha=.5)
            plt.ylim(bottom=0)
            plt.xlim(xx[0]-self.bin_width_mins/2.,xx[-1]+self.bin_width_mins/2.)
            plt.legend()

            plt.xlabel("Time (mins)")
            plt.ylabel("% hit rate")
            plt.title(self.animal_id +  " -- " + session_id)
            plt.savefig(os.path.join(self.save_dir,'intra_session_reward_hist_per_minute_simulated.png'),dpi=200)

            if self.show_plots:
                plt.show()
            else:
                plt.close()

            self.session_id = session_id

            #
            self.plot_rewarded_ensembles()
        #

    def contingency_degradation(self):

        # load test data
        print ("Loading pre data")
        pre = []
        for session_id in self.pre:
            d = np.load(os.path.join(self.root_dir,
                                     self.animal_id,
                                     session_id,
                                     'results/intra_session_reward_hits_per_minute.npy'
                                     ))
            pre.append(d.mean())


        # load contingency degradata
        # here we must recmopute the rewards rather than loading them   
        print ("simulating contingency degradation runs")
        self.simulate_session()

        # load contingency degradata
        cont_deg = []
        print ("Loading post data")
        for session_id in self.during:
            d = np.load(os.path.join(self.root_dir,
                                     self.animal_id,
                                     session_id,
                                     'results/intra_session_reward_hits_per_minute.npy'
                                     ))
            print ("cont deg: ", d, " for session: ", session_id)
            cont_deg.append(d.mean())

        #


        # load contingency degradata
        post = []
        print ("Loading post data")
        for session_id in self.post:
            d = np.load(os.path.join(self.root_dir,
                                     self.animal_id,
                                     session_id,
                                     'results/intra_session_reward_hits_per_minute_simulated.npy'
                                     ))
            post.append(d.mean())

        #
        plt.figure()
        xticks = ['T','CD','R']

        #
        x = np.arange(3)
        y = np.array(([np.mean(pre),
                       np.mean(cont_deg),
                       np.mean(post)]))

        plt.bar(x,y,width=0.9)
        plt.xticks(x,xticks)
        plt.ylabel("% hits (per session)")

        plt.savefig(os.path.join(self.root_dir,
                                self.animal_id,
                                 'contingency_degradation.png'), dpi=200)
        plt.suptitle(self.animal_id)
        if self.show_plots:
            plt.show()
        else:
            plt.close()

        np.save(os.path.join(self.root_dir,
                                self.animal_id, 'contingency_degradation.npy'), y)



    def early_vs_late(self):

        # load test data
        early = []
        for session_id in self.early:
            d = np.load(os.path.join(self.root_dir,
                                     self.animal_id,
                                     session_id,
                                     'results/intra_session_reward_hits_per_minute.npy'
                                     ))
            early.append(d)

        early = np.hstack(early)
        #print (pre)

        # load contingency degradata
        late = []
        for session_id in self.late:
            d = np.load(os.path.join(self.root_dir,
                                     self.animal_id,
                                     session_id,
                                     'results/intra_session_reward_hits_per_minute.npy'
                                     ))
            late.append(d)
        late = np.hstack(late)

        #
        plt.figure()
        xticks = ['Early','Late']

        res = scipy.stats.ttest_ind(early, late)
        res_ks = scipy.stats.ks_2samp(early, late)

        #
        x = np.arange(2)
        #
        plt.bar(0,np.mean(early),width=0.9,
                color='mediumturquoise',alpha=1, edgecolor='black',linewidth=5)

        plt.scatter(np.zeros(early.shape[0]), early,
                    edgecolor='black',
                    c='mediumturquoise', label="ttest statistic: "+str(round(res[0],3))+
                    ", ttest pval: "+str(round(res[1],5)))

        ################################
        plt.bar(1, np.mean(late), width=0.9,
                color='royalblue', alpha=1, edgecolor='black',linewidth=5)
        plt.scatter(np.zeros(early.shape[0])+1, late,edgecolor='black',
                    c='royalblue', label="ks test: "+str(round(res_ks[0],3))+
                    ", ks pval: "+str(round(res_ks[1],5)))

        plt.legend()



        #
        plt.xticks(x,xticks)
        plt.ylabel("% hits (every 5min bin)")
        plt.suptitle(self.animal_id)


        plt.savefig(os.path.join(self.root_dir,
                                self.animal_id,
                                 'early_vs_late.png'), dpi=200)

        if self.show_plots:
            plt.show()
        else:
            plt.close()

        np.save(os.path.join(self.root_dir,
                                self.animal_id, 'early_vs_late.npy'), [early,late])


    def early_vs_late_session(self):

        # load test data
        early = []
        for session_id in self.early:
            d = np.load(os.path.join(self.root_dir,
                                     self.animal_id,
                                     session_id,
                                     'results/intra_session_reward_hits_per_minute.npy'
                                     ))
            early.append(d.mean())

        early = np.hstack(early)
        #print (pre)

        # load contingency degradata
        late = []
        for session_id in self.late:
            d = np.load(os.path.join(self.root_dir,
                                     self.animal_id,
                                     session_id,
                                     'results/intra_session_reward_hits_per_minute.npy'
                                     ))
            late.append(d.mean())
        late = np.hstack(late)

        #
        plt.figure()
        xticks = ['Early','Late']

        res = scipy.stats.ttest_ind(early, late)
        res_ks = scipy.stats.ks_2samp(early, late)

        #
        x = np.arange(2)

        #################################
        plt.bar(0, np.mean(early), width=0.9,
                color='mediumturquoise', alpha=1, edgecolor='black',linewidth=5)
        plt.scatter(np.zeros(early.shape[0]), early,
                    edgecolor='black',
                    c='mediumturquoise', label="ttest statistic: "+str(round(res[0],3))+
                    ", ttest pval: "+str(round(res[1],5)))


        ################################
        plt.bar(1, np.mean(late), width=0.9,
                color='royalblue', alpha=1,edgecolor='black',linewidth=5)
        plt.scatter(np.zeros(early.shape[0])+1, late,edgecolor='black',
                    c='royalblue', label="ks test: "+str(round(res_ks[0],3))+
                    ", ks pval: "+str(round(res_ks[1],5)))

        plt.legend()

        #
        plt.xticks(x,xticks)
        plt.ylabel("% hits (per session)")
        plt.suptitle(self.animal_id)


        plt.savefig(os.path.join(self.root_dir,
                                self.animal_id,
                                 'early_vs_late_per_session.png'), dpi=200)
        #
        if self.show_plots:
            plt.show()
        else:
            plt.close()

        np.save(os.path.join(self.root_dir,
                             self.animal_id, 'early_vs_late_per_session.npy'), [early,late])


    def load_data(self):

        ################################################
        ############ LOAD RESULTS NPZ FILE #############
        ################################################
        fname = os.path.join(self.root_dir,
                             self.animal_id,
                             self.session_id,
                             'data', 'results.npz')
        
        #
        print ("Loading data...")
        try:
            data = np.load(fname, allow_pickle=True)
        
            #
            self.reward_times = np.int32(data['rewarded_times_abs'][:, 1])


            self.trials = data['trials']
            #self.starts_ttl = data['trials'][:, 0]
            #self.starts_time = data['trials'][:, 1]
            self.ends_ttl = data['trials'][:, 2]
            #self.ends_time = data['trials'][:, 3]
            self.rewards = data['trials'][:, 4]

            #
            try:
                self.abs_times = data['abs_times_ttl_read']
            except:
                if self.verbose:
                    print ("missing: ", 'abs_times_ttl_read')
                self.abs_times = data['abs_times']

            #print ("self.abs_times")

            try:
                self.ttl_times = data['abs_times_ca_read']
            except:
                if self.verbose:
                    print ("missing: ", 'abs_times_ca_read')
                self.ttl_times = data['ttl_times']


            self.ttl_comp = data['ttl_n_computed']

            #
            self.ttl_det = np.arange(self.ttl_times.shape[0]).astype('int32')

            # TODO: Need to lock this to time correctly
            self.lick_detector = data['lick_detector_abstime']

            idx = np.where(self.lick_detector > 3)[0]
            self.lick_times = self.abs_times[idx] - self.ttl_times[0]
            # licks = result[0]/(30*33.425)

            #
            self.ttl_times -= self.ttl_times[0]

            #
            self.E1 = data["rois_traces_smooth1"]
            self.E2 = data["rois_traces_smooth2"]


            #
            self.E1[:, :10] = 0
            self.E2[:, :10] = 0

            #self.E2[1, self.ttl_det]

            # old method:
            # self.E = data['ensemble_diff_array']
            self.E = self.E1.sum(0)-self.E2.sum(0)

            #
            self.rec_len_mins = self.E1.shape[1]/self.sample_rate/60.


            # Setdefault high trehsold and white noise in case they are not saved to xlsx document
            self.high_threshold = data['high_threshold']
            self.high_threshold = self.ttl_times*0 + self.high_threshold

            self.white_noise = self.high_threshold*0

        except:
            print ("Could not load file: ", fname)
            self.reward_times = [None]
            
        #
        #######################################################
        ################## LOAD DICTIONARY ####################
        #######################################################
        fname_dict = os.path.join(self.root_dir,
                                  self.animal_id,
                                  self.session_id,
                                  'data', 'results.xlsx')

        #
        if os.path.exists(fname_dict):
            # SEARCH FOR DICTIONARY:
            df = pd.read_excel(fname_dict)
            D = df.iloc[:, 1:].values

            #
            self.white_noise = D[:, 2]
            self.high_threshold = D[:, 1]
            self.ttl_det = np.int32(D[:,0])-1
            post_reward = D[:, 3]

            # this is for the rare case when the results.npz file is corrupt
            if self.reward_times[0] == None:
                # find column name in D that is "water_reward"
                idx = np.where(df.columns == "water_reward")[0][0]
                print ("water_reward loaded from xlsx column: ", idx)
                self.reward_times = np.int32(D[:, idx-1])

                # find where reward_times switches from 0 to 1
                idx = np.where(np.diff(self.reward_times) == 1)[0]
                self.reward_times = idx

                print ("reward times: ", self.reward_times)
                self.rewards = self.reward_times.copy()

        else:
            if self.verbose:
                print ("Missing dictionary (early sessions... skipping)")

        print ("    ... done loading data...")

        #
        if self.verbose:
            print("reward times: ", self.reward_times.shape)
            print ("Recording length (mins): ", self.rec_len_mins)
            print("abs times: ", self.abs_times.shape, self.abs_times)
            print("ttl times: ", self.ttl_times.shape, self.ttl_times[0], self.ttl_times[-1], " total rec time sec: ",
              self.ttl_times[-1] - self.ttl_times[0])
            print("ttl computed: ", self.ttl_comp.shape, self.ttl_comp)
            print("ttl detected: ", self.ttl_det.shape, self.ttl_det)
            print("lick detector: ", self.lick_detector.shape)
            print("lick times: ", self.lick_times)
            print("E1 , E2, ", self.E1.shape, self.E2.shape)


    def process_session_traces(self):

        fname_out = os.path.join(self.save_dir,'session.png')

        if os.path.exists(fname_out) and self.recompute_session==False:
            #print ("Session already processed, skipping")
            return
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
        #plt.plot(self.ttl_times, self.E1[0, self.ttl_comp], c='blue', label='roi1', alpha=alpha)
        #plt.plot(self.ttl_times, self.E1[1, self.ttl_comp] + scale, c='lightblue', label='roi2', alpha=alpha)
        #plt.plot(self.ttl_times, self.E2[0, self.ttl_comp] + scale * 2, c='red', label='roi3', alpha=alpha)
        #plt.plot(self.ttl_times, self.E2[1, self.ttl_comp] + scale * 3, c='pink', label='roi4', alpha=alpha)

        #
        plt.plot(self.ttl_times, self.E1[0, self.ttl_det], c='blue', label='roi1', alpha=alpha)
        plt.plot(self.ttl_times, self.E1[1, self.ttl_det] + scale, c='lightblue', label='roi2', alpha=alpha)
        plt.plot(self.ttl_times, self.E2[0, self.ttl_det] + scale * 2, c='red', label='roi3', alpha=alpha)
        plt.plot(self.ttl_times, self.E2[1, self.ttl_det] + scale * 3, c='pink', label='roi4', alpha=alpha)

        # total ensemble state
        ctr = 6
        plt.plot(self.ttl_times, self.E[self.ttl_det] + scale * ctr, c='black', label='Ensemble state', alpha=.3)

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
        if self.verbose:
            print ("White noise: ", idx.shape)
        ax.scatter(
            self.ttl_times[idx],
            self.ttl_times[idx] * 0 + scale * ctr,
            color='brown',
            alpha=1,
            label='white-nose',
        )

        # lick times
        ctr += .4
        licks = np.unique(np.round(self.lick_times,1))
        plt.scatter(licks, licks * 0 + scale * ctr, alpha=0.8, c='orange', label='lick detector')

        #
        plt.legend()
        plt.ylim(-0.5, 15)
        plt.xlabel("Time (sec)")
        plt.xlim(self.ttl_times[0], self.ttl_times[-1])
        plt.suptitle(self.animal_id + " " + self.session_id)

        plt.savefig(fname_out ,dpi=200)

        if self.show_plots:
            plt.show()
        else:
            plt.close()

	#
    def process_calibration(self):

        d = np.load(os.path.join(self.root_dir,
                                 self.animal_id,
                                 self.session_id,
                                 'rois_pixels_and_thresholds.npz'), allow_pickle=True)
        if False:
            roi_traces = np.array(d['all_roi_traces_submsampled'])
            print ("Calibration roi traces: ", roi_traces.shape)

            cell_ids = d['cell_ids']
            print ("cell ids: ", cell_ids)

            self.roi_traces_calibration = roi_traces[cell_ids]

            f0s = np.mean(self.roi_traces_calibration, axis=0)
            self.roi_traces_calibration = self.roi_traces_calibration/f0s

        else:
            e1_footprints = d['ensemble1_footprints']
            e2_footprints = d['ensemble2_footprints']
            #print ("e1 footprints: ", e1_footprints)

            #
            raw_ca = np.memmap(os.path.join(self.root_dir,
                                            self.animal_id,
                                            self.session_id,
                                            'calibration',
                                            "Image_001_001.raw"),
                                           dtype='uint16',
                                           mode='r')
            raw_ca = raw_ca.reshape(-1, 512, 512)
            print (raw_ca.shape)

            self.calibration_subsample = 1
            self.e1_1 = []
            self.e1_2 = []
            self.e2_1 = []
            self.e2_2 = []

            def get_roi(frame, roi):
                # new way use exact pixel location
                temp = frame[
                        roi[:, 0],  # broadcast/index into the frame as per ROI pixels
                        roi[:, 1]]

                # divide by the number of pixels in the ROI - NOT SURE IF THIS IS CORRECT?!
                # TODO: these algorithms must match the default water disposal algorithms
                # TODO: USE A FUNCTION OVER THIS AND FOLLOWING STEP THAT IS SHARED WITH CALIBRATION CODE
                roi_sum0 = temp / roi.shape[0]

                # sum
                # TODO: not sure this is the correct function; to check literature
                # TODO: also this part shoudl be refactored to a callabale function by both calibration and BMI classes
                roi_sum0 = np.nansum(roi_sum0)

                # Note: Do not remove baseline yet; this is done in the smoothing step;
                # TODO: make sure that this approach is correct
                return roi_sum0


            for k in trange(0,raw_ca.shape[0],self.calibration_subsample):
                self.e1_1.append(get_roi(raw_ca[k], e1_footprints[0]))
                self.e1_2.append(get_roi(raw_ca[k], e1_footprints[1]))
                self.e2_1.append(get_roi(raw_ca[k], e2_footprints[0]))
                self.e2_2.append(get_roi(raw_ca[k], e2_footprints[1]))

            def smooth_ca_time_series4(diff):
                #
                ''' This returns the last value. i.e. no filter

                '''

                temp = (diff[-1] * 0.4 +
                        diff[-2] * 0.25 +
                        diff[-3] * 0.15 +
                        diff[-4] * 0.10 +
                        diff[-5] * 0.10)

                return temp

            def get_dff(trace):

                trace = np.hstack(trace)
                trace_smooth = trace.copy()
                trace_smooth[:5]=0

                f0 = np.mean(trace)

                for p in trange(5, trace.shape[0],1):
                    #
                    roi_history = trace[p-5:p]

                    #
                    rois_dff = (roi_history - f0) / f0

                    #
                    trace_smooth[p] = smooth_ca_time_series4(rois_dff)

                return trace_smooth

            self.e1_1 = get_dff(self.e1_1)
            self.e1_2 = get_dff(self.e1_2)
            self.e2_1 = get_dff(self.e2_1)
            self.e2_2 = get_dff(self.e2_2)


    #
    def process_calibration_new(self):
            
        fname = os.path.join(self.root_dir,
                                 self.animal_id,
                                 self.session_id,
                                 'calibration',
                                 'results.npz')
        print ("fname: ", fname)
        d = np.load(fname, allow_pickle=True)

        #
        self.e1_1 = d['rois_traces_smooth1'][0]
        self.e1_2 = d['rois_traces_smooth1'][1]
        self.e2_1 = d['rois_traces_smooth2'][0]
        self.e2_2 = d['rois_traces_smooth2'][1]

    #
    def show_session_traces_and_calibration(self):

        #


        ########################################################
        ########################################################
        ########################################################
        #
        scale = 2
        alpha = .7

        plt.figure(figsize=(40,20))
        ax = plt.subplot(1, 1, 1)

        # plot calibration
        #t_calibration = np.arange(self.e1_1.shape[0])*self.calibration_subsample/30.
        t_calibration = np.arange(self.e1_1.shape[0])/30.
        t_calibration = t_calibration-t_calibration[-1]


        plt.plot(t_calibration, self.e1_1, c='blue', alpha=alpha)
        plt.plot(t_calibration, self.e1_2+scale , c='lightblue', alpha=alpha)
        plt.plot(t_calibration, self.e2_1+scale*2, c='red', alpha=alpha)
        plt.plot(t_calibration, self.e2_2+scale*3, c='pink', alpha=alpha)


        #
        plt.plot(self.ttl_times, self.E1[0, self.ttl_det], c='blue', label='roi1', alpha=alpha)
        plt.plot(self.ttl_times, self.E1[1, self.ttl_det] + scale, c='lightblue', label='roi2', alpha=alpha)
        plt.plot(self.ttl_times, self.E2[0, self.ttl_det] + scale * 2, c='red', label='roi3', alpha=alpha)
        plt.plot(self.ttl_times, self.E2[1, self.ttl_det] + scale * 3, c='pink', label='roi4', alpha=alpha)

        # total ensemble state
        ctr = 6
        plt.plot(self.ttl_times, self.E[self.ttl_det] + scale * ctr, c='black', label='Ensemble state', alpha=.3)

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
        if self.verbose:
            print ("White noise: ", idx.shape)
        ax.scatter(
            self.ttl_times[idx],
            self.ttl_times[idx] * 0 + scale * ctr,
            color='brown',
            alpha=1,
            label='white-nose',
        )

        # lick times
        ctr += .4
        licks = np.unique(np.round(self.lick_times,1))
        plt.scatter(licks, licks * 0 + scale * ctr, alpha=0.8, c='orange', label='lick detector')


        plt.plot([0,0],[0,scale*ctr],'--',c='grey')
        plt.ylim(-0.5, 15)

        #
        plt.legend()
        plt.xlabel("Time (sec)")
        plt.xlim(t_calibration[0], self.ttl_times[-1])
        plt.suptitle(self.animal_id + " " + self.session_id)

        plt.savefig(os.path.join(self.save_dir,'session.png'),dpi=200)

        if self.show_plots:
            plt.show()
        else:
            plt.close()


    #
    def compute_ensemble_correlations(self):

        fname_out = os.path.join(self.save_dir, 'pearson_corr_ensembles.npy')

        if os.path.exists(fname_out) and self.recompute_session==False:
            return

        #
        from scipy import stats
        self.E1_corr = stats.pearsonr(self.E1[0], self.E1[1])
        self.E2_corr = stats.pearsonr(self.E2[0], self.E2[1])

        if self.verbose:
            print("pearson correlation E1 cells; ", self.E1_corr[0])
            print("pearson correlation E2 cells; ", self.E2_corr[0])

        np.save(fname_out, [self.E1_corr, self.E2_corr])

    #
    def compute_correlograms_reward_vs_licking(self):

        fname_out = os.path.join(self.save_dir, 'correlograms_reward_vs_licking.npy')

        #
        if os.path.exists(fname_out) and self.recompute_session==False:
            return

        from correlograms_phy import correlograms

        # TODO: note all times are given in seconds

        # rewarded times
        spikes1 = self.ttl_times[self.reward_times]

        # lick times
        spikes2 = self.lick_times
        spikes2 = np.unique(np.round(self.lick_times,2))
        if self.verbose:
            print ("lick times: ", spikes2)

        spike_times = np.hstack((spikes1, spikes2))

        idx = np.argsort(spike_times)
        spike_times = spike_times[idx]
        if self.verbose:
            print ("# of spikes: ", spike_times.shape[0])

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
        plt.savefig(os.path.join(self.save_dir, 'correlograms_reward_vs_licking.png'), dpi=200)

        if self.show_plots:
            plt.show()
        else:
            plt.close()

        np.save(fname_out, corr)



    def compute_intra_session_inter_burst_interval(self):

        fname_out = os.path.join(self.save_dir, 'cell_isis.npy')

        if os.path.exists(fname_out) and self.recompute_session==False:
            return

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
        plt.savefig(os.path.join(self.save_dir,'cell_isis.png'),dpi=200)

        if self.show_plots:
            plt.show()
        else:
            plt.close()

        #
        np.save(fname_out, isi_array)




    def compute_intra_session_cell_burst_histogram_v2(self):

        fname_out = os.path.join(self.save_dir, 'cell_burst_histogram_v2.npy')

        if os.path.exists(fname_out) and self.recompute_session==False:
            return


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
        plt.savefig(os.path.join(self.save_dir,'cell_burst_histogram_v2.png'),dpi=200)

        if self.show_plots:
            plt.show()
        else:
            plt.close()

        np.save(fname_out, burst_array)

    #
    def compute_intra_session_cell_burst_histogram(self):

        fname_out = os.path.join(self.save_dir, 'cell_burst_histogram.npy')

        if os.path.exists(fname_out) and self.recompute_session==False:
            return

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
        plt.savefig(os.path.join(self.save_dir,'cell_burst_histogram.png'),dpi=200)

        if self.show_plots:
            plt.show()
        else:
            plt.close()

        np.save(fname_out, burst_array)


    #
    def compute_intra_session_reward_histogram(self):

        fname_out = os.path.join(self.save_dir,'intra_session_reward.npy')

        if os.path.exists(fname_out) and self.recompute_session==False:
            return

        rs = self.reward_times / self.sample_rate/60.

        y = np.histogram(rs,
                        bins = np.arange(0, self.rec_len_mins + self.bin_width, self.bin_width))

        #
        xx = y[1][:-1]+self.bin_width/2.
        yy = y[0]

        #
        from scipy import stats
        res = stats.pearsonr(xx,yy)
        if self.verbose:
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
        plt.savefig(os.path.join(self.save_dir,'intra_session_reward.png'),dpi=200)

        if self.show_plots:
            plt.show()
        else:
            plt.close()
        #
        np.save(fname_out,y   )


    #
    def compute_intra_session_reward_histogram_hist_per_minute(self):

        fname_out = os.path.join(self.save_dir,'intra_session_reward_hits_per_minute.npy')

        if os.path.exists(fname_out) and self.recompute_session==False:
            return

        #
        # make hits per min array
        xx = np.arange(0,self.rec_len_mins,self.bin_width_mins)
        start_ttl = 0                                 # this isn't quite correct; but it make a big difference unless we have weird offsets; 
                                                      #  for now we assume that ttl starts about same time as the recording
        hits_per_bin = np.zeros(xx.shape[0])
        misses_per_bin = np.zeros(xx.shape[0])
        n_trials_per_bin = np.zeros(xx.shape[0])
    
        for k in range(self.rewards.shape[0]):
            temp = self.rewards[k]
            abs_ttl_in_mins = (self.ends_ttl[k]-start_ttl)/self.sample_rate/60.      # here we compute the end of the trial in minutes

            # check if we have reached nans - which indicate we are done with the trials
            if np.isnan(abs_ttl_in_mins):
                break

            # convert the time to the bin index; and add it to the hits or misses array
            idx = int(abs_ttl_in_mins/self.bin_width_mins)
            if temp:
                hits_per_bin[idx]+= 1
            else:
                misses_per_bin[idx]+= 1

            n_trials_per_bin[idx]+=1

        #
        yy = hits_per_bin/n_trials_per_bin

        from scipy import stats
        try:
            res = stats.pearsonr(xx,yy)
        except:
            # 
            print ("THIS session has errors in the reward times", self.save_dir)
            print ("... saving nans array")
  
            np.save(fname_out,
                    yy*0 + np.nan)
            return

        if self.verbose:
            print ("Perason corr: ", res)

        #
        plt.figure()
        plt.plot(np.unique(xx), np.poly1d(np.polyfit(xx, yy, 1))(np.unique(xx)),
                 '--')
        plt.scatter(xx,yy, label = "pcorr: "+str(round(res[0],2))+ ", pval: "+str(round(res[1],5)))
        plt.bar(xx,yy,self.bin_width_mins*0.9, alpha=.5)
        plt.ylim(bottom=0)
        plt.xlim(xx[0]-self.bin_width_mins/2.,xx[-1]+self.bin_width_mins/2.)
        plt.legend()

        plt.xlabel("Time (mins)")
        plt.ylabel("% hit rate")
        plt.title(self.animal_id +  " -- " + self.session_id)
        plt.savefig(os.path.join(self.save_dir,'intra_session_reward_hist_per_minute.png'),dpi=200)

        if self.show_plots:
            plt.show()
        else:
            plt.close()
        #
        np.save(fname_out,yy)

    #
    def compute_animal_hit_rate(self):

        #
        hit_rates = []
        for session_id in self.session_ids:
            hit_rate = np.load(os.path.join(self.root_dir,
                                     self.animal_id,
                                     session_id,
                                     'results',
                                     'intra_session_reward_hits_per_minute.npy'))
            hit_rates.append(hit_rate)

        hit_rates = np.vstack(hit_rates)

        #
        # make hits per min array
        xx = np.arange(0,self.rec_len_mins,self.bin_width_mins)

        #
        yy = np.mean(hit_rates,0)


        #
        from scipy import stats
        res = stats.pearsonr(xx,yy)
        if self.verbose:
            print ("Perason corr: ", res)

        #
        plt.figure()
        plt.plot(np.unique(xx), np.poly1d(np.polyfit(xx, yy, 1))(np.unique(xx)),
                 '--')
        plt.scatter(xx,yy, label = "pcorr: "+str(round(res[0],2))+ ", pval: "+str(round(res[1],5)))
        plt.bar(xx,yy,self.bin_width_mins*0.9, alpha=.5)
        plt.ylim(bottom=0)
        plt.xlim(xx[0]-self.bin_width_mins/2.,xx[-1]+self.bin_width_mins/2.)
        plt.legend()

        plt.xlabel("Time (mins)")
        plt.ylabel("% hit rates- all essions")
        plt.title(self.animal_id +  " -- " + self.session_id + ", average hit rate all sessions")

        self.save_dir_root = os.path.join(self.root_dir,
                                          self.animal_id)
        plt.savefig(os.path.join(self.save_dir_root,'average_hit_rate.png'),dpi=200)

        if self.show_plots:
            plt.show()
        else:
            plt.close()
        #
        np.save(os.path.join(self.save_dir,'average_hit_rate.npy'),yy)


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
        #print ("spike times: ", spike_times)

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

        plt.savefig(os.path.join(self.save_dir,'correlograms_upphase.png'),dpi=200)
        if self.show_plots:
            plt.show()
        else:
            plt.close()

        #
        np.save(os.path.join(self.save_dir,'correlograms_upphase.npy'),corr)

    #
    def compute_correlograms_ensembles_fluorescence(self):

        from scipy import stats

        self.sample_rate = 30  # in Hz
        self.bin_size = 1  # in seconds

        # rewarded times
        std_threshold = 5
        names = ["roi1", "roi2","roi3","roi4",]

        #
        self.window = self.window_size*self.sample_rate

        #
        plt.figure()
        t=np.arange(-self.window, self.window, 1)/self.sample_rate
        cc_array = []
        for k in range(4):
            cc_array.append([])
            for p in range(4):
                cc_array[k].append([])

        #
        for k in range(4):
            for p in range(k,4,1):
                t1 = self.F_filtered[k]
                t2 = self.F_filtered[p]

                #
                cc = []
                for z in range(-self.window, self.window,1):
                    cc.append(stats.pearsonr(np.roll(t1,z), t2)[0])

                #print (k,p, 'cc: ', len(cc), "t: ", len(t))
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

                #
                plt.ylim(bottom=0)
                #plt.plot([0,0],
                #         [0,np.max(cc) ],
                #         '--',c='grey')

                # plot a horizontal line at zero
                plt.plot([t[0],t[-1]],
                            [0,0],
                            '--',c='grey')


        #
        plt.suptitle(self.animal_id + " " + self.session_id + " Raw fluorescence based xcorrelation")
        plt.savefig(os.path.join(self.save_dir, 'correlograms_fluorescence.png'), dpi=200)

        #
        plt.show()

        if self.show_plots==False:
            plt.close()
        #
        np.save(os.path.join(self.save_dir,'correlograms_fluorescence.npy'),cc_array)


    def correlograms_inter_session_early_vs_late(self):

        from scipy import stats

        self.sample_rate = 30  # in Hz
        self.bin_size = 1  # in seconds

        # rewarded times
        std_threshold = 5
        names = ["roi1", "roi2","roi3","roi4",]

        #
        plt.figure()
        t=np.arange(-self.window*self.sample_rate, self.window*self.sample_rate, 1)/30.
        cc_array = []

        #for k in range()
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.session_ids)))
        ctr_sess =0
        
        # print ("cc : ", cc)
        ctr=0
        ymaxs = np.zeros((4,4))
        ymins = np.zeros((4,4))


        for k in range(4):
            max_y = 0
            for p in range(k,4,1):

                early = []
                late = []
                for session_id in tqdm(self.session_ids[1:], desc='loading data for hit rate per session'):

                    #
                    cc = np.load(os.path.join(self.root_dir,
                                      self.animal_id,
                                      session_id,
                                      'results',
                                      'correlograms_fluorescence.npy'), allow_pickle=True)
            
                    y = cc[k][p]

                    if session_id in self.early:
                        early.append(y)
                    else:
                        late.append(y)
                    
                #
                plt.subplot(4,4,k*4+p+1)

                #
                early = np.mean(np.vstack(early),0)
                late = np.mean(np.vstack(late),0)

                #
                if k==0 and p==0:
                    plt.plot(t, early, 
                             color='blue', 
                             label='early')
                    plt.plot(t, late,
                                color='red',
                                label='late')
                    plt.legend(fontsize=8)
                else:
                    plt.plot(t,early, color='blue')
                    plt.plot(t,late, color='red')

                #
                if np.max(early)>ymaxs[k,p]:
                    ymaxs[k,p] = np.max(early)

                if np.max(late)>ymaxs[k,p]:
                    ymaxs[k,p] = np.max(late)

                if np.min(early)<ymins[k,p]:
                    ymins[k,p] = np.min(early)

                if np.min(late)<ymins[k,p]:
                    ymins[k,p] = np.min(late)


                # plot vertical line at zero
                plt.plot([0,0],
                            [ymins[k,p],ymaxs[k,p]],
                            '--',c='grey')
                
                ctr+=1

                plt.title(names[k] + " vs " +names[p])
                plt.xlim(t[0],t[-1])
                if p!=k:
                    plt.xticks([])
                else:
                    plt.xlabel("Time (sec)")

                    # plot a horizontal line at zero
                plt.plot([t[0],t[-1]],
                        [0,0],
                        '--',c='grey')

                        
                # plt.plot([0,0],
                #          [0,max_y],
                #          '--',c='grey')

                #plt.ylim(0, max_y)
                    
                    
            ctr_sess+=1
        plt.suptitle(self.animal_id  + " Raw fluorescence based xcorrelation")
        #plt.savefig(os.path.join(self.save_dir, 'correlograms_fluorescence.png'), dpi=200)
        #import time
        #time.sleep(1)
        plt.show()

        #if self.show_plots==False:
        #    plt.close()
        #
        #np.save(os.path.join(self.save_dir,'correlograms_fluorescence.npy'),cc_array, allow_pickle=True)

    #

    #
    def correlograms_inter_session(self):

        from scipy import stats

        self.sample_rate = 30  # in Hz
        self.bin_size = 1  # in seconds

        # rewarded times
        std_threshold = 5
        names = ["roi1", "roi2","roi3","roi4",]

        #
        plt.figure()
        t=np.arange(-self.window*self.sample_rate, self.window*self.sample_rate, 1)/30.
        cc_array = []

        #for k in range()
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.session_ids)))
        ctr_sess =0
        for session_id in tqdm(self.session_ids[1:], desc='loading data for hit rate per session'):

            #
            cc = np.load(os.path.join(self.root_dir,
                                      self.animal_id,
                                      session_id,
                                      'results',
                                      'correlograms_fluorescence.npy'), allow_pickle=True)
           # print ("cc : ", cc)
            print ("Length of cc: ", len(cc), len(cc[0]))
            ctr=0
            for k in range(4):
                max_y = 0
                for p in range(k,4,1):
                    
                    #
                    plt.subplot(4,4,k*4+p+1)

                    #
                    y = cc[k][p]
                    #print ("y,: ", y)
                    #y = y/np.max(y)
                    
                    #print ("y: ", y.shape, ", t: ", t.shape)
                    if k==0 and p==0:
                        plt.plot(t, y, color=colors[ctr_sess], label=session_id)
                        plt.legend(fontsize=8)
                    else:
                        plt.plot(t,y, color=colors[ctr_sess])


                    ctr+=1

                    plt.title(names[k] + " vs " +names[p])
                    plt.xlim(t[0],t[-1])
                    if p!=k:
                        plt.xticks([])
                    else:
                        plt.xlabel("Time (sec)")
                    #plt.ylim(bottom=0)

                     # plot a horizontal line at zero
                    plt.plot([t[0],t[-1]],
                            [0,0],
                            '--',c='grey')

                        
                # plt.plot([0,0],
                #          [0,max_y],
                #          '--',c='grey')

                #plt.ylim(0, max_y)
                    
                    
            ctr_sess+=1
        plt.suptitle(self.animal_id  + " Raw fluorescence based xcorrelation")
        #plt.savefig(os.path.join(self.save_dir, 'correlograms_fluorescence.png'), dpi=200)
        #import time
        #time.sleep(1)
        plt.show()

        #if self.show_plots==False:
        #    plt.close()
        #
        #np.save(os.path.join(self.save_dir,'correlograms_fluorescence.npy'),cc_array, allow_pickle=True)

    #
    def binarize_ensembles(self):

        fname_out = os.path.join(self.save_dir,'binarized_traces.npy')

        #if os.path.exists(fname_out) and self.recompute_session==False:
        #    return

        from calcium import Calcium

        c = Calcium()
        c.data_dir = os.path.join(self.root_dir,
                                  self.animal_id,
                                  self.session_id)

        # c.detrend_model_order = 1
        c.save_python = True
        c.save_matlab = False
        c.sample_rate = 30

        #
        c.min_width_event_onphase = self.min_width_event_onphase
        c.min_width_event_upphase = self.min_width_event_upphase

        ############# PARAMTERS TO TWEAK ##############
        #     1. Cutoff for calling somthing a spike:
        #        This is stored in: std_Fluorescence_onphase/uppohase: defaults: 1.5
        #                                        higher -> less events; lower -> more events
        #                                        start at default and increase if data is very noisy and getting too many noise-events
        c.min_thresh_std_onphase = self.std_upphase               # set the minimum thrshold for onphase detection; defatul 2.5
        c.min_thresh_std_upphase = self.std_upphase  # set the minimum thershold for uppohase detection; default: 2.5

        #     2. Filter of [Ca] data which smooths the data significantly more and decreases number of binarzied events within a multi-second [Ca] event
        #        This is stored in high_cutoff: default 0.5 to 1.0
        #        The lower we set it the smoother our [Ca] traces and less "choppy" the binarized traces (but we loose some temporal precision)
        c.high_cutoff = 0.5

        #     3. Removing bleaching and drift artifacts using polynomial fits
        #        This is stored in detrend_model_order
        c.detrend_model_order = 1  # 1-5 polynomial fit
        c.detrend_model_type = 'polynomial'  # 'polynomial' or 'exponential'

        # remove first five seconds of data ni traces
        self.E1[0][:5 * self.sample_rate] = 0
        self.E1[1][:5 * self.sample_rate] = 0
        self.E2[0][:5 * self.sample_rate] = 0
        self.E2[1][:5 * self.sample_rate] = 0
        
              #
        traces = np.vstack((self.E1[0],
                            self.E1[1],
                            self.E2[0],
                            self.E2[1])).astype('float32')


        #
        c.F = traces
        c.verbose=False
        c.percentile_threshold = self.percentile_threshold
        c.recompute_binarization = self.recompute_binarization


        #c.binarize_fluorescence2()
        c.load_binarization()


        self.F_upphase_bin = c.F_upphase_bin
        self.F_onphase_bin = c.F_onphase_bin
        self.F_filtered = c.F_filtered

        np.save(fname_out,self.F_upphase_bin)

        ################################################
        ############### SIMPLE VIS TEST ################
        ################################################
        from scipy.optimize import curve_fit
        from scipy import asarray as ar, exp
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

            if self.use_upphase:
                yy = self.F_upphase_bin[k]
            else:
                yy = self.F_onphase_bin[k]
            #
            t=np.arange(self.F_filtered.shape[1])/30.

            #
            y = self.F_filtered[k]

            #
            plt.plot(t,y,c='black',alpha=.5,label=names[k])

            #
            plt.plot(t,yy, c=clrs[k//2],alpha=.5)

          # plot histogram side of panel
            plt.legend()
            plt.xlim(t[0],t[-1])

            plt.suptitle("Using STD for upphase detection: "+str(self.std_upphase))

        plt.savefig(os.path.join(self.save_dir, 'binarized_traces.png'), dpi=200)

        #
        if self.show_plots:
            plt.show()
        else:
            plt.close()



    def n_rewards_intra_session(self):
        #
        labels = []
        n_rewards = []
        rewards_array = []
        ctr=0
        for session_id in self.session_ids[1:]:
            S = ProcessSession(self.root_dir,
                               self.animal_id)
            
            S.session_id = session_id
            #
            S.verbose=self.verbose
            S.load_data()

            #
            labels.append(session_id)

            #
            n_rewards.append(len(S.reward_times))

            temp = np.float32(S.reward_times)/30./60.
            y = np.histogram(temp, bins=np.arange(0,55,5))
            rewards_array.append(y[0])

            ctr+=1
            #if ctr>3:
            #    break

        #
        from scipy import stats

        #
        yy = np.vstack(rewards_array)
        xx = np.arange(yy.shape[1])*5+2.5

        mean = np.mean(yy,axis=0)
        std = np.std(yy, axis=0)

        #
        res = stats.pearsonr(xx,mean)
        if self.verbose:
            print ("Perason corr: ", res)

        plt.figure()
        ax1=plt.subplot(111)
        plt.plot(np.unique(xx), np.poly1d(np.polyfit(xx, mean, 1))(np.unique(xx)),
                 '--')

        plt.plot(xx,mean, c='blue', label = "pcorr: "+str(round(res[0],2))+ ", pval: "+str(round(res[1],5)),
                    linewidth=5)

        ax1.fill_between(xx, mean + std, mean - std, color='blue', alpha=0.1)

        plt.xlim(xx[0]-2.5,xx[-1]+2.5)

        plt.ylabel("# rewards")
        plt.xlabel("Time (mins)")
        plt.ylim(bottom=0)
        plt.legend()
        plt.suptitle(self.animal_id)

        plt.savefig(os.path.join(self.save_dir_root,
                                'n_rewards_intra_session.png'), dpi=200)
        if self.show_plots:
            plt.show()
        else:
            plt.close()

        np.save(os.path.join(self.save_dir_root,'n_rewards_intra_session.npy'),mean)


    def n_rewards_intra_session_normalized(self):

        #
        labels = []
        n_rewards = []
        rewards_array = []
        ctr=0
        for session_id in tqdm(self.session_ids[1:]):
            S = ProcessSession(self.root_dir,
                               self.animal_id,
                               #session_id
                               )
            #
            S.verbose = self.verbose

            #
            S.session_id = session_id

            S.load_data()

            #
            labels.append(session_id)

            #
            n_rewards.append(len(S.reward_times))

            temp = np.float32(S.reward_times)/30./60.
            y = np.histogram(temp, bins=np.arange(0,55,5))

            y_out = (y[0]+1)/(y[0][0]+1)
            rewards_array.append(y_out)

            ctr+=1

            #
            #if ctr>3:
            #    break

        #
        from scipy import stats

        #
        #print (rewards_array)
        yy = np.vstack(rewards_array)
        #print (yy.shape)
        xx = np.arange(yy.shape[1])*5+2.5

        mean = np.mean(yy,axis=0)
        std = np.std(yy, axis=0)

        #
        res = stats.pearsonr(xx,mean)
        if self.verbose:
            print ("Perason corr: ", res)

        plt.figure()
        ax1=plt.subplot(111)
        plt.plot(np.unique(xx), np.poly1d(np.polyfit(xx, mean, 1))(np.unique(xx)),
                 '--')

        plt.plot(xx,mean, c='blue', label = "pcorr: "+str(round(res[0],2))+ ", pval: "+str(round(res[1],5)),
                    linewidth=5)

        ax1.fill_between(xx, mean + std, mean - std, color='blue', alpha=0.1)
        plt.xlim(xx[0]-2.5,xx[-1]+2.5)
        plt.ylabel("# rewards (normalized to first 5mins)")
        plt.xlabel("Time (mins)")
        plt.ylim(bottom=0)
        plt.legend()
        plt.suptitle(self.animal_id)

        plt.savefig(os.path.join(self.save_dir_root,
                                'n_rewards_intra_session_normalized.png'), dpi=200)
        if self.show_plots:
            plt.show()
        else:
            plt.close()

        np.save(os.path.join(self.save_dir_root,'n_rewards_intra_session_normalized.npy'), mean)

    #
    def hit_rate_per_session(self):
        #
        labels = []
        n_rewards = []
        hit_rates = []
        for session_id in tqdm(self.session_ids[1:], desc='loading data for hit rate per session'):

            #
            hr = np.load(os.path.join(self.root_dir,
                                      self.animal_id,
                                     session_id,
                                     'results',
                                     'intra_session_reward_hits_per_minute.npy'))

            #
            hit_rates.append(hr)
            labels.append(session_id)

        # from scipy import stats
        # xx = np.arange(len(n_rewards))
        # yy = np.array(n_rewards)
        # res = stats.pearsonr(xx, yy)
        # if self.verbose:
        #     print("Perason corr: ", res)

        plt.figure()
        #plt.plot(np.unique(xx), np.poly1d(np.polyfit(xx, yy, 1))(np.unique(xx)),
        #         '--')

        for k in range(len(hit_rates)):
            y = hit_rates[k]
            x = y*0 + k
            plt.scatter(x, y, c='black')
            plt.bar(x[0], np.mean(y), 0.9, color='blue', alpha=.5)

        plt.xticks(np.arange(len(hit_rates)), labels, rotation=30)
        #plt.xlim(xx[0] - 0.5, xx[-1] + 0.5)
        plt.ylabel("hit rates")
        plt.ylim(bottom=0)
        plt.legend()
        plt.suptitle(self.animal_id)

        self.save_dir_root = os.path.join(self.root_dir,
                                          self.animal_id)
        plt.savefig(os.path.join(self.save_dir_root,
                                 'hit_rates.png'), dpi=200)
        if self.show_plots:
            plt.show()
        else:
            plt.close()

        np.save(os.path.join(self.save_dir_root, 'hit_Rates.npy'), hit_rates)


    def n_rewards_per_session(self):
        
        self.save_dir_root = os.path.join(self.root_dir,
                                          self.animal_id)
                
        fname_out = os.path.join(self.save_dir_root,'n_rewards_per_session.npy')

        if os.path.exists(fname_out):
            print ("Loading n_rewards_per_session.npy")
            yy = np.load(fname_out)
            labels = []

            for session_id in tqdm(self.session_ids[1:],desc='loading data for n_rewards_per_session'):
                #
                labels.append(session_id)
        else:
            #
            labels = []
            n_rewards = []
            ctr=0
            for session_id in tqdm(self.session_ids[1:],desc='loading data for n_rewards_per_session'):
                S = ProcessSession(self.root_dir,
                                self.animal_id)
                #
                S.verbose = self.verbose

                #
                S.session_id = session_id

                S.load_data()

                #
                labels.append(session_id)

                #
                n_rewards.append(S.reward_times.shape[0])

                #print ("S.reward_times: ", S.reward_times)
                #print ("session_id: ", session_id)

            yy = np.array(n_rewards)


        from scipy import stats
        xx = np.arange(len(yy))
        #yy = np.array(n_rewards)
        print ("xx: ", xx)
        print ("yy: ",yy)
        res = stats.pearsonr(xx,yy)
        if self.verbose:
            print ("Perason corr: ", res)

        ###################################
        ###################################
        ###################################

        plt.figure()
        plt.plot(np.unique(xx), np.poly1d(np.polyfit(xx, yy, 1))(np.unique(xx)),
                 '--')

        plt.scatter(xx,yy, c='blue', label = "pcorr: "+str(round(res[0],2))+ ", pval: "+str(round(res[1],5)),
                    linewidth=5)

        plt.xticks(np.arange(len(yy)), labels, rotation=30)
        plt.xlim(xx[0]-0.5,xx[-1]+0.5)
        plt.bar(xx,yy,0.9, alpha=.5)
        plt.ylabel("# rewards")
        plt.ylim(bottom=0)
        plt.legend()
        plt.suptitle(self.animal_id)


        plt.savefig(os.path.join(self.save_dir_root,
                                'n_rewards_per_session.png'), dpi=200)
        if self.show_plots:
            plt.show()
        else:
            plt.close()

        np.save(fname_out,yy)

    #
    def n_bursts_per_session(self):

        #
        labels = []
        bursts_all = []

        for session_id in tqdm(self.session_ids[1:],desc='loading data for n_bursts_per_session'):
            S = ProcessSession(self.root_dir,
                               self.animal_id)
            #
            #S.load_data()
            self.save_dir = os.path.join(self.root_dir,
                                         self.animal_id,
                                         session_id,
                                         'results')

            burst_array = np.load(os.path.join(self.save_dir, 'cell_burst_histogram_v2.npy'))

            bursts_all.append(burst_array)
            #
            labels.append(session_id)

        #
        from scipy import stats

        ########################################################
        import matplotlib.pyplot as plt
        plt.figure()
        names = ['roi1','roi2','roi3','roi4']
        clrs = ['blue','red']
        for k in range(4):
            plt.subplot(2,2,k+1)

            xx = np.arange(len(labels))

            yys = []
            yys_sums = []
            for s in range(len(bursts_all)):
                yys.append(bursts_all[s][k])
                yys_sums.append(bursts_all[s][k].sum())

            #
            yys_sums = np.array(yys_sums)
            res = stats.pearsonr(xx,yys_sums)
            #print ("Perason corr: ", res)

            plt.plot(np.unique(xx),
                     np.poly1d(np.polyfit(xx, yys_sums, 1))(np.unique(xx)),
                     '--')

            plt.scatter(xx,yys_sums, c=clrs[k//2], label = names[k]+ ", pcorr: "+str(round(res[0],2))+ ", pval: "+str(round(res[1],5)),
                        linewidth=5)

            plt.xticks(np.arange(len(yys_sums)), labels, rotation=30)
            plt.xlim(xx[0]-0.5,xx[-1]+0.5)
            plt.bar(xx,yys_sums,0.9, color=clrs[k//2],alpha=.5)
            plt.ylabel("# bursts")
            plt.ylim(bottom=0)
            plt.legend()
            if k < 2:
                plt.xticks([])

        plt.suptitle(self.animal_id)

        plt.savefig(os.path.join(self.save_dir_root,
                                 'n_bursts_per_session.png'), dpi=200)
        if self.show_plots:
            plt.show()
        else:
            plt.close()



        #########################################################
        #########################################################
        #########################################################
        plt.figure()
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt


        names = ['roi1','roi2','roi3','roi4']
        clrs = ['blue','red']
        for k in range(4):
            plt.subplot(2,2,k+1)

            xx = np.arange(len(labels))

            yys = []
            yys_sums = []
            for s in range(len(bursts_all)):
                yys.append(bursts_all[s][k])
                yys_sums.append(bursts_all[s][k].sum())

            colors = plt.cm.viridis(np.linspace(0, 1, len(yys)))
            #print (cmap)

            #
            for s in range(len(yys)):
                xx = np.arange(len(yys[s])) * 5+2.5

                plt.plot(xx, yys[s], color=colors[s],
                         label=labels[s]
                         )

            #red_patch = mpatches.Patch(color='none', label=names[k])
            #plt.legend(handles=[red_patch])
            if k<2:
                plt.xticks([])
            plt.title(names[k])
            plt.ylabel("# bursts")
            plt.xlabel("Time (mins)")
            plt.ylim(bottom=0)
            plt.xlim(xx[0]-2.5, xx[-1]+2.5)
            #plt.legend()
        plt.suptitle(self.animal_id)


        #
        plt.savefig(os.path.join(self.save_dir_root,
                                 'n_bursts_per_session2.png'), dpi=200)
        if self.show_plots:
            plt.show()
        else:
            plt.close()

#