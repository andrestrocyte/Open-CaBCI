import numpy as np
# Visualisation
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from tqdm import tqdm, trange
import parmap
import scipy

#
from calcium import Calcium

#
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
class ProcessCalcium():

    def __init__(self, root_dir, 
                 animal_id 
                 ):

        #
        self.root_dir = root_dir
        self.animal_id = animal_id
        
        # load yaml file
        fname = os.path.join(self.root_dir,
                            self.animal_id,
                            animal_id+'.yaml')
        
        # load yaml file
        import yaml
        with open(fname) as file:
            doc = yaml.load(file, Loader=yaml.FullLoader)

        self.session_ids = np.array(doc['session_names'],dtype='str')

        #
        self.shifts = doc['shifts']

        #
        self.verbose = True

    # #
    def load_day0_mask(self):

    
        fname = os.path.join(
                        self.root_dir,
                        self.animal_id,
                        'day0',
                        'rois_pixels_and_thresholds_day0.npz')

        #
        try:
            data = np.load(fname, allow_pickle=True)
        except:
            print ("Could not Day0 masks ... trying day1 ")
            fname = os.path.join(
                                self.root_dir,
                                self.animal_id,
                                self.session_ids[1],
                                'rois_pixels_and_thresholds.npz')

            data = np.load(fname, allow_pickle=True)
        #
        contours_all_cells = data['contours_all_cells']
        self.cell_ids = data['cell_ids']

        self.contours_ROIs = contours_all_cells[self.cell_ids]

        
    #
    def compute_reward_centered_traces(self):

        
        if True:
            parmap.map(get_reward_centered_traces,
                        self.session_ids,
                        self.root_dir,
                        self.animal_id,
                        self.reward_window,
                        self.filter,
                        pm_processes=10,
                        pm_pbar=True)

    #
    def plot_network_graph(self):

        print (self.reward_centered_traces.shape)

        #
        temp = self.reward_centered_traces.copy().transpose(1,0,2)
        print (temp.shape)

        temp = temp.reshape(temp.shape[0],-1)
        print (temp.shape)

        # remove means from each cell
        temp = temp - np.median(temp,1)[:,np.newaxis]

        # compute the correlation matrix
        corr_matrix = np.corrcoef(temp)
        print (corr_matrix.shape)

        # set threshold for correlation matrix at 0.3
        corr_matrix[corr_matrix<0.3]=0

        ###################################################
        # use the corr_matrix as an adjacency matrix and compute the graph
        import networkx as nx
        G = nx.from_numpy_array(corr_matrix)

        # remove self-loops
        G.remove_edges_from(nx.selfloop_edges(G))

        # from pylab import rcParams
        # rcParams['figure.figsize'] = 14, 10
        pos = nx.spring_layout(G, scale=20, k=3/np.sqrt(G.order()))
        d = dict(G.degree)
        nx.draw(G, pos, 
                node_color='lightblue', 
                with_labels=False, 
                node_size = 2,
                width = 0.15,
                nodelist=d, 
                alpha=1,
                #node_size=[d[k]*300 for k in d]
                ax=self.ax
                )


        # title should show graph density and also the number of nodes
        density = 2*G.number_of_edges() / (G.number_of_nodes() *(G.number_of_nodes() - 1))

        self.ax.set_title("\n\n\n"+self.animal_id+",  "+self.session_ids[self.session_id]+'\n' 
                    +"# of cells: "+ str(corr_matrix.shape[0])+'\n'
                        +'Graph density: '+str(np.round(density,3))+'\n'
                        +'Number of nodes: '+str(G.number_of_nodes())+'\n'
                        +'Number of edges: '+str(G.number_of_edges())+'\n'
                        +'Number of isolates: '+str(nx.number_of_isolates(G)),
                        fontsize=14)


    def plot_reward_centered_traces_ROIs_all_days(self):


        #
        clrs = ['blue','lightblue','red','pink']
        clrs2 = ['black','green','magenta','orange']

        # make a viridis based discrete colormap
        import matplotlib.colors as colors
        cmap = plt.cm.viridis(np.linspace(0,1,len(self.session_ids)-1))
        
        
        #
        plt.figure()
        for cell_id in range(4):
            #
            ax=plt.subplot(2,2,cell_id+1)

            # plot title about cell_id
            ax.set_title('Cells matching ROI: '+str(cell_id), fontsize=14)

            
            # loop over all the session_ids
            ctr_session = 0
            for session_id in range(1,len(self.session_ids),1):

                # get the roi for the session      
                closest_ROI_cells = self.session_contours[session_id]
                #print ("closest_ROI_cells: ", closest_ROI_cells)

                # load reward centered traces for the session
                fname = os.path.join(
                                    self.root_dir,
                                    self.animal_id,
                                    self.session_ids[session_id],
                                    'reward_centered_traces_rewardwindow_'+str(self.reward_window)+'.npy')

                reward_centered_traces = np.load(fname, allow_pickle=True)
        
                # plot the mean and std for all the cells spacing them out by 3
                t = np.arange(reward_centered_traces.shape[2])/30-(self.reward_window/30)

                #
                cell_traces = []
                for k in range(reward_centered_traces.shape[1]):


                    # loop over all the closest_ROI_cells and check if k is matching
                    for i in range(len(closest_ROI_cells)):
                        matching_mask = closest_ROI_cells[i][0]
                        matching_cell = closest_ROI_cells[i][1]
                        
                        #print ("cell_id: ", cell_id, 'k: ', k, ' matching_cell: ', matching_cell, ' matching_mask: ', matching_mask)
                        if k == matching_cell and matching_mask == cell_id:

                            # grab the data                            #
                            temp = reward_centered_traces[:,k]

                            # subtract the median from each trace
                            # compute f0 as the median of the stacked trace
                            #f0 = np.median(temp.flatten())
                            #print ("f0: ", f0)
                            #temp = (temp-f0)/f0*100

                            #
                            #import scipy
                            #temp_mean = scipy.stats.mode(temp,0)
                            
                            #print ("temp_mean: ", temp_mean.shape)

                            # compute the std of temp in the first axis
                            temp_mean = np.mean(temp,0)
                            
                            cell_traces.append(temp_mean)
                            
                            


                # 
                cell_traces = np.array(cell_traces)

                # pick the cell trace with the highest ptp peak to peak
                #print ("cell_traces: ", cell_traces.shape)

                if cell_traces.shape[0]==0:
                    continue
               
                elif cell_traces.shape[0]==1:
                    cell_trace = cell_traces[0]

                elif cell_traces.shape[0]>1:
                    ptps = np.ptp(cell_traces,1)

                    idx = np.argmax(ptps)
                    cell_trace = cell_traces[idx]


                # plot legend only if cell_id ==0:
                if cell_id==0:
                    plt.plot(t,cell_trace,
                            linewidth=3,
                            c=cmap[ctr_session],
                            label=self.session_ids[session_id]+ ", # rew: "+str(reward_centered_traces[:,k].shape[0]))
                else:
                    plt.plot(t,cell_trace,
                            linewidth=3,
                            c=cmap[ctr_session],
                            )


                ctr_session+=1


            # plot horizontal line at 0
            plt.plot(t, temp_mean*0,'--',
                    c='grey')    
            
            # plot vertical line at 0
            #plt.plot([0,0],[-50,100],'--',
            #        c='grey',
            #        linewidth=3)

            # 
            #plt.plot([0,0],[-.50,100],'--',
            #         c='grey',
            #         linewidth=3)
            #
            plt.ylabel("DFF")
            #
            plt.legend()

        # label y axis using ylabels 
        #plt.yticks(ylabels, np.arange(temp.shape[0]))
        
        #
        plt.suptitle(self.animal_id)
        plt.show()




    #
    def plot_reward_centered_traces_ROIs_only(self):

        #
        clrs = ['blue','lightblue','red','pink']
        clrs2 = ['black','green','magenta','orange','yellow','cyan','purple','brown']


        # plot the mean and std for all the cells spacing them out by 3
        t = np.arange(self.reward_centered_traces.shape[2])/30-(self.reward_window/30)

        #
        closest_ROI_cells = self.session_contours[self.session_id]
        print ("closest_ROI_cells: ", closest_ROI_cells)
    
        #
        linetypes = ['solid',':','--','dashdot']

        plt.figure()
        scale = 100
        for cell_id in range(4):
            #
            ax=plt.subplot(2,2,cell_id+1)
            
            # plot title about cell_id
            ax.set_title('Cells matching ROI: '+str(cell_id), fontsize=14)


            #
            ctr = 0
            for k in range(self.reward_centered_traces.shape[1]):

                #
                temp = self.reward_centered_traces[:,k]

                # subtract the median from each trace
                #temp = temp - np.median(temp,1)[:,np.newaxis]
                #print ("temp: ", temp.shape)
                temp_mean = np.mean(temp,0)

                # compute the std of temp in the first axis
                #temp_mean = np.mean(temp,0)

                # loop over all the closest_ROI_cells and check if k is matching
                for i in range(len(closest_ROI_cells)):
                    matching_mask = closest_ROI_cells[i][0]
                    matching_cell = closest_ROI_cells[i][1]
                    
                    #print ("cell_id: ", cell_id, 'k: ', k, ' matching_cell: ', matching_cell, ' matching_mask: ', matching_mask)
                    if k == matching_cell and matching_mask == cell_id:
                    
                        # if k is the first two values in cell_ids plot using blue
                        plt.plot(t,temp_mean,
                                linewidth=3,
                                #linestyle=linetypes[ctr],
                               # alpha=0.5,
                                c=clrs2[ctr],
                                label=str(matching_cell))

                        ctr+=1
                
            # plot horizontal line at 0
            plt.plot(t, temp_mean*0,'--',
                    c='grey')    
            
            # plot vertical line at 0
            #plt.plot([0,0],[-100,500],'--',
            #        c='grey',
            #        linewidth=3)

            # 
            #plt.plot([0,0],[-100,500],'--',
            #         c='grey',
            #         linewidth=3)

            #
            plt.legend()

        # label y axis using ylabels 
        #plt.yticks(ylabels, np.arange(temp.shape[0]))
        plt.xlabel('Time (s)',fontsize=20)

        #
        plt.suptitle(self.session_ids[self.session_id])
        plt.show()


    #
    def plot_reward_centered_traces(self):
        
        #
        clrs = ['blue','lightblue','red','pink']

        # load reward centered traces for the session
        if self.shuffled==False:
            fname = os.path.join(
                            self.root_dir,
                            self.animal_id,
                            self.session_ids[self.session_id],
                            'reward_centered_traces_rewardwindow_'+str(self.reward_window)+'.npy')
        else:
            fname = os.path.join(
                            self.root_dir,
                            self.animal_id,
                            self.session_ids[self.session_id],
                            'reward_centered_traces_rewardwindow_'+str(self.reward_window)+'_shuffled.npy')


        self.reward_centered_traces = np.load(fname, allow_pickle=True)

        # compute the mean of temp in the first axis
        temp = np.mean(self.reward_centered_traces,0)

        print ('input traces: ', temp.shape)

        # compute the std of temp in the first axis
        #temp_std = np.std(self.reward_centered_traces,0)

        # plot the mean and std for all the cells spacing them out by 3
        t = np.arange(temp.shape[1])/30-(self.reward_window/30)

        #
        closest_ROI_cells = self.session_contours[self.session_id]
        print ("closest_ROI_cells: ", closest_ROI_cells)
    
        #
        plt.figure()
        # looping over all suite2p cells
        traces = []
        clrs2 = []
        print ("loopin gover all suite2p cells: ", temp.shape[0])
        for k in range(temp.shape[0]):
            
            #
            temp2 = temp[k]

            #
            temp2 = temp2 - np.median(temp2)

            # loop over all the closest_ROI_cells and check if k is matching
            traces.append(temp2)
            
            #
            found_match = False
            for i in range(len(closest_ROI_cells)):
                matching_mask = closest_ROI_cells[i][0]
                matching_cell = closest_ROI_cells[i][1]

                if k == matching_cell:
                    print ('k: ', k, ' matching_cell: ', matching_cell, ' matching_mask: ', matching_mask)
                    clrs2.append(clrs[matching_mask])
                    found_match = True
                    break
 
            if found_match == False:
                clrs2.append('black')

        #
        traces = np.vstack(traces)
        clrs2 = np.vstack(clrs2)
        cell_ids = np.arange(traces.shape[0])


        #
        if self.sort_by_peak:
            peak_idxs = np.argmax(traces,1)
            
            # sort by location of ptp
            sorted_idx = np.argsort(peak_idxs)[::-1]

            # sort traces
            traces = traces[sorted_idx]
            clrs2 = clrs2[sorted_idx]
            cell_ids = cell_ids[sorted_idx]

        # plotthe traces
        ctr = 0
        labels = []
        for k in range(traces.shape[0]):
            
            #
            temp3 = traces[k]
            ptp = np.ptp(temp3)

            #
            if ptp < self.min_ptp:
                continue

            # find the argmax of the trace
            peak_idx = np.argmax(temp3)

            plt.plot(t,temp3+ctr*self.scale,
                    linewidth=3,
                    c=clrs2[k][0])
            
            # print colors
            if clrs2[k][0]!='black':
                print ('k: ', k, ' clrs2[k][0]: ', clrs2[k][0])
            
            # plot the peak using peak_idx
            plt.plot(t[peak_idx],temp3[peak_idx]+ctr*self.scale,'o',
                    c='red')


            # plot horizontal line at 0
            plt.plot(t, temp3*0+ctr*self.scale,
                     '--',
                    c='grey')    

            #
            labels.append(cell_ids[k])

            #            
            ctr+=1
        
        # 
        plt.plot([0,0],
                 [0,(ctr+1)*self.scale],
                 'r--',linewidth=3)

        # label y axis using ylabels 
        plt.yticks(np.arange(ctr)*self.scale, labels)
        
        plt.xlabel('Time (s)',fontsize=20)
        plt.suptitle(self.animal_id+' '+self.session_ids[self.session_id]+"\n # rewards: "
                     +str(self.reward_centered_traces.shape[0]),fontsize=20)

        plt.show()

    #


    # #
    # def plot_reward_centered_traces(self):
        
    #     #
    #     clrs = ['blue','lightblue','red','pink']

    #     # load reward centered traces for the session
    #     if self.shuffled==False:
    #         fname = os.path.join(
    #                         self.root_dir,
    #                         self.animal_id,
    #                         self.session_ids[self.session_id],
    #                         'reward_centered_traces_rewardwindow_'+str(self.reward_window)+'.npy')
    #     else:
    #         fname = os.path.join(
    #                         self.root_dir,
    #                         self.animal_id,
    #                         self.session_ids[self.session_id],
    #                         'reward_centered_traces_rewardwindow_'+str(self.reward_window)+'_shuffled.npy')


    #     self.reward_centered_traces = np.load(fname, allow_pickle=True)



    #     # compute the mean of temp in the first axis
    #     temp = np.mean(self.reward_centered_traces,0)

    #     print ('input traces: ', temp.shape)

    #     # compute the std of temp in the first axis
    #     #temp_std = np.std(self.reward_centered_traces,0)

    #     # plot the mean and std for all the cells spacing them out by 3
    #     t = np.arange(temp.shape[1])/30-(self.reward_window/30)

    #     #
    #     closest_ROI_cells = self.session_contours[self.session_id]
    #     print ("closest_ROI_cells: ", closest_ROI_cells)
    
    #     #
    #     plt.figure()
    #     #scale = self.scale
    #     ylabels = []
    #     # looping over all suite2p cells
    #     for k in range(temp.shape[0]):
    #         temp2 = temp[k]

    #         #temp2 = temp2 - np.median(temp2)


    #         # loop over all the closest_ROI_cells and check if k is matching
    #         for i in range(len(closest_ROI_cells)):
    #             matching_mask = closest_ROI_cells[i][0]
    #             matching_cell = closest_ROI_cells[i][1]

    #             if k == matching_cell:
    #                 print ('k: ', k, ' matching_cell: ', matching_cell, ' matching_mask: ', matching_mask)
                
    #                 # if k is the first two values in cell_ids plot using blue
    #                 plt.plot(t,temp2+k*self.scale,
    #                          linewidth=5,
    #                             c=clrs[matching_mask])

    #             else:
    #                 plt.plot(t,temp2+k*self.scale,
    #                         c='black')
    #         # plot horizontal line at 0
    #         plt.plot(t, temp2*0+k*self.scale,'--',
    #                 c='grey')    
            
    #         #
    #         ylabels.append(k*self.scale)

    #     # 
    #     plt.plot([0,0],[0,(k+1)*self.scale],'r--',linewidth=3)

    #     # label y axis using ylabels 
    #     #plt.yticks(ylabels, np.arange(temp.shape[0]))
        
    #     plt.xlabel('Time (s)',fontsize=20)
    #     plt.suptitle(self.animal_id+' '+self.session_ids[self.session_id]+"\n # rewards: "
    #                  +str(self.reward_centered_traces.shape[0]),fontsize=20)

    #     plt.show()

    #
    def load_data(self):

        #for animal_id in self.animal_ids:
        self.sessions = []
        for session_ in tqdm(self.session_ids):
            data_dir = os.path.join(
                            self.root_dir,
                            self.animal_id,
                            session_,
                            'plane0'
                            )

            #
            C = Calcium()       
            C.data_dir = data_dir

            #
            C.load_suite2p()

            C.load_footprints()

            self.sessions.append(C)

    #
    def plot_ROIs_contours(self):
        clrs = ['blue','lightblue','red','pink']
        centres = []
        for k in range(len(self.contours_ROIs)):
            temp2 = self.contours_ROIs[k][0]
            temp= temp2.copy()
            temp[:,0] = temp2[:,1]
            temp[:,1] = temp2[:,0]
            #print (temp.shape)
            plt.plot(temp[:,0],
                    temp[:,1],
                    c=clrs[k],
                    alpha=.5,
                    linewidth=6,
                    label="ROI: "+str(k))
            centre = np.mean(temp,0)
            centres.append(centre)

        # make vstack array
        centres = np.vstack(centres)

        return centres


    #
    def plot_matching_contours(self):
        
        #
        plt.figure()
        self.session_contours = []
        for k in range(len(self.sessions)):

            #
            shift = self.shifts[k]

            #
            plt.subplot(2,5,k+1)

            #
            self.session_contours.append([])

            #
            session_contour = self.sessions[k].contours

            #
            centres = self.plot_ROIs_contours()

            # loop through each session and plot contours individually
            found=False
            for p in range(len(session_contour)):
                
                #
                temp = session_contour[p]
                centre = np.median(temp,0)

                # check to see if temp centre is close to any of the centres
                dist = np.linalg.norm(centres-centre, axis=1)
                idx = np.argmin(dist)

                if dist[idx]<self.contour_ROI_max_dist:
                #print ("session: ", k, "  cell: ", p, "  closest cell: ", idx, "  dist: ", dist[idx])
                    self.session_contours[k].append([idx,p])
                    if found==False:
                        plt.plot(temp[:,0]+shift[0],
                            temp[:,1]+shift[1],
                            #c=colors[k],
                            c='black',
                            alpha=1,
                            linewidth=2,
                            label='Suite2p nearest cell'
                            )
                        found=True
                    else:
                        plt.plot(temp[:,0]+shift[0],
                            temp[:,1]+  shift[1],
                            #c=colors[k],
                            c='black',
                            alpha=1,
                            linewidth=2,
                            )
            plt.xlim(0,512)
            plt.ylim(0,512)

            if k==0:
                plt.legend()
            plt.title(self.session_ids[k])

            # 
            temp = np.vstack(self.session_contours[k])
            # sort by first column
            temp = temp[temp[:,0].argsort()]
            self.session_contours[k] = temp



        plt.suptitle(self.animal_id)
        plt.show()




def get_reward_centered_traces(session_id,
                                root_dir,
                                animal_id,
                                reward_window,
                                filter = True,
                                recompute=True):

    if session_id=='day0':
        return

    #   
    fname_out = os.path.join(root_dir,
                        animal_id,
                        session_id,
                        'reward_centered_traces_rewardwindow_'+str(reward_window)+'.npy')
    
    if os.path.exists(fname_out)==False or recompute==True:

        # load reward times            
        fname = os.path.join(root_dir,
                            animal_id,
                            session_id,
                            'results.npz')
        
        #
        data = np.load(fname, allow_pickle=True)
        rewards = data['reward_times'].T[:,::-1][:,0]
        idx = np.where(rewards>0)[0]
        reward_switches = rewards[idx]

        # load calcium traces
        fname = os.path.join(root_dir,
                            animal_id,
                            session_id,
                            'plane0',
                            'F.npy')

        F = np.load(fname, allow_pickle=True)

        #
        # filter the trace
        if filter:
            F = scipy.signal.savgol_filter(F, 31, 3)

        ########################################################

        # get f0 values first as the median of axis 1
        #print ("F shape: ", F.shape)
        f0s = np.nanmedian(F, axis=1)

        #
        F = (F-f0s[:,np.newaxis])/f0s[:,np.newaxis]


        # split c.sessions[0].F around the reward_witches time with a window of 10 seconds
        temp = []
        temp_shuffled = []
        for k in range(reward_switches.shape[0]):
            
            #
            temp1 = F[:,reward_switches[k]-reward_window:reward_switches[k]+reward_window]
            
            # make temp2 with the index shuffled
            # make a random value from 500 to -500 in F
            idx = np.random.randint(-500,F.shape[0]-500)
            temp2 = F[:,reward_switches[k]-reward_window+idx:reward_switches[k]+reward_window+idx]

            # check if the window is correct 
            if temp1.shape[1]==reward_window*2:
                temp.append(temp1)

            # check if the window is correct 
            if temp2.shape[1]==reward_window*2:
                temp_shuffled.append(temp2)


        #
        reward_centered_traces = np.array(temp)
        reward_centered_traces_shuffled = np.array(temp_shuffled)

        #
        print ("# of cells, times: ", F.shape, ", output: ", reward_centered_traces.shape)

        # save the traces
        np.save(fname_out, reward_centered_traces)
        np.save(fname_out[:-4]+'_shuffled.npy', reward_centered_traces_shuffled)
        