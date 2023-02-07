import numpy as np
# Visualisation
import matplotlib.pyplot as plt
import numpy
import os
    
import scipy.ndimage
#from matplotlib_scalebar.scalebar import ScaleBar

def analyse_mouse(root_dir, mice, date, filename, bin_size):
    file = os.path.join(root_dir,mice, date, 'data',filename)
    data = np.load(file, allow_pickle=True)
    #Eliminate errors from the beginning
    rewards = data["reward_times"][1][data["reward_times"][1] > 20]
    time_rewards = rewards/1800
    learning = 30*60*len(rewards)/len(data["ttl_n_computed"])
    rewards_early = len(time_rewards[time_rewards < bin_size])
    return time_rewards, learning, rewards_early


def plot_histogram_mouse(mouse, dates, root_dir, filename, color, out_dir, cohort, bin_size = 8):
    fig, axs = plt.subplots(1, len(dates), figsize=(2*len(dates),2), sharey = True)
    fig.suptitle(mouse,y=1.2, fontsize=14)
    learning_mouse = []
    learning_curve = []
    early_mouse = []
    for i in range(len(dates)):
        time_rewards, learning, rewards_early = analyse_mouse(root_dir, mouse, dates[i], filename, bin_size)
        n, bins, patches = axs[i].hist(time_rewards, bins= np.arange(0,55,bin_size), color=color)
        axs[i].set_ylim(0,17)
        axs[i].set_xlim(0,50)
        axs[i].set_title("Session "+str(i+1))
        axs[0].set_ylabel("Rewards")
        axs[i].set_xlabel("Time (min)")
        learning_mouse.append(30*60*len(time_rewards)/90000)
        learning_curve.append(n)
        early_mouse.append(rewards_early)
    fig.savefig(os.path.join(out_dir,cohort+'_'+mouse+'_day_curves.png'), dpi=600, bbox_inches = 'tight')
    return learning_mouse, learning_curve, early_mouse

def plot_learning_curve(mice, learning_curves, mice_type, out_dir, cohort, bin_size, normalise=True):
    fig, axs = plt.subplots(1, len(mice), figsize=(2*len(mice),2), sharey = True)
    viridis = plt.cm.get_cmap('viridis', len(max(learning_curves,key=len)))
    cmap = plt.cm.get_cmap('Set2')
    bins= np.arange(0,55,bin_size)
    for mouse in range(len(mice)):
        for date in range(len(learning_curves[mouse])):
            if normalise:
                axs[mouse].plot(bins[:-1], learning_curves[mouse][date]/(learning_curves[mouse][date][0]), color=viridis(date))
            else:
                axs[mouse].plot(bins[:-1], learning_curves[mouse][date], color=viridis(date))
            axs[mouse].set_title(mice[mouse]+' '+mice_type[mouse], color=cmap(mouse))
        axs[mouse].set_xlabel('Time (min)')
        axs[mouse].set_xlim(0,30)
        if normalise:
            axs[mouse].axhline(y = 1, color = 'slategray', linestyle = '--', label= 'Expected random rewards')
    #axs[0].legend([0,1,2,3,4], loc='center left', bbox_to_anchor=(4, 0.5))
    
    axs[0].legend(['S'+str(i) for i in np.arange(1,len(max(learning_curves,key=len))+1)], loc='center left', bbox_to_anchor=(len(mice)+1, 0.5), frameon=False)
    norm = 'absolute'
    if normalise:
        norm = 'norm'
    axs[0].set_ylabel('Rewards '+norm)
    fig.savefig(os.path.join(out_dir,cohort+'_'+norm+'_rewards.png'), dpi=600, bbox_inches = 'tight')
    
def plot_rewards_cohort(mice, learning, early, mice_type, out_dir, cohort, cmap, bin_size, normalise=True):
    fig = plt.figure(figsize=(3,3))
    x = ['S'+str(i) for i in np.arange(1,len(max(learning,key=len))+1)]

    for i in range(len(mice)):
        if normalise:
            plt.plot(x[:len(learning[i])],np.array(learning[i]), label = mice[i]+' '+mice_type[i], c=cmap(i), lw=3)
        else:
            plt.plot(x[:len(learning[i])],np.array(learning[i])/(np.array(early[i])/bin_size), label = mice[i]+' '+mice_type[i], c=cmap(i), lw=3)
            
    plt.axhline(y = 1, color = 'slategray', linestyle = '--', label= 'Expected random rewards')

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    plt.xlabel("Session")
    plt.ylim(0,2)
    plt.title(cohort)
    if normalise:
        plt.ylabel("Rewards normalised")
        fig.savefig(os.path.join(out_dir,cohort+'_norm_rewards.png'), dpi=600, bbox_inches = 'tight')
    else:
        plt.ylabel("Rewards/min")
        fig.savefig(os.path.join(out_dir,cohort+'_abs_rewards.png'), dpi=600, bbox_inches = 'tight')


    
    


def filter_data_make_std_map(data, subsample=10):

    #
    data_sparse = data[::subsample]
    print("data into analysis: ", data_sparse.shape)

    # filter once to remove much of the white noise
    if True:
        sigma = 1
        order = 0
        print(" gaussian filter width: ", sigma, ", order: ", order)
        data_sparse = scipy.ndimage.gaussian_filter(data_sparse,
                                                    sigma,
                                                    order)

    std = np.std(data_sparse, axis=0)

    return std
    
    
def get_image(root_dir, mouse, date):
    calibration_filename = 'Image_001_001.raw'
    file_img = os.path.join(root_dir,mouse, date,'calibration', calibration_filename)
    img = np.fromfile(file_img, dtype='uint16')
    img = img.reshape(-1, 512, 512)
    std_map = filter_data_make_std_map(img, subsample=10)
    file = os.path.join(root_dir,mouse, date,'rois_pixels_and_thresholds.npz')
    data = np.load(file, allow_pickle=True)
    return std_map, data['ensemble1_contours'], data['ensemble2_contours']

def plot_image(img, contours1, contours2, mouse, date,cohort, out_dir):
    fig, ax = plt.subplots(figsize=(5,5))
    ax.imshow(img, cmap='gray',  vmin=100, vmax=1000)
    ax.set_xticks([])
    ax.set_yticks([])
    scalebar = ScaleBar(0.791, "um", length_fraction=0.25, color='white',box_alpha=0, location='lower right')
    ax.plot(contours1[0][0][:,0],contours1[0][0][:,1], c='royalblue', lw=3)
    ax.plot(contours1[1][0][:,0],contours1[1][0][:,1], c='royalblue', lw=3)
    ax.plot(contours2[0][0][:,0],contours2[0][0][:,1], c='orchid', lw=3)
    ax.plot(contours2[1][0][:,0],contours2[1][0][:,1], c='orchid', lw=3)
    ax.add_artist(scalebar)
    fig.savefig(os.path.join(out_dir,cohort+'_'+mouse+date+'_img_contours.png'), dpi=600, bbox_inches = 'tight')

def plot_images_mouse(mouse, dates, cohort, root_dir, out_dir):
    fig, axs = plt.subplots(1, len(dates), figsize=(2*len(dates),2), sharex = True, sharey = True)
    for i in range(len(dates)):
        img, contours1, contours2 = get_image(root_dir, mouse, dates[i])
        axs[i].imshow(img, cmap='gray',  vmin=100, vmax=1000)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        scalebar = ScaleBar(0.791, "um", length_fraction=0.25, color='white',box_alpha=0, location='lower right')
        axs[i].plot(contours1[0][0][:,0],contours1[0][0][:,1], c='royalblue', lw=2)
        axs[i].plot(contours1[1][0][:,0],contours1[1][0][:,1], c='royalblue', lw=2)
        axs[i].plot(contours2[0][0][:,0],contours2[0][0][:,1], c='orchid', lw=2)
        axs[i].plot(contours2[1][0][:,0],contours2[1][0][:,1], c='orchid', lw=2)
        axs[i].add_artist(scalebar)
    fig.savefig(os.path.join(out_dir,cohort+'_'+mouse+'_img_contours.png'), dpi=600, bbox_inches = 'tight')
    
    
def plot_calcium_mouse(root_dir, mouse, dates, filename):
    fig, axs = plt.subplots(len(dates),1, figsize=(10,2*len(dates)), sharex = True)
    fig.suptitle(mouse,y=0.91, fontsize=14)
    for i in range(len(dates)):
        file = os.path.join(root_dir,mouse, dates[i],'data', filename)
        data = np.load(file, allow_pickle=True)
        x_array = np.arange(0,len(data["rois_traces_smooth1"][0][200:]))/30
        axs[i].plot(x_array,data["rois_traces_smooth1"][0][200:]/max(data["rois_traces_smooth1"][0][200:]), c='royalblue')
        axs[i].plot(x_array,1+data["rois_traces_smooth1"][1][200:]/max(data["rois_traces_smooth1"][1][200:]), c='royalblue')
        axs[i].plot(x_array,2+data["rois_traces_smooth2"][0][200:]/max(data["rois_traces_smooth1"][0][200:]), c='orchid')
        axs[i].plot(x_array,3+data["rois_traces_smooth1"][1][200:]/max(data["rois_traces_smooth1"][1][200:]), c='orchid')
        axs[i].set_xlim(0,90000/30)
        axs[i].set_ylim(-0.5,4.2)
        axs[len(dates)-1].set_xlabel('Time [s]')
        axs[i].set_ylabel('ΔF/F')
    fig.savefig(os.path.join(out_dir,cohort+'_'+mouse+'_calciumtraces.png'), dpi=600, bbox_inches = 'tight')

def plot_features_mouse(root_dir, mouse, dates, filename,cohort, out_dir):
    fig, axs = plt.subplots(len(dates),1, figsize=(10,2*len(dates)), sharex = True)
    fig.suptitle(mouse,y=0.91, fontsize=14)

    #
    for i in range(len(dates)):
        file = os.path.join(root_dir,mouse, dates[i],'data', filename)
        data = np.load(file, allow_pickle=True)
        offset = 200
        x_array = np.arange(0,len(data["rois_traces_smooth1"][0][offset:]))/30
        axs[i].plot(x_array,data["rois_traces_smooth1"][0][offset:]/max(data["rois_traces_smooth1"][0][offset:]), c='royalblue')
        axs[i].plot(x_array,1+data["rois_traces_smooth1"][1][offset:]/max(data["rois_traces_smooth1"][1][offset:]), c='royalblue')
        axs[i].plot(x_array,2+data["rois_traces_smooth2"][0][offset:]/max(data["rois_traces_smooth1"][0][offset:]), c='orchid')
        axs[i].plot(x_array,3+data["rois_traces_smooth1"][1][offset:]/max(data["rois_traces_smooth1"][1][offset:]), c='orchid')
        
        axs[i].plot(data['ttl_times'], 6+data['ensemble_diff_array'][0:len(data['ttl_times'])], label='E1-E2', c='slategray')
        rewards = data["reward_times"][1][data["reward_times"][1] > 20]
        axs[i].scatter(rewards/30,8*np.ones(len(rewards)), alpha= 0.8, c='darkcyan', label='rewards')
        lick_detector  = data['lick_detector_abstime']
        result = np. where(lick_detector >3)
        licks = result[0]/(30*33.425)
        axs[i].scatter(licks,9*np.ones(len(licks)), alpha= 0.01, c='orange', label='lick detector')

        axs[i].set_xlim(0,90000/30)
        axs[i].set_ylim(-0.5,9)
        axs[len(dates)-1].set_xlabel('Time [s]')
        axs[i].set_ylabel('ΔF/F')
    fig.savefig(os.path.join(out_dir,cohort+'_'+mouse+'_features.png'), dpi=600, bbox_inches = 'tight')
