import json, yaml
import os
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.ticker as mtick
from sklearn.utils import shuffle
import tensorflow as tf
from keras.utils.np_utils import to_categorical
np.random.seed(0) #fix the seed to keep track of validation split

line_style = {
    'gluon_tagging': '-',
    'top_tagging':'-',
    'HV':'-',
    'gluon_tagging_ll': 'dotted',
    'top_tagging_ll':'dotted',
    'HV_ll':'dotted',
}

colors = {
    'gluon_tagging':'#d95f02',
    'top_tagging':'#1b9e77',
    'HV':'#7570b3',
    'gluon_tagging_ll':'#d95f02',
    'top_tagging_ll':'#1b9e77',
    'HV_ll':'#7570b3'
}

name_translate={
    'gluon_tagging':'QCD',
    'top_tagging':'Top quark',
    'HV': "Z'",
    'gluon_tagging_ll':'QCD Max. Likelihood',
    'top_tagging_ll':'Top quark Max. Likelihood',
    'HV_ll': "Z' Max. Likelihood"
}


nevts = -1


def SetStyle():
    from matplotlib import rc
    rc('text', usetex=True)

    import matplotlib as mpl
    rc('font', family='serif')
    rc('font', size=22)
    rc('xtick', labelsize=15)
    rc('ytick', labelsize=15)
    rc('legend', fontsize=15)

    # #
    mpl.rcParams.update({'font.size': 19})
    #mpl.rcParams.update({'legend.fontsize': 18})
    mpl.rcParams['text.usetex'] = False
    mpl.rcParams.update({'xtick.labelsize': 18}) 
    mpl.rcParams.update({'ytick.labelsize': 18}) 
    mpl.rcParams.update({'axes.labelsize': 18}) 
    mpl.rcParams.update({'legend.frameon': False}) 
    mpl.rcParams.update({'lines.linewidth': 2})
    
    import matplotlib.pyplot as plt
    import mplhep as hep
    hep.set_style(hep.style.CMS)
    
    # hep.style.use("CMS") 

    
def SetGrid(ratio=True):
    fig = plt.figure(figsize=(9, 9))
    if ratio:
        gs = gridspec.GridSpec(2, 1, height_ratios=[3,1]) 
        gs.update(wspace=0.025, hspace=0.1)
    else:
        gs = gridspec.GridSpec(1, 1)
    return fig,gs

def SetFig(xlabel,ylabel):
    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(1, 1) 
    ax0 = plt.subplot(gs[0])
    ax0.yaxis.set_ticks_position('both')
    ax0.xaxis.set_ticks_position('both')
    ax0.tick_params(direction="in",which="both")    
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel(xlabel,fontsize=18)
    plt.ylabel(ylabel,fontsize=18)
    
    ax0.minorticks_on()
    return fig, ax0

    
def PlotRoutine(feed_dict,xlabel='',ylabel='',reference_name='gen'):
    assert reference_name in feed_dict.keys(), "ERROR: Don't know the reference distribution"
    
    fig,gs = SetGrid() 
    ax0 = plt.subplot(gs[0])
    plt.xticks(fontsize=0)
    ax1 = plt.subplot(gs[1],sharex=ax0)

    for ip,plot in enumerate(feed_dict.keys()):
        if 'steps' in plot or 'r=' in plot:
            ax0.plot(np.mean(feed_dict[plot],0),label=plot,marker=line_style[plot],color=colors[plot],lw=0)
        else:
            ax0.plot(np.mean(feed_dict[plot],0),label=plot,linestyle=line_style[plot],color=colors[plot])
        if reference_name!=plot:
            ratio = 100*np.divide(np.mean(feed_dict[reference_name],0)-np.mean(feed_dict[plot],0),np.mean(feed_dict[reference_name],0))
            #ax1.plot(ratio,color=colors[plot],marker='o',ms=10,lw=0,markerfacecolor='none',markeredgewidth=3)
            if 'steps' in plot or 'r=' in plot:
                ax1.plot(ratio,color=colors[plot],markeredgewidth=1,marker=line_style[plot],lw=0)
            else:
                ax1.plot(ratio,color=colors[plot],linewidth=2,linestyle=line_style[plot])
                
        
    FormatFig(xlabel = "", ylabel = ylabel,ax0=ax0)
    ax0.legend(loc='best',fontsize=16,ncol=1)

    plt.ylabel('Difference. (%)')
    plt.xlabel(xlabel)
    plt.axhline(y=0.0, color='r', linestyle='--',linewidth=1)
    plt.axhline(y=10, color='r', linestyle='--',linewidth=1)
    plt.axhline(y=-10, color='r', linestyle='--',linewidth=1)
    plt.ylim([-100,100])

    return fig,ax0

class ScalarFormatterClass(mtick.ScalarFormatter):
    #https://www.tutorialspoint.com/show-decimal-places-and-scientific-notation-on-the-axis-of-a-matplotlib-plot
    def _set_format(self):
        self.format = "%1.1f"


def FormatFig(xlabel,ylabel,ax0):
    #Limit number of digits in ticks
    # y_loc, _ = plt.yticks()
    # y_update = ['%.1f' % y for y in y_loc]
    # plt.yticks(y_loc, y_update) 
    ax0.set_xlabel(xlabel,fontsize=20)
    ax0.set_ylabel(ylabel)

def WriteText(xpos,ypos,text,ax0):

    plt.text(xpos, ypos,text,
             horizontalalignment='center',
             verticalalignment='center',
             transform = ax0.transAxes, fontsize=25, fontweight='bold')


def HistRoutine(feed_dict,
                xlabel='',ylabel='',
                reference_name='Geant',
                logy=False,binning=None,
                fig = None, gs = None,
                plot_ratio= True,
                idx = None,
                label_loc='best'):
    assert reference_name in feed_dict.keys(), "ERROR: Don't know the reference distribution"

    if fig is None:
        fig,gs = SetGrid(plot_ratio) 
    ax0 = plt.subplot(gs[0])
    if plot_ratio:
        plt.xticks(fontsize=0)
        ax1 = plt.subplot(gs[1],sharex=ax0)
        
    
    if binning is None:
        binning = np.linspace(np.quantile(feed_dict[reference_name],0.0),np.quantile(feed_dict[reference_name],1),20)
        
    xaxis = [(binning[i] + binning[i+1])/2.0 for i in range(len(binning)-1)]
    reference_hist,_ = np.histogram(feed_dict[reference_name],bins=binning,density=True)
    maxy = np.max(reference_hist)
    print(maxy)
    for ip,plot in enumerate(feed_dict.keys()):
        dist,_,_=ax0.hist(feed_dict[plot],bins=binning,label=name_translate[plot],linestyle=line_style[plot],color=colors[plot],density=True,histtype="step")
        if plot_ratio:
            if reference_name!=plot:
                ratio = 100*np.divide(reference_hist-dist,reference_hist)
                ax1.plot(xaxis,ratio,color=colors[plot],marker='o',ms=10,lw=0,markerfacecolor='none',markeredgewidth=3)
        
    ax0.legend(loc=label_loc,fontsize=14,ncol=2)
    ax0.set_ylim(top=2.2*maxy)
    if logy:
        ax0.set_yscale('log')



    if plot_ratio:
        FormatFig(xlabel = "", ylabel = ylabel,ax0=ax0)
        plt.ylabel('Difference. (%)')
        plt.xlabel(xlabel)
        plt.axhline(y=0.0, color='r', linestyle='-',linewidth=1)
        # plt.axhline(y=10, color='r', linestyle='--',linewidth=1)
        # plt.axhline(y=-10, color='r', linestyle='--',linewidth=1)
        plt.ylim([-100,100])
    else:
        FormatFig(xlabel = xlabel, ylabel = ylabel,ax0=ax0)
    
    return fig,gs, binning


def LoadJson(file_name):
    import json,yaml
    JSONPATH = os.path.join(file_name)
    return yaml.safe_load(open(JSONPATH))

def SaveJson(save_file,data):
    with open(save_file,'w') as f:
        json.dump(data, f)


def DataLoader(data_path,labels,
               part,
               rank=0,size=1,
               use_train=True,
               batch_size=64,make_tf_data=True):
    particles = []
    jets = []

    def _preprocessing(particles,jets,save_json=False):
        num_part = particles.shape[1]
        particles=particles.reshape(-1,particles.shape[-1]) #flatten

        def _logit(x):                            
            alpha = 1e-6
            x = alpha + (1 - 2*alpha)*x
            return np.ma.log(x/(1-x)).filled(0)

        if save_json:
            mask = particles[:,-1]
            mean_particle = np.average(particles[:,:-1],axis=0,weights=mask)
            data_dict = {
                'max_jet':np.max(jets,0).tolist(),
                'min_jet':np.min(jets,0).tolist(),
                'mean_jet':np.mean(jets,0).tolist(),
                'std_jet':np.std(jets,0).tolist(),
                'max_particle':np.max(particles[:,:-1],0).tolist(),
                'min_particle':np.min(particles[:,:-1],0).tolist(),
                'mean_particle':mean_particle.tolist(),
                'std_particle':np.sqrt(np.average((particles[:,:-1] - mean_particle)**2,axis=0,weights=mask)).tolist(),
                
            }                
            
            SaveJson('preprocessing_{}.json'.format(part),data_dict)
        else:
            data_dict = LoadJson('preprocessing_{}.json'.format(part))

        #normalize
        jets = np.ma.divide(jets-data_dict['mean_jet'],np.array(data_dict['std_jet']))
        particles[:,:-1] = np.ma.divide(particles[:,:-1]-data_dict['mean_particle'],np.array(data_dict['std_particle']))
        
        #particles[:,:-1] += np.random.uniform(-0.5e-3,0.5e-3,size=particles[:,:-1].shape)

        # jets = np.ma.divide(jets-data_dict['min_jet'],np.array(data_dict['max_jet']) - data_dict['min_jet'])
        # particles[:,:-1] = np.ma.divide(particles[:,:-1]-data_dict['min_particle'],np.array(data_dict['max_particle']) - data_dict['min_particle'])

        
        # jets = np.ma.divide(jets-data_dict['min_jet'],np.array(data_dict['max_jet'])- data_dict['min_jet']).filled(0)
        # jets = 2*jets -1.
        # particles[:,:-1]= np.ma.divide(particles[:,:-1]-data_dict['min_particle'],np.array(data_dict['max_particle'])- data_dict['min_particle']).filled(0)
        # particles[:,:-1] = 2*particles[:,:-1] -1.

        
        
        particles = particles.reshape(jets.shape[0],num_part,-1)
        return particles.astype(np.float32),jets.astype(np.float32)
            
            
    for label in labels:        
        with h5.File(os.path.join(data_path,label),"r") as h5f:
            ntotal = h5f['jet_features'][:].shape[0]

            if use_train:
                particle = h5f['particle_features'][rank:int(0.7*ntotal):size].astype(np.float32)
                jet = h5f['jet_features'][rank:int(0.7*ntotal):size].astype(np.float32)

            else:
                #load evaluation data
                particle = h5f['particle_features'][int(0.7*ntotal):].astype(np.float32)
                jet = h5f['jet_features'][int(0.7*ntotal):].astype(np.float32)


            particles.append(particle)
            jets.append(jet)

    particles = np.concatenate(particles)
    jets = np.concatenate(jets)
    particles,jets = shuffle(particles,jets, random_state=0)
    
    data_size = jets.shape[0]

    particles,jets = _preprocessing(particles,jets)

    if use_train:
        train_particles = particles[:int(0.8*data_size)]
        train_jets = jets[:int(0.8*data_size)]
        
        test_particles = particles[int(0.8*data_size):]
        test_jets = jets[int(0.8*data_size):]
        
        if make_tf_data:
            def _prepare_batches(particles,jets):            
                nevts = jets.shape[0]
                tf_jet = tf.data.Dataset.from_tensor_slices(jets)
                
                mask = np.expand_dims(particles[:,:,-1],-1)
                masked = particles[:,:,:-1]*mask
                tf_part = tf.data.Dataset.from_tensor_slices(masked)
                tf_mask = tf.data.Dataset.from_tensor_slices(mask)
                tf_zip = tf.data.Dataset.zip((tf_part, tf_jet,tf_mask))
                return tf_zip.shuffle(nevts).repeat().batch(batch_size)
    
            train_data = _prepare_batches(train_particles,train_jets)
            test_data  = _prepare_batches(test_particles,test_jets)    
            return data_size, train_data,test_data
        else:
            mask_train = np.expand_dims(train_particles[:,:,-1],-1)
            mask_test = np.expand_dims(test_particles[:,:,-1],-1)
            
            return train_particles[:,:,:-1]*mask_train,test_particles[:,:,:-1]*mask_test
                
    else:
        #nevts = particles.shape[0]
        nevts = 40000
        mask = np.expand_dims(particles[:nevts,:,-1],-1)
        return particles[:nevts,:,:-1]*mask,jets[:nevts],mask
