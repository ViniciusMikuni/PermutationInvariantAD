import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib import gridspec
import argparse
import h5py as h5
import os
import utils
import tensorflow as tf
from GSGM_uniform import GSGM
import time
import gc
import sys
from sklearn.metrics import roc_curve, auc


def evaluate(model,particles,jets,mask,nsplit=1):
    part_split = np.array_split(particles,nsplit)
    jet_split = np.array_split(jets,nsplit)
    mask_split = np.array_split(mask,nsplit)
    
    likelihoods_jet = []
    likelihoods_part = []
    Ns = []
    start = time.time()
    for i in range(nsplit):
        print(i,part_split[i].shape[0])
        #if i> 0:break
        ll_part = []
        ll_jet = []
        for _ in range(1):
            llp,llj = model.get_likelihood(part_split[i],jet_split[i],mask_split[i])
            ll_part.append(llp)
            ll_jet.append(llj)
        ll_part = np.median(ll_part,0)
        ll_jet = np.median(ll_jet,0)
        
        likelihoods_part.append(ll_part)
        likelihoods_jet.append(ll_jet)
        Ns.append(np.sum(mask_split[i],(1,2)))
        
    likelihoods_part = np.concatenate(likelihoods_part)
    likelihoods_jet = np.concatenate(likelihoods_jet)
    Ns = np.concatenate(Ns)
    
    end = time.time()
    print("Time for sampling {} events is {} seconds".format(particles.shape[0],end - start))
    
    return {'ll_part':likelihoods_part,'ll_jet': likelihoods_jet,'N': Ns}


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


    utils.SetStyle()


    parser = argparse.ArgumentParser()

    parser.add_argument('--data_folder', default='/global/cfs/cdirs/m3929/TOP', help='Folder containing data and MC files')
    parser.add_argument('--plot_folder', default='../plots', help='Folder to save results')
    parser.add_argument('--config', default='config_AD.json', help='Training parameters')

    parser.add_argument('--ll', action='store_true', default=False,help='Load model training with MLE')
    parser.add_argument('--npart', default=100,type=int, help='Which particle is the anomaly')
    parser.add_argument('--nshuffle', default=100,type=int, help='Which particle is the anomaly')


    flags = parser.parse_args()
    npart = flags.npart
    config = utils.LoadJson(flags.config)
    model_name = config['MODEL_NAME']
    if flags.ll:
        add_text = '_ll'
    else:
        add_text = ''
        
    model_name += add_text
    processes = ['gluon_tagging','top_tagging','HV']
    #processes = ['HV']
    


    model_gluon = GSGM(config=config,npart=npart,particle='gluon_tagging')
    checkpoint_folder_gluon = '../checkpoints_{}/checkpoint'.format(model_name+ '_gluon_tagging')
    model_gluon.load_weights('{}'.format(checkpoint_folder_gluon)).expect_partial()
    nll_list = {}
    fig,gs = utils.SetGrid(ratio=False) 
    ax0 = plt.subplot(gs[0])
    
    for process in processes:
        print(process)
        particles,jets,mask = utils.DataLoader(flags.data_folder,
                                               labels=['%s.h5'%process],
                                               part='gluon_tagging', #name of the preprocessing file, the same for all datasets
                                               use_train=False,
                                               make_tf_data=False)

        #pick just a single event
        particles = particles[3:4]
        jets = jets[3:4]
        mask = mask[3:4]
            
        #Repeat the same entry multiple times to evaluate the fluctuation of the logp
        nrepeat = 10
        particles = np.repeat(particles,nrepeat,0)
        jets = np.repeat(jets,nrepeat,0)
        mask = np.repeat(mask,nrepeat,0)

        nll_list[process] = []

        for _ in range(flags.nshuffle):
            perm = np.random.permutation(range(npart)).reshape(1,npart,1)
            nll = evaluate(model_gluon,np.take_along_axis(particles,perm,1),
                            jets,np.take_along_axis(mask,perm,1))
            nll_list[process].append(np.array([-np.mean(nll['ll_part']),np.std(nll['ll_part'])]))

        nll_list[process] = np.reshape(nll_list[process],(flags.nshuffle,2))
        print(nll_list[process].shape)
        ax0.errorbar(range(flags.nshuffle),
                     nll_list[process][:,0],                     
                     yerr=nll_list[process][:,1],
                     ls='none',
                     label=utils.name_translate[process],
                     marker='o',color=utils.colors[process])
    
    ax0.legend(loc='best',fontsize=16,ncol=1)
    ax0.set_ylim(bottom=38,top=75)
    utils.FormatFig(xlabel = 'Permutation index', ylabel = 'Negative Log-Likelihood',ax0=ax0) 
    fig.savefig('{}/nll_permutations.pdf'.format(flags.plot_folder),bbox_inches='tight')
