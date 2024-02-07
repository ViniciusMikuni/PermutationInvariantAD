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
from deepsets_cond import DeepSetsAttClass
import time
import gc
import sys
from sklearn.metrics import roc_curve, auc


def evaluate(model,particles,jets,mask,idx=0,nsplit=5):
    part_split = np.array_split(particles,nsplit)
    jet_split = np.array_split(jets,nsplit)
    mask_split = np.array_split(mask,nsplit)

    start = time.time()
    print("Split size: {}".format(jet_split[idx].shape[0]))
    likelihoods_part,likelihoods_jet = model.get_likelihood(
        part_split[idx],jet_split[idx],mask_split[idx])
    Ns = np.sum(mask_split[idx],(1,2))
    
    end = time.time()
    print("Time for sampling {} events is {} seconds".format(particles.shape[0],end - start))
    
    return {'ll_part':likelihoods_part,'ll_jet': likelihoods_jet,'N': Ns}

def evaluate_classifier(num_feat,checkpoint_folder,data_path):
    #load the model
    from tensorflow import keras
    inputs, outputs = DeepSetsAttClass(
        num_feat,
        num_heads=2,
        num_transformer = 6,
        projection_dim = 128,
    )
    model = keras.Model(inputs=inputs,outputs=outputs)
    model.load_weights('{}'.format(checkpoint_folder)).expect_partial()
    
    #load the data
    data_bkg, _, _ = utils.DataLoader(data_path,
                                      ['gluon_tagging.h5'],
                                      use_train=False,
                                      make_tf_data = False,
    )

    data_sig,_,_ = utils.DataLoader(data_path,
                                    ['top_tagging.h5'],
                                    use_train=False,
                                    make_tf_data = False,
    )

    labels = np.concatenate([np.zeros(data_bkg.shape[0]),np.ones(data_sig.shape[0])],0)
    pred = model.predict(np.concatenate([data_bkg,data_sig],0))
    return labels,pred
    
    



if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


    utils.SetStyle()


    parser = argparse.ArgumentParser()

    #parser.add_argument('--data_folder', default='/global/cfs/cdirs/m3929/TOP', help='Folder containing data and MC files')
    parser.add_argument('--data_folder', default='/pscratch/sd/v/vmikuni/TOP/GSGM', help='Folder containing data and MC files')
    parser.add_argument('--plot_folder', default='../plots', help='Folder to save results')
    parser.add_argument('--config', default='config_AD.json', help='Training parameters')

    parser.add_argument('--sample', action='store_true', default=False,help='Sample from the generative model')
    parser.add_argument('--nidx', default=0, type=int,help='Parallel sampling of the data')

    parser.add_argument('--sup', action='store_true', default=False,help='Plot only the ROC for classifier and density ratio')
    parser.add_argument('--ll', action='store_true', default=False,help='Load Max LL training model')
    parser.add_argument('--npart', default=100,type=int, help='Which particle is the anomaly')


    flags = parser.parse_args()
    npart = flags.npart
    config = utils.LoadJson(flags.config)
    model_name = config['MODEL_NAME']
    if flags.ll:
        model_name+='_ll'

    processes = ['gluon_tagging','top_tagging','HV']
    
    
    if flags.sample:    
        nll_qcd = {}
        model_gluon = GSGM(config=config,npart=npart,particle='gluon_tagging',ll_training=flags.ll)
        checkpoint_folder_gluon = '../checkpoints_{}/checkpoint'.format(model_name+ '_gluon_tagging')
        model_gluon.load_weights('{}'.format(checkpoint_folder_gluon)).expect_partial()




        model_top = GSGM(config=config,npart=npart,particle='top_tagging',ll_training=flags.ll)
        checkpoint_folder_top = '../checkpoints_{}/checkpoint'.format(model_name+ '_top_tagging')
        model_top.load_weights('{}'.format(checkpoint_folder_top)).expect_partial()
        nll_top = {}

        for process in processes:
            print(process)
            particles,jets,mask = utils.DataLoader(flags.data_folder,
                                                   labels=['%s.h5'%process],
                                                   use_train=False,
                                                   make_tf_data=False)

            print("Loaded {} events".format(jets.shape[0]))
            nll_qcd[process] = evaluate(model_gluon,particles,jets,mask,flags.nidx)
            nll_top[process] = evaluate(model_top,particles,jets,mask,flags.nidx)
            
            for ll in nll_qcd[process]:
                print("Avg log(p) gluon model ll {}: {}".format(ll,np.mean(nll_qcd[process][ll])))
                print("Avg log(p) top model ll {}: {}".format(ll,np.mean(nll_top[process][ll])))


        with h5.File(os.path.join(flags.data_folder,model_name+ '_gluon_tagging_{}.h5'.format(flags.nidx)),"w") as h5f:
            for process in processes:
                for ll in nll_qcd[process]:
                    dset = h5f.create_dataset("{}_{}".format(process,ll), data=nll_qcd[process][ll])

        with h5.File(os.path.join(flags.data_folder,model_name+ '_top_tagging_{}.h5'.format(flags.nidx)),"w") as h5f:
            for process in processes:
                for ll in nll_top[process]:
                    dset = h5f.create_dataset("{}_{}".format(process,ll), data=nll_top[process][ll])

    else:
        nll_qcd = {}
        
        with h5.File(os.path.join(flags.data_folder,model_name + '_gluon_tagging.h5'),"r") as h5f:
            for process in processes:
                nll_qcd[process] = {}
                for ll in ['ll_part','ll_jet','N']: 
                    nll_qcd[process][ll] = h5f["{}_{}".format(process,ll)][:]
                    
        with h5.File(os.path.join(flags.data_folder,model_name + '_gluon_tagging_HV.h5'),"r") as h5f:
            for ll in ['ll_part','ll_jet','N']: 
                nll_qcd['HV'][ll] = h5f["HV_orig_{}".format(ll)][:]
                    
                

        #LL training
        with h5.File(os.path.join(flags.data_folder,model_name + '_ll_gluon_tagging.h5'),"r") as h5f:
            for process in processes:
                nll_qcd[process+'_ll'] = {}
                for ll in ['ll_part','ll_jet','N']: 
                    nll_qcd[process+'_ll'][ll] = h5f["{}_{}".format(process,ll)][:]
                


        nll_top = {}
        with h5.File(os.path.join(flags.data_folder,model_name + '_top_tagging.h5'),"r") as h5f:
            for process in processes:
                nll_top[process] = {}
                for ll in ['ll_part','ll_jet','N']: 
                    nll_top[process][ll] = h5f["{}_{}".format(process,ll)][:]


        with h5.File(os.path.join(flags.data_folder,model_name + '_top_tagging_HV.h5'),"r") as h5f:
            for ll in ['ll_part','ll_jet','N']: 
                nll_top['HV'][ll] = h5f["HV_orig_{}".format(ll)][:]
        


        with h5.File(os.path.join(flags.data_folder,model_name + '_ll_top_tagging.h5'),"r") as h5f:
            for process in processes:
                nll_top[process+'_ll'] = {}
                for ll in ['ll_part','ll_jet','N']: 
                    nll_top[process+'_ll'][ll] = h5f["{}_{}".format(process,ll)][:]
                

        #Define the anomaly detection score, independent from the likelihood of the samples!

        nll_qcd_anomaly = {}
        nll_top_anomaly = {}
        for process in processes:
            for ll in ['_ll','']: #standard and likelihood training

                #AD Score
                nll_qcd_anomaly[process+ll] =  -nll_qcd[process+ll]['ll_jet'] - nll_qcd[process+ll]['ll_part']/nll_qcd[process+ll]['N']
                nll_top_anomaly[process+ll] =  -nll_top[process+ll]['ll_jet'] - nll_top[process+ll]['ll_part']/nll_top[process+ll]['N']

                #NLL                
                # nll_qcd_anomaly[process+ll] = - nll_qcd[process+ll]['ll_jet'] - nll_qcd[process+ll]['ll_part']
                # nll_top_anomaly[process+ll] = - nll_top[process+ll]['ll_jet']-  nll_top[process+ll]['ll_part']

            
                print(process)
                print("Avg anomaly score gluon model: {}".format(np.mean(nll_qcd_anomaly[process+ll])))
                print("Avg anomaly score top model: {}".format(np.mean(nll_top_anomaly[process+ll])))
                
        fig,ax,_ = utils.HistRoutine(nll_qcd_anomaly,plot_ratio=False,
                                     xlabel='Anomaly Score',
                                     #xlabel='Negative Log-likelihood',
                                     #binning = np.linspace(4,20,30),
                                     binning = np.linspace(0,250,30),
                                     ylabel='Normalized events',
                                     reference_name='gluon_tagging')
    
        fig.savefig('{}/nll_qcd_AD_nll.pdf'.format(flags.plot_folder),bbox_inches='tight')



        fig,ax,_ = utils.HistRoutine(nll_top_anomaly,plot_ratio=False,
                                     binning = np.linspace(4,20,30),
                                     #binning = np.linspace(0,250,30),
                                     #xlabel='Negative Log-likelihood',
                                     xlabel='Anomaly Score',
                                     ylabel='Normalized events',
                                     reference_name='top_tagging')
    
        fig.savefig('{}/nll_top_nll.pdf'.format(flags.plot_folder),bbox_inches='tight')
        
        fig,gs = utils.SetFig("True positive rate","Fake Rate")


        if flags.sup:        
            #Density ratio
            
            llr_gluon = - nll_qcd['gluon_tagging']['ll_part'] - nll_qcd['gluon_tagging']['ll_jet']  + nll_top['gluon_tagging']['ll_part'] + nll_top['gluon_tagging']['ll_jet']
            llr_top = - nll_qcd['top_tagging']['ll_part'] - nll_qcd['top_tagging']['ll_jet']  + nll_top['top_tagging']['ll_part'] + nll_top['top_tagging']['ll_jet']
                        
        
            labels = np.concatenate([np.zeros(llr_gluon.shape[0]),np.ones(llr_top.shape[0])],0)
            likelihoods = np.concatenate([llr_gluon,llr_top],0)
            fpr, tpr, _ = roc_curve(labels,likelihoods, pos_label=1)
            print("Density ratio AUC: {}".format(auc(fpr, tpr)))
            
            plt.plot(tpr,fpr,label="Ratio of Densities",
                     color='black',
                     linestyle=utils.line_style['top_tagging'])
            
            # MLE Density
            
            llr_gluon_ll = - nll_qcd['gluon_tagging_ll']['ll_part'] - nll_qcd['gluon_tagging_ll']['ll_jet']  + nll_top['gluon_tagging_ll']['ll_part'] + nll_top['gluon_tagging_ll']['ll_jet']
            llr_top_ll = - nll_qcd['top_tagging_ll']['ll_part'] - nll_qcd['top_tagging_ll']['ll_jet']  + nll_top['top_tagging_ll']['ll_part'] + nll_top['top_tagging_ll']['ll_jet']

            
            likelihoods = np.concatenate([llr_gluon_ll,llr_top_ll],0)
            fpr, tpr, _ = roc_curve(labels,likelihoods, pos_label=1)
            print("Density ratio AUC LL: {}".format(auc(fpr, tpr)))
            plt.plot(tpr,fpr,label="Ratio of Densities Max. Likelihood Training",
                     color='black',
                     linestyle=utils.line_style['top_tagging_ll'])
            

            #Classifier
            
            labels_top,likelihoods_top = evaluate_classifier(
                config['NUM_FEAT'],
                '../checkpoints_{}/checkpoint'.format(model_name+ '_class_gluon_tagging'),
                flags.data_folder,
            )
            fpr, tpr, _ = roc_curve(labels_top,likelihoods_top, pos_label=1)
            print("Classifier AUC: {}".format(auc(fpr, tpr)))
            plt.plot(tpr,fpr,label="Supervised Classifier",
                     color='gray',
                     linestyle=utils.line_style['top_tagging_ll'])
        

        
            plt.yscale('log')           
            plt.legend(frameon=False,fontsize=14)
            fig.savefig('{}/{}.pdf'.format(flags.plot_folder,"supervised_ROC"))
            
        

        fpr_dict = {}
        tpr_dict = {}
        anomalies = ['top_tagging','HV','top_tagging_ll','HV_ll']
        for anomaly in anomalies:
            likelihoods = np.concatenate([nll_qcd_anomaly['gluon_tagging'],nll_qcd_anomaly[anomaly]],0)
            labels = np.concatenate([np.zeros(nll_qcd_anomaly['gluon_tagging'].shape[0]),
                                     np.ones(nll_qcd_anomaly[anomaly].shape[0])],0)
    
            fpr, tpr, _ = roc_curve(labels,likelihoods, pos_label=1)
            print("Unsup.Gluon vs {} AUC: {}".format(anomaly, auc(fpr, tpr)))
        
            if flags.sup ==False:
                plt.plot(tpr,fpr,label="QCD vs {}".format(utils.name_translate[anomaly]),
                         color=utils.colors[anomaly],linestyle=utils.line_style[anomaly])
                
            fpr_dict[anomaly] = fpr
            tpr_dict[anomaly] = tpr

        inverse_anomalies = ['gluon_tagging','HV','gluon_tagging_ll','HV_ll']
        for anomaly in inverse_anomalies:
            likelihoods = np.concatenate([nll_top_anomaly['top_tagging'],nll_top_anomaly[anomaly]],0)
            labels = np.concatenate([np.zeros(nll_qcd_anomaly['top_tagging'].shape[0]),
                                     np.ones(nll_qcd_anomaly[anomaly].shape[0])],0)
    
            fpr, tpr, _ = roc_curve(labels,likelihoods, pos_label=1)
            print("Unsup.Top vs {} AUC: {}".format(anomaly, auc(fpr, tpr)))
        
            if flags.sup ==False:
                plt.plot(tpr,fpr,label="Top vs {}".format(utils.name_translate[anomaly]),
                         color=utils.colors_add[anomaly],linestyle=utils.line_style[anomaly])

            # fpr_dict[anomaly] = fpr
            # tpr_dict[anomaly] = tpr
                            
        if flags.sup ==False:
            plt.yscale('log')
            plt.ylim(1e-6, 1)
            plt.legend(frameon=False,fontsize=12,ncols=2)
            fig.savefig('{}/unsupervised_ROC.pdf'.format(flags.plot_folder))


        fig,gs = utils.SetFig("True positive rate","TPR/Sqrt(FPR)")

        anomalies = ['top_tagging','HV','top_tagging_ll','HV_ll']

        for anomaly in anomalies:
            print(anomaly,np.max(np.ma.divide(tpr_dict[anomaly],np.sqrt(fpr_dict[anomaly]))))
            plt.plot(fpr_dict[anomaly],np.ma.divide(tpr_dict[anomaly],np.sqrt(fpr_dict[anomaly])).filled(0),
                     label="QCD vs {}".format(utils.name_translate[anomaly]),
                     color=utils.colors[anomaly],linestyle=utils.line_style[anomaly])
            
        plt.xscale('log')
        plt.legend(frameon=False,fontsize=14)
        fig.savefig('{}/SIC.pdf'.format(flags.plot_folder))
    
