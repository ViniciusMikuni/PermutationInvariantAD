import numpy as np
import argparse
import h5py as h5
import os
import utils



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    #parser.add_argument('--data_folder', default='/global/cfs/cdirs/m3929/TOP', help='Folder containing data and MC files')
    parser.add_argument('--data_folder', default='/pscratch/sd/v/vmikuni/TOP/', help='Folder containing data and MC files')
    parser.add_argument('--config', default='config_AD.json', help='Training parameters')
    parser.add_argument('--maxidx', default=200, type=int,help='Parallel sampling of the data')
    parser.add_argument('--ll', action='store_true', default=False,help='Load Max LL training model')

    flags = parser.parse_args()
    config = utils.LoadJson(flags.config)
    model_name = config['MODEL_NAME']
    if flags.ll:
        model_name+='_ll'

    processes = ['gluon_tagging','top_tagging','HV']
    
    combined = {}
        
    for idx in range(flags.maxidx):
        #if idx in [22,41]:continue
        with h5.File(os.path.join(flags.data_folder,model_name + '_gluon_tagging_{}.h5'.format(idx)),"r") as h5f:
            for key in h5f:
                if key in combined:                                    
                    combined[key] = np.concatenate([combined[key],h5f[key][:]])
                else:
                    combined[key] = h5f[key][:]

    with h5.File(os.path.join(flags.data_folder,model_name+ '_gluon_tagging.h5'),"w") as h5f:
            for key in combined:
                dset = h5f.create_dataset(key, data=combined[key])

    combined = {}

    for idx in range(flags.maxidx):
        with h5.File(os.path.join(flags.data_folder,model_name + '_top_tagging_{}.h5'.format(idx)),"r") as h5f:
            for key in h5f:
                if key in combined:                                    
                    combined[key] = np.concatenate([combined[key],h5f[key][:]])
                else:
                    combined[key] = h5f[key][:]

    with h5.File(os.path.join(flags.data_folder,model_name+ '_top_tagging.h5'),"w") as h5f:
            for key in combined:
                dset = h5f.create_dataset(key, data=combined[key])

