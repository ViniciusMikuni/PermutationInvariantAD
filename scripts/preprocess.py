import pandas as pd
import h5py
import os
import numpy as np
from optparse import OptionParser


#Preprocessing for the top tagging dataset

def convert_coordinate(data):
    px = data[:,:,1]
    py = data[:,:,2]
    pz = data[:,:,3]
    energy  = data[:,:,0]


    pt = np.sqrt(px**2 + py**2)
    phi = np.ma.arctan2(py,px).filled(0)
    eta = np.ma.arcsinh(np.ma.divide(pz,pt).filled(0))

    return pt,eta,phi,energy

def convert_jet(px,py,pz,e):

    pt = np.sqrt(px**2 + py**2)
    phi = np.arctan2(py,px)
    eta = np.arcsinh(np.divide(pz,pt))
    m = np.sqrt(np.abs(e**2 - pt**2 - pz**2))
    return pt,eta,phi,m



def clustering_sum(data,folder,nevents=1000,nparts=100):

    npid = data[:nevents,-1]
    
    particles = data[...,0:200*4]
    particles=particles.reshape((data.shape[0],-1,4))
    particles=particles[:nevents,:nparts,:]

    jets = np.sum(particles,axis=1)
    
    jets_pt,jets_eta,jets_phi, jets_mass = convert_jet(jets[:,1],jets[:,2],jets[:,3], jets[:,0])
    
    NFEAT=4
    points = np.zeros((particles.shape[0],particles.shape[1],NFEAT))
    pt,eta,phi,energy = convert_coordinate(particles)


    delta_phi = phi - np.expand_dims(jets_phi,-1)
    delta_phi[delta_phi>np.pi] -=  2*np.pi
    delta_phi[delta_phi<= - np.pi] +=  2*np.pi


    points[:,:,0] = (eta - np.expand_dims(jets_eta,-1))*(pt!=0)
    points[:,:,1] = delta_phi*(pt!=0)
    points[:,:,2] = np.ma.log(1.0 - pt/jets_pt[:,None]).filled(0)
    #points[:,:,2] = np.ma.log(pt).filled(0)
    points[:,:,3] = (pt>0.0).astype(np.float32)
    
    
    jet_info = np.zeros((particles.shape[0],4))
    jet_info[:,0] += jets_pt
    jet_info[:,1] += jets_eta
    jet_info[:,2] += jets_mass
    jet_info[:,3] += np.sum(pt>0.0 , 1)

    
    with h5py.File('{}/top_tagging3.h5'.format(folder), "w") as fh5:
        dset = fh5.create_dataset('particle_features', data=points[npid==1])
        dset = fh5.create_dataset('jet_features', data=jet_info[npid==1])

    with h5py.File('{}/gluon_tagging3.h5'.format(folder), "w") as fh5:
        dset = fh5.create_dataset('particle_features', data=points[npid==0])
        dset = fh5.create_dataset('jet_features', data=jet_info[npid==0])




if __name__=='__main__':
    parser = OptionParser(usage="%prog [opt]  inputFiles")
    parser.add_option("--npoints", type=int, default=100, help="Number of particles per event")
    parser.add_option("--folder", type="string", default='/global/cfs/cdirs/m3929/TOP/', help="Folder containing input files")
    parser.add_option("--sample", type="string", default='val.h5', help="Input file name")

    (flags, args) = parser.parse_args()
        

    samples_path = flags.folder
    sample = flags.sample
    NPARTS = flags.npoints

    store = pd.HDFStore(os.path.join(samples_path,sample))
    data = store['table'].values
    # print(data.shape[0])
    # input()
    clustering_sum(data,samples_path,data.shape[0],NPARTS)
