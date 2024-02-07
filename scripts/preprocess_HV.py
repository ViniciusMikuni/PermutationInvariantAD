import os
import pandas as pd
import numpy as np
from optparse import OptionParser
import h5py
import re



def featurizer(jets,particles,npart=200):
    
    new_jets = np.zeros((jets.shape[0],4))
    new_jets[:,0] = jets[:,0] #pt
    new_jets[:,1] = jets[:,1] #eta
    new_jets[:,2] = jets[:,3] #mass
    new_jets[:,3] = np.sum(particles[:,:,0]>0,1)
                              
    mask = particles[:,:,0] > 0

    delta_phi = particles[:,:,2] - np.expand_dims(jets[:,2],-1)
    delta_phi[delta_phi>np.pi] -=  2*np.pi
    delta_phi[delta_phi<= - np.pi] +=  2*np.pi

    mt2 = jets[:,0]**2 + jets[:,3]**2
    jet_energy = np.expand_dims(np.sqrt(mt2*np.cosh(jets[:,1])**2),-1)
    new_particles = np.zeros((jets.shape[0],npart,4))
    new_particles[:,:,0] = (particles[:,:,1] - np.expand_dims(jets[:,1],-1))*mask
    new_particles[:,:,1] = delta_phi*mask
    new_particles[:,:,2] = np.ma.log(1. - particles[:,:,0]/np.expand_dims(jets[:,0],-1)).filled(0)
    new_particles[:,:,3] = mask.astype(np.float32)
    
    # new_particles[:,:,2] = np.ma.log(particles[:,:,0])
    # new_particles[:,:,3] = np.ma.log(particles[:,:,0]*np.cosh(particles[:,:,1])).filled(0)
    # new_particles[:,:,4] = np.ma.log(particles[:,:,0]/np.expand_dims(jets[:,0],-1)).filled(0)
    # new_particles[:,:,5] = np.ma.log(particles[:,:,0]*np.cosh(particles[:,:,1])/jet_energy).filled(0)
    # new_particles[:,:,6] = np.sqrt((particles[:,:,1] - np.expand_dims(jets[:,1],-1))**2 + delta_phi**2)*mask
    # new_particles[:,:,7] = mask.astype(np.float32)
    
    return new_jets, new_particles


def parse_file(file_path,npart=200):
    jet_pattern = re.compile(r'(\d+) (\d+) J (\S+) (\S+) (\S+) (\S+) (\S+) (\S+) (\S+) (\S+) (\S+) (\S+)')
    particle_pattern = re.compile(r'P (\S+) (\S+) (\S+)')

    
    j = []
    p = []


    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            jets = []
            particles = []
        
            jet_match = jet_pattern.match(line)
            particle_match = particle_pattern.findall(line)

            if jet_match:
                jet_info = [float(jet_match.group(i)) for i in range(3, 13)]
                jets.append(jet_info)
            if particle_match:
                for match in particle_match:
                    particle_info = [float(value) for value in match]
                    particles.append(particle_info)
                
            jets = np.array(jets)
            particles = np.array(particles)
            particles = particles[:npart]

            # Zero-pad particles if fewer than npart
            if len(particles) < npart:
                num_padding = npart - len(particles)
                padding = np.zeros((num_padding, 3))
                particles = np.concatenate((particles, padding), axis=0)

            j.append(jets)
            p.append([particles])

    return np.concatenate(j,0), np.concatenate(p,0)


if __name__=='__main__':
    parser = OptionParser(usage="%prog [opt]  inputFiles")
    parser.add_option("--npoints", type=int, default=100, help="Number of particles per event")
    parser.add_option("--folder", type="string", default='/global/cfs/cdirs/m3929/TOP/', help="Folder containing input files")
    parser.add_option("--sample", type="string", default='events10k.txt', help="Input file name")

    (flags, args) = parser.parse_args()
    file_path = os.path.join(flags.folder,flags.sample)

    jets, particles = parse_file(file_path,flags.npoints)
    print('Jets:')
    print(jets.shape)
    print('Particles:')
    print(particles.shape)

    jets, particles = featurizer(jets,particles,flags.npoints)
    print(np.min(jets,0))
    print(np.min(particles.reshape(-1,particles.shape[-1]),0))
    print(np.max(jets,0))
    print(np.max(particles.reshape(-1,particles.shape[-1]),0))


    with h5py.File('/global/cfs/cdirs/m3929/TOP/HV_orig.h5', "w") as fh5:
        dset = fh5.create_dataset('particle_features', data=particles)
        dset = fh5.create_dataset('jet_features', data=jets)
