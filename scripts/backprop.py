from model_m2dcnn import M2DCNN
from dataset import mt_Dataset
from make_split import make_split

import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
from configparser import ConfigParser

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from captum.attr import IntegratedGradients

from nilearn.datasets import load_mni152_template, fetch_atlas_destrieux_2009, fetch_atlas_aal
from nilearn.image import new_img_like, smooth_img, load_img, coord_transform
from nilearn.plotting import plot_glass_brain, plot_stat_map, plot_epi, plot_roi

def seed_everything(seed=1234):
    #random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def find_atlas(volume):
    #atlas = fetch_atlas_destrieux_2009(lateralized=True, data_dir=None, url=None, resume=True, verbose=1)
    atlas = fetch_atlas_aal(version='SPM12', data_dir=None, url=None, resume=True, verbose=1)
    dictionary = dict(zip([int(i) for i in atlas.indices],atlas.labels))
    A = load_img(atlas.maps).affine
    trA = np.linalg.inv(A)
    atlas_map = load_img(atlas.maps).get_fdata()
    affine = [[-3,0,0,78],[0,3,0,-112],[0,0,6,-50],[0,0,0,1]]
    
    def cast(x,y,z,affine,trA):
        coord = coord_transform(x,y,z,affine)
        return np.dot(trA, [coord[0],coord[1],coord[2],1])

    def find(x,y,z,affine,trA):
        vox_coord = cast(x,y,z,affine,trA)
        label = atlas_map[int(vox_coord[0]),int(vox_coord[1]),int(vox_coord[2])]
        if int(label) == 0: return
        return dictionary[int(label)]

    index = np.where(volume > 0.8)
    for i in range(np.sum(volume > 0.8)):
        x,y,z = index[0][i],index[1][i],index[2][i]
        print(x,y,z,coord_transform(x,y,z,affine),find(x,y,z,affine,trA),volume[x,y,z])
        
    index = np.where(volume < -0.8)
    for i in range(np.sum(volume < -0.8)):
        x,y,z = index[0][i],index[1][i],index[2][i]
        print(x,y,z,coord_transform(x,y,z,affine),find(x,y,z,affine,trA),volume[x,y,z]) 
        
def main():
    seed_everything()

    split_method = 'SessionShuffleSplit'
    cv = 2
    weights_path = './results/M2DCNN_{}_cv{}_weights.pth'.format(split_method,cv)

    model = M2DCNN(numClass=2, numFeatues=6528, DIMX=53, DIMY=63, DIMZ=17)
    print(model.load_state_dict(torch.load(weights_path)))
    model.eval()
    print(model)

    config_ini = ConfigParser()
    config_ini.read('config.ini', encoding='utf-8')
    dir_path = config_ini.get('PATH','dir_path')
    with open('./data/labels_dict.pickle', 'rb') as f: labels_dict = pickle.load(f)
    dataset_path, label_list = make_split(dir_path, labels_dict, split_method)

    test_dataset_path, test_label_list = dataset_path[2][cv][:120], label_list[2][cv][:120]

    mammal_index = np.where(np.array(test_label_list)[:,0]==1)
    tool_index = np.where(np.array(test_label_list)[:,1]==1)

    baseline_dataset = mt_Dataset(test_dataset_path, test_label_list)
    mammal_dataset = mt_Dataset(np.array(test_dataset_path)[mammal_index].tolist(),
                                np.array(test_label_list)[mammal_index].tolist())
    tool_dataset = mt_Dataset(np.array(test_dataset_path)[tool_index].tolist(),
                                np.array(test_label_list)[tool_index].tolist())

    baseline_input, _ = next(iter(DataLoader(baseline_dataset, batch_size = len(baseline_dataset))))
    baseline_input = baseline_input.mean(dim=0,keepdim=True).float()

    mammal_input, mammal_label = next(iter(DataLoader(mammal_dataset, batch_size = len(mammal_dataset))))
    ig = IntegratedGradients(model)
    mammal_attributions, delta = ig.attribute(mammal_input.float(),
                        baselines=baseline_input ,target=mammal_label, return_convergence_delta=True)
    print('delta:',delta)

    tool_input, tool_label = next(iter(DataLoader(tool_dataset, batch_size = len(tool_dataset))))
    ig = IntegratedGradients(model)
    tool_attributions, delta = ig.attribute(tool_input.float(),
                        baselines=baseline_input ,target=tool_label, return_convergence_delta=True)
    print('delta:',delta)

    ## Cohen's d
    M1 = mammal_attributions.numpy().mean(axis=0)
    M2 = tool_attributions.numpy().mean(axis=0)
    V1 = mammal_attributions.numpy().var(axis=0)
    V2 =tool_attributions.numpy().var(axis=0)

    d = (M1-M2)/ np.sqrt((V1+V2)/2)

    base = baseline_input[0].numpy()
    contrast = d.copy()
    contrast[np.where(base == base[0][0][0])] = 0
    contrast = contrast.transpose(2,1,0)

    volume = np.concatenate(
        [np.zeros((contrast.shape[0],contrast.shape[1],2)),
        contrast,
        np.zeros((contrast.shape[0],contrast.shape[1],4))], axis=2)

    baseline_volume = np.concatenate(
        [np.zeros((contrast.shape[0],contrast.shape[1],2)),
        baseline_input[0].numpy().transpose(2,1,0),
        np.zeros((contrast.shape[0],contrast.shape[1],4))], axis=2)

    affine = [[-3,0,0,78],[0,3,0,-112],[0,0,6,-50],[0,0,0,1]]
    image = new_img_like(ref_niimg=load_mni152_template(), data=volume, affine=affine, copy_header=False)
    baseline_image = new_img_like(ref_niimg=load_mni152_template(), data=baseline_volume, affine=affine, copy_header=False)

    plot_stat_map(stat_map_img=smooth_img(image,8),  cut_coords=20,
            output_file=None, display_mode='z', colorbar=True, figure=None, axes=None,
            title='Mammal > Tool', threshold=0.2, annotate=True, draw_cross=False, black_bg=True, cmap='coolwarm',
            symmetric_cbar='auto', dim='auto')

    plot_glass_brain(stat_map_img=image,
            output_file=None, display_mode='ortho', colorbar=True, 
            figure=None, axes=None, title='Mammal > Tool', threshold='auto', annotate=True,
            black_bg=False, cmap='bwr', alpha=0.7, vmin=None, vmax=None, plot_abs=False)

    find_atlas(smooth_img(image,6).get_fdata())

if __name__ == "__main__":
    main()