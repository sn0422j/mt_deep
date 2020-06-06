import numpy as np
from nibabel import load, save
from nipy.labs.mask import compute_mask_files
from tqdm.notebook import tqdm_notebook as tqdm
import pandas as pd
from configparser import ConfigParser

def load_labels(fname):
    txt = np.loadtxt(fname,dtype='str')
    labels_dict_ = dict((word, i) for i,word in enumerate([i for i in sorted(list(set(txt[:,0]))) if i != 'base' and i != 'rest']))
    labels_dict_['base'] = len(set(txt[:,0])) - 2
    labels_dict_['rest'] = len(set(txt[:,0])) - 1
    labels_ = []
    res_ = []
    cnt = 0
    for i, chunks in enumerate(txt):
        if i < 4:
            res_.append([len(txt)-4+i,6,41])
            continue
        
        if labels_dict_[chunks[0]] < 40:
            cnt += 1                
            if cnt > 1:
                labels_.append([i-4,int(chunks[1]),41])
            else:
                labels_.append([i-4,int(chunks[1]),labels_dict_[chunks[0]]])
        else:
            cnt = 0
            labels_.append([i-4,int(chunks[1]),labels_dict_[chunks[0]]])
    for res in res_:
        labels_.append(res)
    return labels_, labels_dict_
    
def get_keys_from_value(d, val):
    return [k for k, v in d.items() if v == val]

def prep_fmri(vox, start, duration = 12):
    return vox[:,:,:,start:start+duration]

def dataset(labels_, labels_dict_, fmri_fname_, sub_name_):
    img = load(fmri_fname_)
    mask = compute_mask_files(fmri_fname_)
    vox = img.get_data() * mask[:,:,:, np.newaxis]
    print(img.affine)
    for lab in tqdm(labels_):
        if lab[2] < 40:
            vox_ = prep_fmri(vox, lab[0])
            print(vox_.shape)
            save_fname = sub_name_ + '_' + get_keys_from_value(labels_dict_, lab[2])[0] + '_run' + str(lab[1]) + '.npy'
            np.save('./data_masked/'+save_fname, vox_)
            print(save_fname)

def main():
    config_ini = ConfigParser()
    config_ini.read('config.ini', encoding='utf-8')
    dir_path = config_ini.get('PREPARE','dir_path')
    subjects = config_ini.get('PREPARE','subjects').split()
    f_name = config_ini.get('PREPARE','f_name')
    l_name = config_ini.get('PREPARE','l_name')

    for sub_name in subjects:     
        fmri_fname = dir_path + sub_name + f_name #4D.nii
        labels_fname = dir_path + sub_name + l_name #chuncksTargegts.txt
        labels, labels_dict = load_labels(labels_fname)
        print(labels_dict)
        dataset(labels, labels_dict, fmri_fname, sub_name)

    with open('./data/labels_dict.pickle', 'wb') as f: pickle.dump(labels_dict, f)
