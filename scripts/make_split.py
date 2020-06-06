import glob
import os
import pickle
import numpy as np
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold, StratifiedShuffleSplit
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def make_datapath_list(dir_path, labels_dict):
    target_path = os.path.join(dir_path + '*.npy')
    
    path_list = []
    labels = []
    for path in glob.glob(target_path):
        path_list.append(path)
        for k,v in labels_dict.items():
            if k in path:
                if v < 20:
                    labels.append(0)
                else:
                    labels.append(1)
                break
    path_list = np.array(path_list)
    oh_labels = np.eye(2)[labels]
    
    index = []
    for i in range(5):
        for j in range(6):
            index.extend([i*240+j+6*k for k in range(40)])
    
    N = len(path_list)
    if N != len(oh_labels):
        print('Error!')
        raise ValueError

    return path_list[index], oh_labels[index], np.array(labels)

def LeaveOneSubjectOut(path_list, labels, fold = 5):
    kf = KFold(n_splits=5, shuffle=False, random_state=None)
    test_index = []
    for _, test in kf.split(path_list):
        test_index.append(test)

    train_index, valid_index = [], []
    for i in range(fold):
        test = test_index[i]
        valid = test_index[(i+1)%fold]
        train = np.array([i for i in range(len(path_list)) if i not in test and i not in valid])
        train_index.append(train)
        valid_index.append(valid)
    
    return train_index, valid_index, test_index

def SessionShuffleSplit(path_list, labels, fold = 5):
    skf = KFold(n_splits = fold, shuffle=True, random_state = 1234)
    sessions = [i for i in range(6*5)]
    test_index = []
    for _, test in skf.split(sessions):
        temp_test = []
        for i in test:
            temp_test.extend([i*40+j for j in range(40)])
        test_index.append(temp_test)
        
    train_index, valid_index = [], []
    for i in range(fold):
        test = test_index[i]
        valid = test_index[(i+1)%fold]
        train = np.array([i for i in range(len(path_list)) if i not in test and i not in valid])
        train_index.append(train)
        valid_index.append(valid)
    
    return train_index, valid_index, test_index

def SampleShuffleSplit(path_list, labels, fold = 5):
    skf = StratifiedKFold(n_splits = fold, shuffle=True, random_state = 1234)
    test_index = []
    for _, test in skf.split(path_list, labels):
        test_index.append(test)
        

    train_index, valid_index = [], []
    for i in range(fold):
        test = test_index[i]
        valid = test_index[(i+1)%fold]
        train = np.array([i for i in range(len(path_list)) if i not in test and i not in valid])
        train_index.append(train)
        valid_index.append(valid)
    
    return train_index, valid_index, test_index

def plot_contrast(contrast, split):
    f,ax = plt.subplots(1,1,figsize=(25,5), )
    cmap = LinearSegmentedColormap.from_list('mycmap', [(0, 'seagreen'), (0.5, 'skyblue'), (1, 'sandybrown')])
    im = ax.imshow(np.array(contrast), aspect='auto', cmap=cmap)
    #f.colorbar(im)
    if split == 'SampleShuffleSplit':
        ax.set_title('Sample Shuffle Split', fontsize=20)
    elif split == 'SessionShuffleSplit':
        ax.set_title('Session Shuffle Split', fontsize=20)
    elif split == 'LeaveOneSubjectOut':
        ax.set_title('Leave One Subject Out', fontsize=20) 
    ax.set_xlabel('sample (trials)', fontsize=18)
    ax.set_ylabel('fold', fontsize=18)
    ax.set_xticks([119,239,359,479,599,719,839,959,1079], minor=False)
    ax.set_yticks([0,1,2,3,4], minor=False)
    ax.set_xticklabels(['subject1','','subject2','','subject3','','subject4','','subject5'], fontsize=13)
    ax.set_yticklabels(['1','2','3','4','5'], fontsize=13)
    plt.savefig('./results/{}.png'.format(split))

def make_split(dir_path, labels_dict, split, fold = 5):
    path_list, oh_labels, labels = make_datapath_list(dir_path, labels_dict)
    
    train_list, valid_list, test_list = [], [], []
    train_labels, valid_labels, test_labels = [], [], []
    contrast = np.zeros((fold,len(path_list)))

    if split == 'SampleShuffleSplit':
        train_index, valid_index, test_index = SampleShuffleSplit(path_list, labels)
    elif split == 'SessionShuffleSplit':
        train_index, valid_index, test_index = SessionShuffleSplit(path_list, labels)
    elif split == 'LeaveOneSubjectOut':
        train_index, valid_index, test_index = LeaveOneSubjectOut(path_list, labels)
    else:
        raise ValueError

    for i, (train, valid, test) in enumerate(zip(train_index, valid_index, test_index)):
        train_list.append(path_list[train].tolist())
        valid_list.append(path_list[valid].tolist())
        test_list.append(path_list[test].tolist())
        train_labels.append(oh_labels[train].tolist())
        valid_labels.append(oh_labels[valid].tolist())
        test_labels.append(oh_labels[test].tolist())
        contrast[i,valid], contrast[i,test] = 1, 2
        
    plot_contrast(contrast,split)
    return [train_list, valid_list, test_list], [train_labels, valid_labels, test_labels]
