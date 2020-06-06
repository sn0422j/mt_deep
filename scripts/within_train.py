from train_sklearn import train_plr, train_svm

import numpy as np
import pandas as pd
import pickle
import argparse
from tqdm.notebook import tqdm

import glob
import os
from sklearn.model_selection import KFold

from configparser import ConfigParser

import argparse
import warnings
warnings.simplefilter('ignore')

def label_check(dataset_path, label_list):
    for i,j in zip(dataset_path[0][0][:5],label_list[0][0][:5]):
        print(i,j)

def make_datapath_list(dir_path, labels_dict, subject):
    target_path = os.path.join(dir_path + subject + '*.npy')
    
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
    for j in range(6):
        index.extend([j+6*k for k in range(40)])
    
    N = len(path_list)
    if N != len(oh_labels):
        print('Error!')
        raise ValueError

    return path_list[index], oh_labels[index], np.array(labels)

def LeaveOneOut(path_list, labels, fold = 6):
    kf = KFold(n_splits=fold, shuffle=False, random_state=None)
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', help='subject', type=str, default=0)
    args = parser.parse_args()
    print(args)
    
    config_ini = ConfigParser()
    config_ini.read('config.ini', encoding='utf-8')
    dir_path = config_ini.get('PATH','dir_path')
    with open('./data/labels_dict.pickle', 'rb') as f: labels_dict = pickle.load(f)

    path_list, oh_labels, labels = make_datapath_list(dir_path, labels_dict, subject=args.subject)
    
    train_list, valid_list, test_list = [], [], []
    train_labels, valid_labels, test_labels = [], [], []

    train_index, valid_index, test_index = LeaveOneOut(path_list, labels, fold = 6)

    for i, (train, valid, test) in enumerate(zip(train_index, valid_index, test_index)):
        train_list.append(path_list[train].tolist())
        valid_list.append(path_list[valid].tolist())
        test_list.append(path_list[test].tolist())
        train_labels.append(oh_labels[train].tolist())
        valid_labels.append(oh_labels[valid].tolist())
        test_labels.append(oh_labels[test].tolist())

    dataset_path, label_list = [train_list, valid_list, test_list], [train_labels, valid_labels, test_labels]
    for i in range(6):
        print('train:',len(dataset_path[0][i]),'valid:',len(dataset_path[1][i]),'test:',len(dataset_path[2][i]))
    label_check(dataset_path, label_list)

    df_result = pd.DataFrame(columns=['fold','accuracy'])
    for cv in tqdm(range(6)):
        temp_accuracy = train_svm([path[cv] for path in dataset_path],
                                    [label[cv] for label in label_list])
        tmp_se = pd.Series( [cv, temp_accuracy], index=df_result.columns, name=cv)
        df_result = df_result.append(tmp_se)

    print(df_result.head(6))
    print(df_result.describe())
    
    df_result = pd.DataFrame(columns=['fold','accuracy'])
    for cv in tqdm(range(6)):
        temp_accuracy = train_plr([path[cv] for path in dataset_path],
                                    [label[cv] for label in label_list])
        tmp_se = pd.Series( [cv, temp_accuracy], index=df_result.columns, name=cv)
        df_result = df_result.append(tmp_se)

    print(df_result.head(6))
    print(df_result.describe())
    #df_result.to_csv('./results/result_within_subjects.csv', index=False)

    
if __name__ == "__main__":
    main()