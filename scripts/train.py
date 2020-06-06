from make_split import make_split
from train_sklearn import train_plr, train_svm
from train_pytorch import train_m2dcnn, train_3dcnn

import numpy as np
import pandas as pd
import pickle
import argparse
from tqdm.notebook import tqdm
from configparser import ConfigParser


def label_check(dataset_path, label_list):
    for i,j in zip(dataset_path[0][0][:5],label_list[0][0][:5]):
        print(i,j)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('split_method', help='splitting method (SampleShuffleSplit, SessionShuffleSplit, LeaveOneSubjectOut)', type=str)
    parser.add_argument('train_method', help='training method (PLR, SVM, )', type=str)
    args = parser.parse_args()
    print(args)

    config_ini = ConfigParser()
    config_ini.read('config.ini', encoding='utf-8')
    dir_path = config_ini.get('PATH','dir_path')
    with open('./data/labels_dict.pickle', 'rb') as f: labels_dict = pickle.load(f)
    print(labels_dict,'\n')

    dataset_path, label_list = make_split(dir_path, labels_dict, args.split_method)
    for i in range(5):
        print('train:',len(dataset_path[0][i]),'valid:',len(dataset_path[1][i]),'test:',len(dataset_path[2][i]))
    label_check(dataset_path, label_list)

    df_result = pd.DataFrame(columns=['split_method','train_method','fold','accuracy'])
    if args.train_method == 'PLR':
        for cv in tqdm(range(5)):
            temp_accuracy = train_plr([path[cv] for path in dataset_path],
                                        [label[cv] for label in label_list])
            tmp_se = pd.Series( [args.split_method, args.train_method, cv, temp_accuracy], index=df_result.columns, name=cv)
            df_result = df_result.append(tmp_se)
    elif args.train_method == 'SVM':
        for cv in tqdm(range(5)):
            temp_accuracy = train_svm([path[cv] for path in dataset_path],
                                        [label[cv] for label in label_list])
            tmp_se = pd.Series( [args.split_method, args.train_method, cv, temp_accuracy], index=df_result.columns, name=cv)
            df_result = df_result.append(tmp_se)
    elif args.train_method == 'M2DCNN':
        for cv in tqdm(range(5)):
            temp_accuracy = train_m2dcnn([path[cv] for path in dataset_path],
                                        [label[cv] for label in label_list],
                                        condition='M2DCNN_{}_cv{}'.format(args.split_method,cv))
            tmp_se = pd.Series( [args.split_method, args.train_method, cv, temp_accuracy], index=df_result.columns, name=cv)
            df_result = df_result.append(tmp_se)
    elif args.train_method == '3DCNN':
        for cv in tqdm(range(5)):
            temp_accuracy = train_3dcnn([path[cv] for path in dataset_path],
                                        [label[cv] for label in label_list],
                                        condition='3DCNN_{}_cv{}'.format(args.split_method,cv))
            tmp_se = pd.Series( [args.split_method, args.train_method, cv, temp_accuracy], index=df_result.columns, name=cv)
            df_result = df_result.append(tmp_se)
    else:
        raise ValueError

    print(df_result.head(5))
    df_result.to_csv('./results/result_{}_{}.csv'.format(args.split_method, args.train_method), index=False)
    

if __name__ == "__main__":
    main()