from make_split import make_split
from train_sklearn import train_plr

import numpy as np
import pandas as pd
import pickle
import argparse
from joblib import Parallel, delayed

from configparser import ConfigParser

def label_check(dataset_path, label_list):
    for i,j in zip(dataset_path[0][0][:5],label_list[0][0][:5]):
        print(i,j)

def permute_labal(dataset_path, label_list, N):
    np.random.seed(1234)
    permute_data_path = []
    permute_label_list = []
    for _ in range(N):
        cv = np.random.randint(5)
        permute_data_path.append(
            [ dataset_path[0][cv], dataset_path[1][cv], dataset_path[2][cv]]
        )
        permute_label_list.append(
            [
                np.random.permutation(label_list[0][cv]),
                np.random.permutation(label_list[1][cv]),
                np.random.permutation(label_list[2][cv])
            ]
        )
    
    return permute_data_path, permute_label_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('split_method', help='splitting method (SampleShuffleSplit, SessionShuffleSplit, LeaveOneSubjectOut)', type=str)
    parser.add_argument('permute_number', help='the number of repetition for permutation test', type=int)
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

    permute_data_path, permute_label_list = permute_labal(dataset_path, label_list, args.permute_number)
    label_check(permute_data_path, permute_label_list)

    df_result = pd.DataFrame(columns=['split_method','train_method','fold','accuracy'])

    def process(dataset_path, label_list, cv, split_method, columns):
        temp_accuracy = train_plr(dataset_path, label_list)
        return pd.Series( [split_method, 'Permutation', cv, temp_accuracy], index=columns, name=cv)

    se_list = Parallel(n_jobs=-1, verbose=10)( [delayed(process)(permute_data_path[cv], permute_label_list[cv], cv, args.split_method, df_result.columns) for cv in range(args.permute_number)] )
    for tmp_se in se_list:
        df_result = df_result.append(tmp_se)
    
    print(df_result.head(5))
    df_result.to_csv('./results/result_permutation_{}.csv'.format(args.split_method), index=False)

if __name__ == "__main__":
    main()
