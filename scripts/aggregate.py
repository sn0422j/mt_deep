import pandas as pd
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats

def load_DataFrame(path_list):
    df = pd.DataFrame()
    for path in path_list:
        df = df.append(pd.read_csv(path))
    return df.reset_index(drop=False)

def main():
    results_df = load_DataFrame(glob('./results/*.csv'))
    print(results_df.groupby(['split_method', 'train_method']).agg({"accuracy": "mean"}))

    for split_method in ['LeaveOneSubjectOut','SessionShuffleSplit','SampleShuffleSplit']:
        split_df = results_df.loc[results_df['split_method'] == split_method]
        for train_method in ['PLR','SVM','M2DCNN','3DCNN']:
            test_df = split_df.loc[split_df['train_method']==train_method].append(split_df.loc[split_df['train_method']=='Permutation']).reset_index()
            statistic, p = stats.ranksums(test_df.loc[test_df['train_method']==train_method,'accuracy'],
                                                test_df.loc[test_df['train_method']=='Permutation','accuracy'])
            print(split_method,train_method,'pvalue:',round(p,4))
            
            

if __name__ == "__main__":
    main()