import pandas as pd
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import ticker

sns.set_style("whitegrid")

def load_DataFrame(path_list):
    df = pd.DataFrame()
    for path in path_list:
        df = df.append(pd.read_csv(path))
    return df.reset_index(drop=False)

def main():
    results_df = load_DataFrame(glob('./results/*.csv'))

    order = ['LeaveOneSubjectOut','SessionShuffleSplit','SampleShuffleSplit']
    hue_order = ['Permutation','PLR','SVM','M2DCNN','3DCNN']

    # Box Plot
    f,ax = plt.subplots(1,1,figsize=(10,5), dpi=100)
    sns.boxplot(x='split_method',y='accuracy',hue='train_method',order=order,
                hue_order=hue_order, showfliers=False, palette='Set1', data=results_df,ax=ax)
    sns.stripplot(x='split_method',y='accuracy',hue='train_method',order=order,
                hue_order=hue_order, jitter=True, dodge=True, color='black',size=2, data=results_df, ax=ax)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[0:len(hue_order)], labels[0:len(hue_order)], frameon = False)
    ax.set_xlabel('Split Method')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0,1)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1)) 
    plt.savefig('./results/accuracy_boxplot.png')
    plt.show()

    # Violin Plot
    f,ax = plt.subplots(1,1,figsize=(10,5), dpi=100)
    sns.violinplot(x='split_method',y='accuracy',hue='train_method',order=order,
                hue_order=hue_order, showfliers=False, palette='Set1',inner=None, data=results_df,ax=ax)
    sns.stripplot(x='split_method',y='accuracy',hue='train_method',order=order,
                hue_order=hue_order, jitter=True, dodge=True, color='black',size=2, data=results_df, ax=ax)
    handles, labels = ax.get_legend_handles_labels()
    #plt.title('cosine similarity')
    ax.legend(handles[0:len(hue_order)], labels[0:len(hue_order)], frameon = False)
    ax.set_xlabel('Split Method')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0,1)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1)) 
    plt.savefig('./results/accuracy_violinplot.png')
    plt.show()


    # Bar Plot
    f,ax = plt.subplots(1,1,figsize=(10,5), dpi=100)
    sns.barplot(x='split_method',y='accuracy',hue='train_method',order=order,
                hue_order=hue_order, palette='Set1', errwidth=0, data=results_df, ax=ax)
    sns.stripplot(x='split_method',y='accuracy',hue='train_method',order=order,
                hue_order=hue_order, jitter=True, dodge=True, color='black',size=2, data=results_df, ax=ax)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[len(hue_order):len(hue_order)*2], labels[len(hue_order):len(hue_order)*2], frameon = False)
    ax.set_xlabel('Split Method')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0,1)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1)) 
    plt.savefig('./results/accuracy_barplot.png')
    plt.show()

    f,ax = plt.subplots(1,1,figsize=(10,5), dpi=100)
    sns.barplot(x='split_method',y='accuracy',hue='train_method',order=order,
                hue_order=hue_order, palette='Set1', errwidth=0, data=results_df, ax=ax)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[0:len(hue_order)], labels[0:len(hue_order)], frameon = False)
    ax.set_xlabel('Split Method', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_ylim(0,1)
    ax.set_xticklabels(['Leave One Subject Out','Session Shuffle Split','Sample Shuffle Split'], fontsize=12)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1)) 
    plt.savefig('./results/accuracy_barplot_notstrip.png')
    plt.show()

    


if __name__ == "__main__":
    main()