import pickle
from configparser import ConfigParser

import numpy as np
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import ImageGrid

from make_split import make_split
from dataset import mt_Dataset
from train_sklearn import Flatten

def main():
    config_ini = ConfigParser()
    config_ini.read('config.ini', encoding='utf-8')
    dir_path = config_ini.get('PATH','dir_path')
    with open('./data/labels_dict.pickle', 'rb') as f: labels_dict = pickle.load(f)

    dataset_path, label_list = make_split(dir_path, labels_dict, "LeaveOneSubjectOut")
    dataset_path = dataset_path[2][0] + dataset_path[1][0] + dataset_path[0][0]
    label_list = label_list[2][0] + label_list[1][0] + label_list[0][0]

    dataset = mt_Dataset(dataset_path, label_list)
    dataloader = DataLoader(dataset, batch_size = len(dataset_path), shuffle = False)
    data_list = [i.numpy() for i in next(iter(dataloader))]
    data = Flatten(data_list[0])

    data = PCA(random_state=0).fit_transform(data)

    clusters = KMeans(n_clusters=5, random_state=0).fit_predict(data)
    
    subjects = []
    for i in range(5): subjects +=[i]*240
    ARI = adjusted_rand_score(subjects, clusters)

    embedding = TSNE(n_components=2, random_state=0).fit_transform(data)

    cmap = plt.get_cmap("Set1", 5)
    fig = plt.figure(figsize=(10,4))
    grid = ImageGrid(fig, 111, nrows_ncols=(1, 2), axes_pad=1, label_mode='all',
                cbar_location='right', cbar_mode='each', cbar_pad=0.2)
    im = grid[0].scatter(embedding[:, 0], embedding[:, 1], c=subjects, cmap=cmap, s=5)
    grid[0].set_title('true')
    cbar = grid.cbar_axes[0].colorbar(im, ticks=[0.34, 1.2, 2, 2.8, 3.66])
    cbar.ax.set_yticklabels([1, 2, 3, 4, 5])
    im = grid[1].scatter(embedding[:, 0], embedding[:, 1], c=clusters, cmap=cmap, s=5)
    grid[1].set_title(f'predict (ARI:{ARI:.3f})')
    cbar = grid.cbar_axes[1].colorbar(im, ticks=[0.34, 1.25, 2, 2.75, 3.66])
    cbar.ax.set_yticklabels([1, 2, 3, 4, 5])
    plt.savefig("./results/tsne.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    main()