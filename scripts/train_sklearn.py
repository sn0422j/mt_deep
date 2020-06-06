from dataset import mt_Dataset

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest, f_classif


def Flatten(array):
    new_array = []
    for mat in array:
        new_array.append(mat.flatten())
    return np.array(new_array)


def prepare_dataset(dataset_path, label_list):
    train_dataset = mt_Dataset(dataset_path[0], label_list[0])
    valid_dataset = mt_Dataset(dataset_path[1], label_list[1])
    test_dataset = mt_Dataset(dataset_path[2], label_list[2])

    train_dataloader = DataLoader(train_dataset, batch_size = len(dataset_path[0]), shuffle = False)
    valid_dataloader = DataLoader(valid_dataset, batch_size = len(dataset_path[1]), shuffle = False)
    test_dataloader = DataLoader(test_dataset, batch_size = len(dataset_path[2]), shuffle = False)
    
    train = [i.numpy() for i in next(iter(train_dataloader))]
    valid = [i.numpy() for i in next(iter(valid_dataloader))]
    test = [i.numpy() for i in next(iter(test_dataloader))]
    train_data = Flatten(train[0])
    valid_data = Flatten(valid[0])
    test_data = Flatten(test[0])
    train_label = train[1]
    valid_label = valid[1]
    test_label = test[1]

    return train_data, valid_data, test_data, train_label, valid_label, test_label


def train_plr(dataset_path, label_list, k=500):
    train_data, valid_data, test_data, train_label, valid_label, test_label = prepare_dataset(dataset_path, label_list)

    selector = SelectKBest(score_func=f_classif, k=k)
    train_data = selector.fit_transform(train_data, train_label)
    valid_data = selector.transform(valid_data)
    test_data = selector.transform(test_data)
    
    best_score, best_c = 0, 0
    max_iter = 10000
    for c in np.logspace(-10,10):
        model = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=c, fit_intercept=True,
                intercept_scaling=1, class_weight=None, random_state=1234,
                solver='lbfgs', max_iter=max_iter, multi_class='auto', verbose=0,
                warm_start=False, n_jobs=-1, l1_ratio=None).fit(train_data, train_label)
        if model.score(valid_data, valid_label) > best_score:
            best_c = c
            best_score = model.score(valid_data, valid_label)

    best_model = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=best_c, fit_intercept=True,
                intercept_scaling=1, class_weight=None, random_state=1234,
                solver='lbfgs', max_iter=max_iter, multi_class='auto', verbose=0,
                warm_start=False, n_jobs=-1, l1_ratio=None).fit(train_data, train_label)

    return best_model.score(test_data, test_label)

def train_svm(dataset_path, label_list, k=500):
    train_data, valid_data, test_data, train_label, valid_label, test_label = prepare_dataset(dataset_path, label_list)

    selector = SelectKBest(score_func=f_classif, k=k)
    train_data = selector.fit_transform(train_data, train_label)
    valid_data = selector.transform(valid_data)
    test_data = selector.transform(test_data)
    
    best_score, best_c = 0, 0
    max_iter = 10000
    for c in np.logspace(-10,10):
        model = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001,
                        C=c, multi_class='ovr', fit_intercept=True, intercept_scaling=1,
                        class_weight=None, verbose=0, random_state=1234, max_iter=max_iter).fit(train_data, train_label)
        if model.score(valid_data, valid_label) > best_score:
            best_c = c
            best_score = model.score(valid_data, valid_label)

    best_model = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001,
                        C=best_c, multi_class='ovr', fit_intercept=True, intercept_scaling=1,
                        class_weight=None, verbose=0, random_state=1234, max_iter=max_iter).fit(train_data, train_label)

    return best_model.score(test_data, test_label)