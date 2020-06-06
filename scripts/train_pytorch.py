import os
import sys
import datetime
from tqdm.notebook import tqdm
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from model_m2dcnn import M2DCNN
from model_3dcnn import CNN3D
from dataset import mt_Dataset

import torch
from torch import nn
from torch.functional import F
from torch.utils.data import DataLoader
from torch.optim import Adam 
from torch.optim.lr_scheduler import ExponentialLR

device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')

def seed_everything(seed=1234):
    #random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def train_model_m2dcnn(model, dataloaders_dict, criterion, optimizer, scheduler, num_epochs, schedule=True):
    model.to(device)
    train_acc = []
    valid_acc = []
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-------------')
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            epoch_loss, epoch_corrects, epoch_acc = 0.0, 0, 0.0
            iteration = 0
            length = len(dataloaders_dict[phase].dataset)

            for inputs, labels in dataloaders_dict[phase]:
                iteration += 1
                optimizer.zero_grad()
                inputs = inputs.float()              
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                with torch.set_grad_enabled(phase == 'train'):
                    # Forward
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    
                    # Backprop
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                    batch_loss = loss.item() * inputs.size(0)  
                    epoch_loss += batch_loss
                    epoch_corrects += torch.sum(preds == labels.data)
                #print('{} : Minibatch {}/{} finished (Minibatch Loss: {:.4f})'.format(datetime.datetime.now(),min(batch_size*iteration,length),length, batch_loss/batch_size))
        
            epoch_loss = epoch_loss / length
            epoch_acc = epoch_corrects.double() /length
            if phase == 'train':
                train_acc.append([epoch_acc,epoch_loss])
            else:
                valid_acc.append([epoch_acc,epoch_loss])
            print('##### {} Loss: {:.4f} Acc: {:.4f} #####'.format(phase, epoch_loss, epoch_acc))
            
        if schedule:
            scheduler.step()
        #torch.save(model.state_dict(), save_path)
        
        # Fast stop
        if valid_acc[-1][1] < 0.1:
            print("Stop!")
            return model, train_acc, valid_acc
        
    return model, train_acc, valid_acc

def test_model(model, dataloaders_dict):
    model.eval()
    model.to(device)
    corrects = 0
    for inputs, labels in tqdm(dataloaders_dict["test"]):
        inputs = inputs.float()
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)
    acc = corrects.double() / len(dataloaders_dict["test"].dataset)
    print('Test Accuracy: {:.4f}'.format(acc))
    return model, acc

def plot_results(train_acc, valid_acc, test_acc, path):
    f,ax = plt.subplots()
    ax.plot(train_acc)
    ax.plot(valid_acc)
    ax.set_ylim(0,1)
    ax.set_title('test: {}'.format(test_acc))
    plt.savefig(path)
    plt.show()

def plot_loss_accuracy(train_accuracy, valid_accuracy, test_accuracy, condition):
    trac,trls,vrac,vrls = [], [], [], []
    for acc,los in train_accuracy:
        trac.append(acc)
        trls.append(los)
    for acc,los in valid_accuracy:
        vrac.append(acc)
        vrls.append(los)

    path_to_image = './results/{}_ACresults.png'.format(condition)
    plot_results(trac, vrac, test_accuracy, path_to_image)
    #send_image(path_to_img=path_to_image, message='Accuracy results')

    path_to_image = './results/{}_LSresults.png'.format(condition)
    plot_results(trls, vrls, test_accuracy, path_to_image)
    #send_image(path_to_img=path_to_image, message='Loss results')


def train_m2dcnn(dataset_path, label_list, condition, batch_size = 128, num_epochs = 300):
    seed_everything()

    # DataLoader
    train_dataset = mt_Dataset(dataset_path[0], label_list[0])
    valid_dataset = mt_Dataset(dataset_path[1], label_list[1])
    test_dataset = mt_Dataset(dataset_path[2], label_list[2])

    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    valid_dataloader = DataLoader(valid_dataset, batch_size = batch_size, shuffle = False)
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
    dataloaders_dict = {"train": train_dataloader, "valid": valid_dataloader, "test": test_dataloader}

    model = M2DCNN(numClass=2, numFeatues=6528, DIMX=53, DIMY=63, DIMZ=17)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(params=model.parameters(),lr=0.001,betas=(0.9, 0.999))
    scheduler = ExponentialLR(optimizer, gamma=0.99)

    model, train_accuracy, valid_accuracy = train_model_m2dcnn(model, dataloaders_dict, criterion,
                                                        optimizer, scheduler, num_epochs = num_epochs)
    model, test_accuracy = test_model(model, dataloaders_dict)

    plot_loss_accuracy(train_accuracy, valid_accuracy, test_accuracy, condition)
    torch.save(model.state_dict(), './results/{}_weights.pth'.format(condition))
    scipy.io.savemat('./results/{}_results.pth'.format(condition),
            {
                'train_accuracy':train_accuracy,
                'valid_accuracy':valid_accuracy,
                'test_accuracy':test_accuracy,
            }
        )
    
    return test_accuracy.cpu().numpy()

def train_3dcnn(dataset_path, label_list, condition, batch_size = 128, num_epochs = 300):
    seed_everything()

    # DataLoader
    train_dataset = mt_Dataset(dataset_path[0], label_list[0], boxcar=True)
    valid_dataset = mt_Dataset(dataset_path[1], label_list[1], boxcar=True)
    test_dataset = mt_Dataset(dataset_path[2], label_list[2], boxcar=True)

    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    valid_dataloader = DataLoader(valid_dataset, batch_size = batch_size, shuffle = False)
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
    dataloaders_dict = {"train": train_dataloader, "valid": valid_dataloader, "test": test_dataloader}

    model = CNN3D()

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(params=model.parameters(),lr=0.001,betas=(0.9, 0.999))
    scheduler = ExponentialLR(optimizer, gamma=0.99)

    model, train_accuracy, valid_accuracy = train_model_m2dcnn(model, dataloaders_dict, criterion,
                                                        optimizer, scheduler, num_epochs = num_epochs)
    model, test_accuracy = test_model(model, dataloaders_dict)

    plot_loss_accuracy(train_accuracy, valid_accuracy, test_accuracy, condition)
    torch.save(model.state_dict(), './results/{}_weights.pth'.format(condition))
    scipy.io.savemat('./results/{}_results.pth'.format(condition),
            {
                'train_accuracy':train_accuracy,
                'valid_accuracy':valid_accuracy,
                'test_accuracy':test_accuracy,
            }
        )
    
    return test_accuracy.cpu().numpy()
