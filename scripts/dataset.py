import numpy as np
from torch.utils.data import Dataset
import scipy.stats

class mt_Dataset(Dataset):
    def __init__(self, file_list, label_list, boxcar=False):
        self.boxcar = boxcar
        self.file_list = file_list
        self.label_list = label_list

    def __getitem__(self, index):
        img = self.normalize_data(np.load(self.file_list[index]))
        target = np.argmax(self.label_list[index])
        return img, target
    
    def __len__(self):
        return len(self.label_list)
    
    def normalize_data(self, data):
        # Data auguentaion
        if self.boxcar:
            data = data[:, :, 2:-4, 4:8]
        else:
            data = np.mean(data[:, :, 2:-4, 4:8],axis=3)
        
        data = scipy.stats.zscore(data, axis=None)
        data[~ np.isfinite(data)] = 0

        if self.boxcar:
            return data.transpose(3, 0, 1, 2)
        else:
            return data.transpose(2, 1, 0)