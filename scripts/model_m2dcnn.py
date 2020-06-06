import torch
import torch.nn as nn
import torch.nn.functional as F

class mish(nn.Module):
    '''
    @misc{misra2019mish,
    title={Mish: A Self Regularized Non-Monotonic Neural Activation Function},
    author={Diganta Misra},
    year={2019},
    eprint={1908.08681},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
    }
    '''
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.tanh(F.softplus(input))

class Conv2d(nn.Module):
    def __init__(self, input_shape):
        super(Conv2d, self).__init__()
        self.input_shape = input_shape
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=self.input_shape[2], out_channels=16, kernel_size=(3,3), stride=1, padding=0),
            mish(),
            nn.MaxPool2d(kernel_size=(2,2), stride=None, padding=0),
            nn.BatchNorm2d(num_features=16),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), stride=1, padding=0),
            mish(),
            nn.MaxPool2d(kernel_size=(2,2), stride=None, padding=0),
            nn.BatchNorm2d(num_features=32)
        )
    
    def forward(self, inputs):
        outputs = self.conv_block(inputs) #conv
        outputs = outputs.view(outputs.size(0), -1) # flatten
        return outputs

class M2DCNN(nn.Module):
    def __init__(self, numClass, numFeatues, DIMX, DIMY, DIMZ):
        super(M2DCNN, self).__init__()
        self.up_conv = Conv2d(input_shape=(DIMX, DIMY, DIMZ))
        self.front_conv = Conv2d(input_shape=(DIMX, DIMZ, DIMY))
        self.left_conv = Conv2d(input_shape=(DIMY, DIMZ, DIMX))
                
        self.numFeatues = numFeatues
        self.fc = nn.Sequential(
            nn.BatchNorm1d(num_features=self.numFeatues),
            nn.Linear(in_features=self.numFeatues, out_features=128, bias=True),
            mish(),
            nn.BatchNorm1d(num_features=128)
        )

        self.numClass = numClass
        self.clfc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=128, out_features=self.numClass, bias=True)
        )

        self._initialize_weights()

    def forward(self, inputs):
        up_outputs = self.up_conv(inputs) #zyx
        front_outputs = self.front_conv(inputs.transpose(1,2)) #yzx
        left_outputs = self.left_conv(inputs.transpose(1,3).transpose(2,3)) #xzy

        merged = torch.cat((up_outputs,front_outputs,left_outputs),dim=1)
        outputs = self.fc(merged)
        outputs = self.clfc(outputs)
        return outputs

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
    
    def decompose(self, inputs):
        up_outputs = self.up_conv(inputs) #zyx
        front_outputs = self.front_conv(inputs.transpose(1,2)) #yzx
        left_outputs = self.left_conv(inputs.transpose(1,3).transpose(2,3)) #xzy

        merged = torch.cat((up_outputs,front_outputs,left_outputs),dim=1)
        outputs = self.fc(merged)
        return outputs

        


