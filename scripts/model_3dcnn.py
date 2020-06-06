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

class BasicBlock(nn.Module):
    def __init__(self, n_in, n_out, stride = 1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(n_in, n_out, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm3d(n_out)
        self.mish1 = mish()
        self.conv2 = nn.Conv3d(n_out, n_out, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm3d(n_out)

        self.mish2 = mish()
        if stride != 1 or n_out != n_in:
            self.shortcut = nn.Sequential(
                nn.Conv3d(n_in, n_out, kernel_size = 1, stride = stride),
                nn.BatchNorm3d(n_out))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.mish1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.mish2(out)
        return out
    
class CNN3D(nn.Module):
    def __init__(self, block=BasicBlock, inplanes=4, n_classes=2):
        super(CNN3D, self).__init__()
        self.n_classes = n_classes
        
        self.preBlock = nn.Sequential(
            nn.Conv3d(inplanes, 24, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(24),
            mish()
        )
        
        self.layer_1 = self._make_layer(block,  24, 32, 2)
        self.layer_2 = self._make_layer(block, 32, 64, 2, pooling=True)
        self.layer_3 = self._make_layer(block, 64, 64, 2, pooling=True)
        self.layer_4 = self._make_layer(block, 64, 128, 2, pooling=True)
        
        self.post_conv = nn.Conv3d(128, 64, kernel_size=(4, 4, 2), stride=(1, 1, 1))
        
        self.classifier = nn.Sequential(
            nn.Linear(64, 64),
            mish(),
            nn.Dropout(),
            nn.Linear(64, self.n_classes)
        )            
        
        self._initialize_weights()
        
    def _make_layer(self, block, planes_in, planes_out, num_blocks, pooling=False, drop_out=False):
        layers = []
        if pooling:
            layers.append(block(planes_in, planes_out, stride=2))
        else:
            layers.append(block(planes_in, planes_out))
        for _ in range(num_blocks - 1):
            layers.append(block(planes_out, planes_out))
            
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                _ = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                
    def forward(self, x):
        x = self.preBlock(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.post_conv(x)
        x = x.view(-1, 64 * 1)        
        x = self.classifier(x)
        
        return x