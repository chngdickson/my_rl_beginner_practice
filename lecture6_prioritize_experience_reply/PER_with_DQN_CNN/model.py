import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import os

class Dueling_DQN_2016_Modified(nn.Module):
    def __init__(self, output_file_name, n_k_stacked, n_actions,lr, init_weights=True, chkpt_dir="model"):
        super().__init__()

        self.name = output_file_name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir,output_file_name+'.pth')
        
        self.cnn = nn.Sequential(nn.Conv2d(n_k_stacked, 32, kernel_size=8, stride=4,bias=False),
                                        nn.ReLU(True),
                                        nn.Conv2d(32, 64, kernel_size=4, stride=2,bias=False),
                                        nn.ReLU(True),
                                        nn.Conv2d(64, 64, kernel_size=3, stride=1,bias=False),
                                        nn.ReLU(True),
                                        nn.Conv2d(64,1024,kernel_size=7,stride=1,bias=False),
                                        nn.ReLU(True)
                                        )
        self.streamA = nn.Linear(512, n_actions) #Actions
        self.streamV = nn.Linear(512, 1) # Value

        if init_weights:
            self._initialize_weights()
            
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.criterion = torch.nn.SmoothL1Loss()
        self.to(self.device)
        

    def forward(self, x):
        x = x.to(self.device)
        x = self.cnn(x)
        sA,sV = torch.split(x,512,dim = 1)
        sA = torch.flatten(sA,start_dim=1)
        sV = torch.flatten(sV, start_dim=1)
        sA = self.streamA(sA) #(B,n_actions)
        sV = self.streamV(sV) #(B,1)
        # combine this 2 values together
        Q_value = sV + (sA - torch.mean(sA,dim=1,keepdim=True))
        return Q_value #(B,n_actions)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)
        self.load_checkpoint(self.name)
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self, filename):
        checkpoint_file = os.path.join(self.checkpoint_dir,filename+'.pth')
        self.load_state_dict(torch.load(checkpoint_file))