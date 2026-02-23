import torch as t
import torch.nn as nn
import numpy as np

def _orthogonal_init(module, gain=np.sqrt(2)):
    """
    Orthognal initialisation, skips layers that are not Conv2d or Linear
    """
    for mod in module.modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(mod.weight, gain=gain)

            if mod.bias is not None:
                nn.init.constant_(mod.bias, 0)

class PMAlpha(nn.Module):
    """
    First CNN test model for PacMan project
    2592 * 256 = 663552 parameters
    Small 2 layer model for first run
    Actor and Value heads for model are in 2 separate classes 
    to match Torchrl PPO import expectations
    """
    def __init__(self, num_actions=5):
        super().__init__()
        # PRE RESIZE (224, 240, 3) 224 height, 240 width, 3 RGB input channels 
        # POST RESIZE (3, 84, 84) 3 RGB input channels, 84 height/width
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=8, stride=4)
        #PRE RESIZE h=55, w=59
        # POST RESIZE h=20, w=20
        self.activation1 = nn.SiLU()

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        #PRE RESIZE h=26, w=28
        # POST RESIZE h=9, w=9
        self.activation2 = nn.SiLU()

        #self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=1)
        #PRE RESIZE h=25, w=27
        #self.activation3 = nn.ReLU()

        self.flatten = nn.Flatten() 

        self.fc = nn.Linear(2592, 256) # 32 * 9 * 9 = 2592

        self.activation3 = nn.SiLU()

        _orthogonal_init(self)
    
    def forward(self, x):
        squeezed = False 
        if x.dim() == 3:
            x = x.unsqueeze(0)
            squeezed = True
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.activation2(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.activation3(x)
        if squeezed:
            x = x.squeeze(0)
        return x

class PMPolicy(nn.Module):
    """
    Actor head for model
    Takes features from PMAlpha and outputs action logits
    """
    def __init__(self, backbone, num_actions=5):
        super().__init__()
        self.backbone = backbone
        self.policy_head = nn.Linear(256, num_actions)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.constant_(self.policy_head.bias, 0)
    
    def forward(self, x):
        features = self.backbone(x)
        return self.policy_head(features)

class PMValue(nn.Module):
    """
    Critic head for model 
    Takes features from PMAlpha and outputs state value as scalar
    """
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.value_head = nn.Linear(256, 1)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.constant_(self.value_head.bias, 0)
    
    def forward(self, x):
        features = self.backbone(x)
        return self.value_head(features)

        
        
