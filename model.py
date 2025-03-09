import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTmodel(nn.Module):
    def __init__(self) -> None :
        super().__init__()

        self.conv1 = nn.Conv2d(1,1,3)
        self.conv2 = nn.Conv2d(1,1,3)

        self.pool = nn.MaxPool2d(2,1)

        self.fc1 = nn.Linear(484,256)
        self.fc2 = nn.Linear(256, 10)

        self.dr = nn.Dropout(p=0.2)
        return None

    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.dr(x)
        x = self.fc2(x)

        return x
