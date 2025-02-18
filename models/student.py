
### models/student.py ###
import torch
import torch.nn as nn
import torch.nn.functional as F


class StudentModel(nn.Module):
    """
    Lightweight CNN Student Model
    """

    def __init__(self):
        super(StudentModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 32 * 32, 256)
        self.output_layer = nn.Linear(256, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.output_layer(x)
