### models/teacher.py ###
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from ultralytics import YOLO

class TeacherModel(nn.Module):
    """
    Teacher Model: Combination of MViT, ViT, YOLO, and CNN.
    """
    def __init__(self):
        super(TeacherModel, self).__init__()
        self.mvit = models.video.mvit_v2_s(weights="DEFAULT")
        self.vit = models.vit_b_16(weights="IMAGENET1K_V1")
        self.yolo = YOLO("yolov8n.pt")
        self.cnn = models.resnet18(weights="IMAGENET1K_V1")
        self.fc = nn.Linear(1000 + 1000 + 512 + 1000, 256)
        self.output_layer = nn.Linear(256, 2)

    def forward(self, x):
        x_mvit = self.mvit(x)
        x_vit = self.vit(x)
        x_yolo = torch.tensor(self.yolo(x).numpy())  # Convert YOLO output to tensor
        x_cnn = self.cnn(x)
        x_fused = torch.cat([x_mvit, x_vit, x_yolo, x_cnn], dim=1)
        x_fused = F.relu(self.fc(x_fused))
        return self.output_layer(x_fused)
