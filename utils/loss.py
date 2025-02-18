### utils/loss.py ###
import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss combining Cross-Entropy and KL Divergence.
    """

    def __init__(self, temperature=5.0, alpha=0.5):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits, true_labels):
        soft_targets = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        kd_loss = self.kl_div(soft_targets, soft_teacher)
        ce_loss = F.cross_entropy(student_logits, true_labels)
        return self.alpha * ce_loss + (1 - self.alpha) * kd_loss
