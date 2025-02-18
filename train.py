### train.py ###
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.teacher import TeacherModel
from models.student import StudentModel
from utils.loss import DistillationLoss
from utils.dataset import CancerDataset
from config import *

# Initialize Models
teacher_model = TeacherModel().to(DEVICE)
teacher_model.eval()
student_model = StudentModel().to(DEVICE)

# Optimizer & Loss
optimizer = optim.Adam(student_model.parameters(), lr=LEARNING_RATE)
criterion = DistillationLoss()

# Dummy DataLoader (Replace with actual dataset)
data_loader = DataLoader(CancerDataset(
    [...], [...]), batch_size=BATCH_SIZE, shuffle=True)

# Training Loop
for epoch in range(EPOCHS):
    student_model.train()
    for images, labels in data_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        with torch.no_grad():
            teacher_logits = teacher_model(images)
        student_logits = student_model(images)
        loss = criterion(student_logits, teacher_logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Save Model
torch.save(student_model.state_dict(), "student_model.pth")
print("Training Complete. Student Model Saved.")
