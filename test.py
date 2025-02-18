import torch
from models.student import StudentModel


def test_student_model():
    model = StudentModel()
    input_tensor = torch.randn(1, 3, 32, 32)
    output = model(input_tensor)
    assert output.shape == (1, 2), "Output shape mismatch"
