import torch
from src.model import SimpleCNN

model = SimpleCNN()
model.load_state_dict(torch.load("experiments/checkpoints/epoch_9.pt"))
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

# TorchScript export
scripted = torch.jit.trace(model, dummy_input)
scripted.save("experiments/model_scripted.pt")

# ONNX export
torch.onnx.export(model, dummy_input, "experiments/model.onnx", input_names=["input"], output_names=["output"])