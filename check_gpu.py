import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda devices:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
