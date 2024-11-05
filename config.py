import torch
from torchvision.transforms import transforms

IMAGE_SIZE = 100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
STEPS = 10000
LR = 1e-3
ALPHA = 1
BETA = 0.01

loader = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])