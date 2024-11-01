import torch.utils
from utils import load_image
from VGG import VGG
import config
from torch import optim
import torch
from torchvision.utils import save_image
from tqdm import tqdm

original_image = load_image("batman.png")
style_image = load_image("starry_night.png")
# generated = original_image.clone().requires_grad_(True)
generated = torch.randn(original_image.shape, device=config.DEVICE, requires_grad=True)

model = VGG().to(config.DEVICE).eval()

optimiser = optim.Adam([generated], lr = config.LR)

for epoch in tqdm(range(config.STEPS), total=config.STEPS):
    generated_features = model(generated)
    original_image_features = model(original_image)
    style_image_features = model(style_image)

    style_loss = content_loss = 0

    for generated_feature, original_image_feature, style_image_feature in zip(generated_features, original_image_features, style_image_features):
        batch_size, channel, height, width = generated_feature.shape

        content_loss += torch.mean((generated_feature - original_image_feature)**2)

        generated_gram_matrix = generated_feature.view(channel, height*width).mm(
                                generated_feature.view(channel, height*width).t()
                                )
        
        style_gram_matrix = style_image_feature.view(channel, height*width).mm(
                                style_image_feature.view(channel, height*width).t()
                                )
        
        style_loss += torch.mean((generated_gram_matrix - style_gram_matrix)**2)
    
    total_loss = config.ALPHA * content_loss + config.BETA * style_loss

    optimiser.zero_grad()
    total_loss.backward()
    optimiser.step()

    if epoch % 200 == 0 or epoch == config.STEPS-1: 
        print(total_loss.item())
        save_image(generated, f"generated2_{epoch}.png")
