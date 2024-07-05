import os
import numpy as np
from torch.optim import Adam
from model import SimpleUnet
import torch
import torch.nn.functional as F
import torch
from dataloader import Data_loader  
from config import *
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SimpleUnet()
model.to(device)
optimizer = Adam(model.parameters(), lr=0.001)
epochs = 30 
trainloader, testloader = Data_loader()

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)


n_steps = 300
betas = linear_beta_schedule(timesteps=n_steps)
# Pre-calculate different terms for closed form
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

def apply_noise(xt: torch.Tensor, t: torch.Tensor, device: torch.device) -> torch.Tuple[torch.Tensor, torch.Tensor]:
    """ 
    Takes an image and a timestep as input and 
    returns the noisy version of it
    """
    noise = torch.randn_like(xt, device= device)
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod.gather(-1, t).reshape(t.shape[0],1,1)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod.gather(-1, t).reshape(t.shape[0],1,1)
    # mean + variance
    noise_image= sqrt_alphas_cumprod_t.to(device) * xt.to(device) + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device)
    return noise_image, noise
def save_model(model, optimizer, epoch, path='checkpoint.pth'):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

# sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
# posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

if __name__ == "__main__":
    
    for epoch in range(epochs):
        for step, batch in enumerate(trainloader):
            x = batch[0]
            optimizer.zero_grad()
            t = torch.randint(0, n_steps, (x.shape[0],),device=device).long()
            x_noise_t, noise = apply_noise(xt=x, t=t, device=device)
            noise_pred = model(x_noise_t, t)
            loss = F.l1_loss(x_noise_t, noise_pred)
            loss.backward()
            optimizer.step()

            if epoch % 5 == 0 and step == 0:
                print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
            if (epoch + 1) % 5 == 0:
                save_path = os.path.join(save_weight, f'model_epoch_{epoch+1}.pth')
                save_model(model, optimizer, epoch+1, save_path)
                print(f"Model saved at {save_path}")
    