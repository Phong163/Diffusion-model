import os
from PIL import Image
from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

IMG_SIZE = 64
BATCH_SIZE = 128
dir_path = r"C:\Users\OS\Desktop\pytorch-stable-diffusion\data"

class CustomDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image
def Data_loader():
    # Get list of all image paths
    all_image_paths = [os.path.join(dir_path, img) for img in os.listdir(dir_path)]

    # Split data into train and test sets
    train_paths, test_paths = train_test_split(all_image_paths, test_size=0.2, random_state=42)

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

    # Create train and test datasets
    train_dataset = CustomDataset(train_paths, transform=transform)
    test_dataset = CustomDataset(test_paths, transform=transform)

    # Create DataLoader instances
    trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    return trainloader,testloader
