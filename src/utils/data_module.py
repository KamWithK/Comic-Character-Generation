import math

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from pytorch_lightning import LightningDataModule

class DataModule(LightningDataModule):
    def __init__(self, path="../data", split=0.2, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]), batch_size=16, num_workers=4):
        super().__init__()
        
        self.path, self.split, self.transform = path, split, transform
        self.batch_size, self.num_workers = batch_size, num_workers
    
    # Called once    
    def prepare_data(self):
        self.full_dataset = ImageFolder(self.path, transform=self.transform)
        
        self.train_length = math.floor(len(self.full_dataset) * (1-self.split))
        self.val_length = len(self.full_dataset) - self.train_length
        
    # Called once per GPU
    def setup(self, stage=None):
        self.train_dataset, self.val_dataset = random_split(self.full_dataset, (self.train_length, self.val_length))
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
