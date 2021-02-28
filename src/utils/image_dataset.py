from torch.utils.data import Dataset
from PIL import Image
from os import listdir
from torch import randn

class ImageDataset(Dataset):
    def __init__(self, path="../data/superhero", transform=None, latent_vector=100):
        self.paths = [f"{path}/{filename}" for filename in listdir(path)]
        
        self.transform = transform
        
        self.latent_vector = latent_vector
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        img = Image.open(self.paths[index]).convert("RGB")
        img = self.transform(img)# if self.transform != None else transforms.ToTensor()(img)
        
        noise = randn(self.latent_vector, 1, 1)
        
        return noise, img