from torch.utils.data import Dataset
from PIL import Image
from os import listdir
from torch import randn

class ImageDataset(Dataset):
    def __init__(self, path="../data/superhero", transform=None, size=None, latent_vector=None):
        self.paths = [f"{path}/{filename}" for filename in listdir(path)]
        
        self.transform = transform
        
        self.size, self.latent_vector = size, latent_vector
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        img = Image.open(self.paths[index]).convert("RGB")
        img = self.transform(img)# if self.transform != None else transforms.ToTensor()(img)

        return img if None in [self.size, self.latent_vector] else randn(self.latent_vector, self.size, self.size), img
