from torch.utils.data import Dataset
from PIL import Image
from os import listdir, walk
from os.path import join
from torch import randn

class ImageDataset(Dataset):
    def __init__(self, path="../data/superhero", transform=None, size=None, latent_vector=None):
        self.paths = [join(folder, image_path) for folder, _, fn in walk(path) for image_path in fn]        
        self.transform = transform
        
        self.size, self.latent_vector = size, latent_vector
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        img = Image.open(self.paths[index]).convert("RGB")
        img = self.transform(img)# if self.transform != None else transforms.ToTensor()(img)
        
        if None in [self.size, self.latent_vector]:
            return img
        else:
            return randn(self.latent_vector, self.size, self.size), img
