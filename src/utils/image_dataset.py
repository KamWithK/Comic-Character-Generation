from torch.utils.data import Dataset
from PIL import Image, ImageFile
from os import walk
from os.path import join
from torch import randn

ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImageDataset(Dataset):
    def __init__(self, path="../data/superhero", max_images=None, transform=None, size=None, latent_vector=None):
        self.all_paths = [join(folder, image_path) for folder, _, fn in walk(path) for image_path in fn]
        self.transform = transform

        self.paths = self.all_paths if max_images == None else self.all_paths[:max_images]
        
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
