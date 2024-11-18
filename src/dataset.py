import os
from glob import glob
from pathlib import Path
import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms

class MVTec(torch.utils.data.Dataset):
    def __init__(self, root, category, config, is_train=True):
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((config.data.image_size, config.data.image_size)),  
                transforms.ToTensor(), 
                transforms.Lambda(lambda t: (t * 2) - 1)
            ]
        )
        self.config = config
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize((config.data.image_size, config.data.image_size)),
                transforms.ToTensor(),
            ]
        )
        if is_train:
            if category:
                self.image_files = glob(
                    os.path.join(root, category, "train", "good", "*.png")
                )
            else:
                self.image_files = glob(
                    os.path.join(root, "train", "good", "*.png")
                )
        else:
            if category:
                self.image_files = glob(os.path.join(root, category, "test", "*", "*.png"))
            else:
                self.image_files = glob(os.path.join(root, "test", "*", "*.png"))
        self.is_train = is_train

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = self.image_transform(image)
        if(image.shape[0] == 1):
            image = image.expand(3, self.config.data.image_size, self.config.data.image_size)
        if self.is_train:
            label = 'good'
            return image, label
        else:
            if self.config.data.mask:
                if os.path.dirname(image_file).endswith("good"):
                    target = torch.zeros([1, image.shape[-2], image.shape[-1]])
                    label = 'good'
                else :
                    if self.config.data.name == 'MVTec':
                        target = Image.open(
                            image_file.replace("/test/", "/ground_truth/").replace(
                                ".png", "_mask.png"
                            )
                        )
                    else:
                        target = Image.open(
                            image_file.replace("/test/", "/ground_truth/"))
                    target = self.mask_transform(target)
                    label = 'defective'
            else:
                if os.path.dirname(image_file).endswith("good"):
                    target = torch.zeros([1, image.shape[-2], image.shape[-1]])
                    label = 'good'
                else :
                    target = torch.zeros([1, image.shape[-2], image.shape[-1]])
                    label = 'defective'
                
            return image, target, label

    def __len__(self):
        if self.is_train:
            return int(len(self.image_files)*1)
        else:
            return len(self.image_files)


class BTAD(torch.utils.data.Dataset):
    def __init__(self, root, category, config, is_train=True):
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((config.data.image_size, config.data.image_size)),  
                transforms.ToTensor(), # Scales data into [0,1] 
                transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
            ]
        )
        self.config = config
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize((config.data.image_size, config.data.image_size)),
                transforms.ToTensor(), # Scales data into [0,1] 
            ]
        )

        if is_train:
            if category:
                self.image_files = glob(
                    os.path.join(root, category, "train", "ok", "*.png")
                )
                if len(self.image_files) == 0:
                    self.image_files = glob(os.path.join(root, category, "train", "ok", "*.bmp"))
        else:
            if category:
                self.image_files = glob(os.path.join(root, category, "test", "*", "*.png"))
                if len(self.image_files) == 0:
                    self.image_files = glob(os.path.join(root, category, "test", "*", "*.bmp"))

        self.is_train = is_train

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = self.image_transform(image)
        if(image.shape[0] == 1):
            image = image.expand(3, self.config.data.image_size, self.config.data.image_size)
        if self.is_train:
            label = 'good'
            return image, label
        else:
            if self.config.data.mask:
                if os.path.dirname(image_file).endswith("ok"):
                    target = torch.zeros([1, image.shape[-2], image.shape[-1]])
                    label = 'good'
                else :
                    mask_path = image_file.replace("/test/", "/ground_truth/")
                    if os.path.exists(mask_path):
                        target = Image.open(mask_path)
                    else:
                        mask_path = mask_path.replace('.png', '.bmp')
                        if not os.path.exists(mask_path):
                            mask_path = mask_path.replace('.bmp', '.png')
                        target = Image.open(mask_path)
                    target = self.mask_transform(target)
                    label = 'defective'
                
            return image, target, label

    def __len__(self):
        return len(self.image_files)
    
