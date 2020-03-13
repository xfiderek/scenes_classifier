import torch 
import torchvision.transforms as transforms

#transform applied before changing pil image to tensor 
def training_transforms(mean, std):
    return transforms.Compose(
        [
            transforms.RandomCrop(128,pad_if_needed=True, padding_mode="edge"),
            transforms.Resize((64,64)),
            transforms.ColorJitter(brightness=0.1,saturation=0.1,contrast=0.1, hue=0.05),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(), 
            transforms.Normalize(mean=mean, std=std)
        ]
        )

def eval_transforms(mean, std):
    return transforms.Compose(
        [
            transforms.Resize((64 , 64)),
            transforms.ToTensor(), 
            transforms.Normalize(mean=mean, std=std)
        ]
        )

def calc_mean_transforms(mean, std):
    return transforms.Compose(
        [
            transforms.Resize((64 , 64)),
            transforms.ToTensor(), 
        ]
        )
