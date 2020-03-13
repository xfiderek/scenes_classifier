import pandas as pd 
import torch
from torch.utils.data import DataLoader
import numpy as np 
import PIL
import os 

from .transforms import eval_transforms, calc_mean_transforms
from .classes_dict import name_to_lbl

class IntelDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, csv_path, mean=[0.4302, 0.4575, 0.4539], 
                    std=[0.2101, 0.2092, 0.2191], transforms=[], labeled=True):
        self.dataset = pd.read_csv(csv_path).dropna()
        self.data_dir = data_dir
        self.transforms = transforms
        self.labeled = labeled
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        rel_path = self.dataset.iloc[idx, 1]
        img = PIL.Image.open(os.path.join(self.data_dir, rel_path))
        img = self.transforms(img)
        
        if self.labeled:
            label = np.array([self.dataset.iloc[idx, 0]])
            return (img, label)
        else:
            return (img, rel_path) 

class DatasetUtils:
    @staticmethod
    def generate_labeled_csv(root_dir, rel_data_dir, csv_path, classes=name_to_lbl):
        df = pd.DataFrame(columns=["class", "path"])
        for c in classes:
            rel_dir = os.path.join(rel_data_dir, c)
            directory = os.path.join(root_dir, rel_dir)
            img_names = os.listdir(directory)
            img_names = list(map(lambda img_name: os.path.join(rel_dir, img_name), img_names))
            df_frac = pd.DataFrame(data={"class":[classes[c] for _ in range(len(img_names))], 
                                            "path":img_names})
            df = df.append(df_frac, ignore_index=True)
        #just to shuffle data
        df = df.sample(frac=1).reset_index(drop=True)
        df.to_csv(csv_path, index=False, header=True)

    #creates csv for prediction dataset
    @staticmethod
    def generate_unlabeled_csv(root_dir, rel_data_dir, csv_path):
        direct = os.path.join(root_dir, rel_data_dir)
        img_names = os.listdir(direct)
        img_names = list(map(lambda img_name: os.path.join(rel_data_dir, img_name), img_names))
        df = pd.DataFrame(data={"path":img_names})
        
        #to push img names to second column
        df = df.reset_index() 
        df.to_csv(csv_path, index=False, header=True)

    @staticmethod
    def get_mean_and_std(data_dir, csv_path):
        """
        mean and std has been already calculated
        mean = [0.43, 0.457, 0.454]
        std = [0.21, 0.209, 0.22]
        """
        dataset = IntelDataset(data_dir, csv_path, pil_transforms=calc_mean_transforms())
        loader = DataLoader(dataset, batch_size=10)
        with torch.no_grad():
            mean = 0.
            std = 0.
            samples = 0.
            for (img, _) in loader:
                samples_i = img.size(0)
                img = img.view(samples_i, img.size(1), -1)
                mean += img.mean(2).sum(0)
                std += img.std(2).sum(0)
                samples += samples_i

            mean /= samples
            std /= samples
            return (mean, std)