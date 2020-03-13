import torch 
from torch.utils.data import DataLoader
import PIL
import pandas as pd 

from .model import Model 
from .transforms import eval_transforms
from .IntelDataset import IntelDataset

class Classifier:
    def __init__(self, model_path, disable_cuda=False):
        self.model = Model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.model.load_state_dict(torch.load(model_path)["model_state_dict"])
        self.mean = torch.load(model_path)["mean"]
        self.std = torch.load(model_path)["std"]

        self.model = self.model.to(self.device)         
        self.model.eval()
        self.transforms = eval_transforms(self.mean, self.std)
        
    def classify_img(self, img_path) -> int:
        img = PIL.Image.open(img_path)
        tensor = self.transforms(img)
        label = self.classify_batch(tensor)
        tensor = tensor.to(self.device)        
        with torch.no_grad():
            output = self.model(tensor)
        label = torch.argmax(output, dim=1)
        
        return (label.item())

    def classify_from_csv(self, data_dir, csv_path, write_path="") -> pd.DataFrame:
        df = pd.DataFrame(columns=["path", "label"])
        dataset = IntelDataset(data_dir, csv_path,self.mean, self.std, transforms=self.transforms, labeled=False)
        loader = DataLoader(dataset, batch_size=1, pin_memory=True)

        with torch.no_grad():
            for (img, img_path) in loader:
                #img_path is list, but batch size is 1
                img_path = img_path[0]
                img = img.to(self.device)
                output = self.model(img)
                label = torch.argmax(output).item()
                df = df.append({"path":img_path, "label":label}, ignore_index=True)
        if write_path:
            df.to_csv(write_path, index=False, header=True)

        return df
