from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms

class ImgDataset(Dataset):
    def __init__(self, datas, labels):
        self.datas = datas

        self.labels = labels
        self.transform = transforms.Compose([                                   
            transforms.ToTensor()
        ])
    def __len__(self):
        return len(self.datas)
    def __getitem__(self, index):
        X = self.datas[index]
        X = self.transform(X)
        if self.labels is not None:
            Y = self.labels[index]
            return X, Y
        else:
            return X