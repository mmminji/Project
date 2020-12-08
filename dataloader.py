import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
import torch
from torch.utils.data import Subset
from torch.utils.data import DataLoader


class GalD(Dataset):

    def __init__(self, root: str, train: bool = True, transform = None, target_transform = None, val_size : float =0.3):
        self.train = train
        self.transform = transform
        self.root = root
        self.labels_name = os.listdir(self.root)
        self.val_size = val_size
        self.le = LabelEncoder()

        self.enc = self.le.fit(self.labels_name)

        edge_files = os.listdir(os.path.join(self.root, self.labels_name[self.labels_name.index('edge')]))
        spiral_files = os.listdir(os.path.join(self.root, self.labels_name[self.labels_name.index('spiral')]))
        smooth_files = os.listdir(os.path.join(self.root, self.labels_name[self.labels_name.index('smooth')]))

        edge_files = list(map(lambda x : os.path.join(self.root, self.labels_name[self.labels_name.index('edge')]) +'/'+ x, edge_files))
        spiral_files = list(map(lambda x : os.path.join(self.root, self.labels_name[self.labels_name.index('spiral')]) +'/'+ x, spiral_files))
        smooth_files = list(map(lambda x : os.path.join(self.root, self.labels_name[self.labels_name.index('smooth')]) +'/'+ x, smooth_files))

        self.gl_files = np.concatenate((
        np.concatenate((np.array(edge_files).reshape(-1,1), np.repeat('edge', len(edge_files)).reshape(-1,1)),axis=1),
        np.concatenate((np.array(spiral_files).reshape(-1,1), np.repeat('spiral', len(spiral_files)).reshape(-1,1)),axis=1),
        np.concatenate((np.array(smooth_files).reshape(-1,1), np.repeat('smooth', len(smooth_files)).reshape(-1,1)),axis=1)), axis=0)

        self.data_len = len(self.gl_files)

    def __getitem__(self, index):
        single_file = self.gl_files[index]

        image_path = single_file[0]
        label = single_file[1]

        as_im = Image.open(image_path)
        label_np = self.enc.transform(np.array([label]))
        label_ten = torch.from_numpy(label_np)
        
        if self.transform:
          as_im = self.transform(as_im)
        #   as_im_ten = torch.from_numpy(as_im).float() ##
          return (as_im, label_ten)

        elif self.transform==None:
        #   as_im = np.asarray(as_im) ##
        #   as_im_ten = torch.from_numpy(as_im).float()  ##
          return (as_im, label_ten)  ##
        #   return (as_im_ten, label_ten)  ##
    
    def __len__(self):
        return self.data_len


class MapDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, map_fn):
        self.dataset = dataset
        self.map = map_fn

    def __getitem__(self, index):
        return (self.map(self.dataset[index][0]), self.dataset[index][1])

    def __len__(self):
        return len(self.dataset)
