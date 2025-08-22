from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import cv2 as cv
import torch

class R3DicDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):

        self.Speckles_frame = pd.read_csv(csv_file,header=None)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.Speckles_frame)

    def __getitem__(self, idx):

        Ref_name = os.path.join(self.root_dir,"img", self.Speckles_frame.iloc[idx, 0])
        Def_name = os.path.join(self.root_dir, "img", self.Speckles_frame.iloc[idx, 1])
        Dispx_name = os.path.join(self.root_dir, "Displacement", self.Speckles_frame.iloc[idx, 2])
        Dispy_name = os.path.join(self.root_dir,"Displacement",  self.Speckles_frame.iloc[idx, 3])

        Ref_img = cv.imread(Ref_name,cv.IMREAD_GRAYSCALE)
        Def_img= cv.imread(Def_name, cv.IMREAD_GRAYSCALE)

        Dispx = np.genfromtxt(Dispx_name, delimiter=',')
        Dispy = np.genfromtxt(Dispy_name, delimiter=',')

        Ref_img  = Ref_img
        Def_img = Def_img
        Dispx = Dispx
        Dispy = Dispy

        Ref_img= Ref_img[np.newaxis, ...]
        Def_img = Def_img[np.newaxis, ...]
        Dispx = Dispx[np.newaxis, ...]
        Dispy = Dispy[np.newaxis, ...]


        sample = {'Ref':Ref_img, 'Def': Def_img, 'Dispx': Dispx, 'Dispy': Dispy}

        if self.transform:
            sample = self.transform(sample)

        return  sample



class Normalization(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        Ref, Def, Dispx, Dispy= sample['Ref'], sample['Def'], sample['Dispx'], sample['Dispy']

        self.mean = 0.0
        self.std = 255.0
        self.mean1 = 0.0
        self.std1 = 1.0

        return {'Ref': torch.from_numpy((Ref - self.mean) / self.std).float(),
                'Def': torch.from_numpy((Def - self.mean) / self.std).float(),
                'Dispx': torch.from_numpy((Dispx - self.mean1) / self.std1).float(),
                'Dispy': torch.from_numpy((Dispy - self.mean1) / self.std1).float(),

                }


