import numpy as np

import os

import torch
from torch.utils.data import Dataset


class Simple2DDataset(Dataset):
    def __init__(self, split='train'):
        super().__init__()
        # Read either train or validation data from disk based on split parameter using np.load.
        # Data is located in the folder "data".
        # Hint: you can use os.path.join to obtain a path in a subfolder.
        # Save samples and annotations to class members self.samples and self.annotations respectively.
        # Samples should be an Nx2 numpy array. Annotations should be Nx1.
        assert split in ['train', 'valid'], f'Split parameters "{split}" must be either "train" or "valid".'
        npzpath = os.path.join("data",split + ".npz")
        npz_file = np.load(npzpath)
        self.samples = npz_file["samples"]
        self.annotations = npz_file["annotations"]

        #raise NotImplementedError()

    def __len__(self):
        # Returns the number of samples in the dataset.
        return self.samples.shape[0]

    def __getitem__(self, idx):
        # Returns the sample and annotation with index idx.

        sample = self.samples[idx]
        annotation = self.annotations[idx]
        #raise NotImplementedError()

        # Convert to tensor.
        return {
            'input': torch.from_numpy(sample).float(),
            'annotation': torch.from_numpy(annotation[np.newaxis]).float()
        }


class Simple2DTransformDataset(Dataset):
    def __init__(self, split='train'):
        super().__init__()
        assert split in ['train', 'valid'], f'Split parameters "{split}" must be either "train" or "valid".'
        npzpath = os.path.join("data",split + ".npz") #Step 2.1 reading train.npz file
        npz_file = np.load(npzpath)
        self.samples = npz_file["samples"]
        self.annotations = npz_file["annotations"]
        # Read either train or validation data from disk based on split parameter.
        # Data is located in the folder "data".
        # Hint: you can use os.path.join to obtain a path in a subfolder.
        # Save samples and annotations to class members self.samples and self.annotations respectively.
        # Samples should be an Nx2 numpy array. Annotations should be Nx1.
        #raise NotImplementedError()

    def __len__(self):
        # Returns the number of samples in the dataset.
        return self.samples.shape[0]

    def __getitem__(self, idx):
        # Returns the sample and annotation with index idx.
        #raise NotImplementedError()
        sample = self.samples[idx]      #Step 2.1: recovering a single data sample
        annotation = self.annotations[idx]
        # Transform the sample to a different coordinate system.
        sample = transform(sample)

        # Convert to tensor.
        return {
            'input': torch.from_numpy(sample).float(),
            'annotation': torch.from_numpy(annotation[np.newaxis]).float()
        }


def transform(sample):
    #raise NotImplementedError()

    new_sample = np.array([sample[0]**2 + sample[1]**2, sample[1] / sample[0]])

    return new_sample
