import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import numpy as np


class UnalignedMaskDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets with their corresponding mask.

    It requires 4 directories to host training images from domain A '/path/to/data/trainA' and their mask
    '/path/to/data/trainA_mask/', from domain B '/path/to/data/trainB' and their mask '/path/to/data/trainB_mask/'
    respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_A_mask = os.path.join(opt.dataroot, opt.phase + 'A_mask')  # create a path '/path/to/data/trainA_mask'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'
        self.dir_B_mask = os.path.join(opt.dataroot, opt.phase + 'B_mask')  # create a path '/path/to/data/trainB_mask'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.A_paths_mask = sorted(make_dataset(self.dir_A_mask, opt.max_dataset_size))   # load images from '/path/to/data/trainA_mask'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.B_paths_mask = sorted(make_dataset(self.dir_B_mask, opt.max_dataset_size))    # load images from '/path/to/data/trainB_mask'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, A_mask, B, B_mask, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            A_mask (tensor)  -- a mask of the image A for the loss
            B (tensor)       -- its corresponding image in the target domain
            B_mask (tensor)  -- a mask of the image A for the loss
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        index_A = index % self.A_size
        A_path = self.A_paths[index_A]  # make sure index is within then range
        A_path_mask = self.A_paths_mask[index_A]
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        B_path_mask = self.B_paths_mask[index_B]

        # get the images
        A_img = Image.open(A_path).convert('RGB')
        A_mask = np.array(Image.open(A_path_mask))
        B_img = Image.open(B_path).convert('RGB')
        B_mask = np.array(Image.open(B_path_mask))
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        return {'A': A, 'A_mask': A_mask, 'B': B, 'B_mask': B_mask, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
