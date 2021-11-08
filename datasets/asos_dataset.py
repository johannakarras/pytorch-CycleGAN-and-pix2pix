"""Dataset class template
This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
import glob, cv2
import torch
import random
import numpy as np
from data.base_dataset import BaseDataset, get_transform
# from data.image_folder import make_dataset
# from PIL import Image


class AsosDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.
        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        Returns:
            the modified parser.
        """
        parser.add_argument('--frames_folder', type=str, default=None, help='Path to directory of frames and poses')
        parser.add_argument('--flows_folder', type=str, default=None, help='Path to directory of optical flows')
        parser.set_defaults(max_dataset_size=10, new_dataset_option=2.0)  # specify dataset-specific default values
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)

        frames_folder = '/content/drive/MyDrive/Research/Datasets/Asos/Data/Poses/dress-1635455208-11'
        flows_folder = '/content/drive/MyDrive/Research/Datasets/Asos/Data/Flows/dress-1635455208-11-resized'

        self.frames = [path for path in glob.glob(frames_folder+'/*') if ('_pose' not in path)]
        self.poses = [path for path in glob.glob(frames_folder+'/*') if ('_pose' in path)]
        self.flows = glob.glob(flows_folder + '/*')
        
        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index -- a random integer for data indexing
        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.
        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        # create input
        frame = cv2.resize(cv2.imread(self.frames[index]), (256, 256))
        pose1 = cv2.resize(cv2.imread(self.poses[index]), (256, 256))
        pose2 = cv2.resize(cv2.imread(self.poses[index+1]), (256, 256))
        gray_pose_1 = cv2.cvtColor(pose1, cv2.COLOR_BGR2GRAY)
        gray_pose_2 = cv2.cvtColor(pose2, cv2.COLOR_BGR2GRAY)
        r, g, b = cv2.split(frame)
        frame_pose = cv2.merge((r, g, b, gray_pose_1, gray_pose_2))
        frame_pose = np.expand_dims(np.moveaxis(np.array(frame_pose), -1, 0), 0)
        frame_pose = torch.Tensor(frame_pose)

        # create output
        frame_2 = cv2.resize(cv2.imread(self.frames[index+1]), (256, 256))
        frame_2 = np.expand_dims(np.moveaxis(np.array(frame_2), -1, 0), 0)
        frame_2 = torch.Tensor(frame_2)
        flow = np.load(self.flows[index])
        flow = torch.Tensor(flow)

        path = self.frames[index]    # needs to be a string
        data_A = frame_pose    # needs to be a tensor
        data_B = flow    # needs to be a tensor

        return {'A': data_A, 'B': flow, 'A_paths': path}

    def __len__(self):
        """Return the total number of images."""
        return min(len(self.frames), len(self.poses))
