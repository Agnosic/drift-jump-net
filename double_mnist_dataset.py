from torch.utils.data import Dataset
import numpy as np
import os
import imageio.v2 as imageio
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from torchvision import datasets

DIGITS_TO_KEY = {0: 0, 2: 1, 4: 2, 9: 3}

def get_prediction_vector_from_dir_name(dir_name):
    label = [0] * 10


    for digit in list(map(int, dir_name)):
        label[digit] += 1
    return np.array(label)

class DoubleMnistDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.file_names = []
        self.labels = []
        for dir_name in os.listdir(root_dir):
            if not dir_name.isdigit():
                continue
            label = get_prediction_vector_from_dir_name(dir_name)
            path_name = os.path.join(root_dir, dir_name)
            for file_name in os.listdir(path_name):
                self.file_names.append(os.path.join(path_name, file_name))
                self.labels.append(label)
        self.transform = transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        # img_name = os.path.join(self.root_dir,
        #                         self.landmarks_frame.iloc[idx, 0])

        image = Image.open(self.file_names[idx])

        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)

        return image, label


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_loader = DataLoader(DoubleMnistDataset('./double_mnist/train', transform=transform_train), batch_size = 2, shuffle=True)

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        print(inputs)
        break