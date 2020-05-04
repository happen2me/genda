from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from models.generator import Generator
from models.lenet import LeNet5
from models.lenet_half import LeNet5Half
from utils import partial_load

generator_path = 'cache/models/generator_with_do.pt'
classifier_path = 'cache/models/teacher_with_do.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GenImgs(Dataset):
    def __init__(self, latent_dim=100, length=1000000):
        self.length = length
        self.latent_dim = latent_dim
        self.generator = partial_load(Generator, generator_path)
        self.classifier = partial_load(LeNet5, classifier_path)

    def __getitem__(self, index):
        rand = torch.randn(1, self.latent_dim).to(device)
        img = self.generator(rand)
        label = torch.squeeze(self.classifier(img), dim=0)
        label = torch.argmax(label, dim=0)
        img = torch.squeeze(img, dim=0)
        return img.detach(), label

    def __len__(self):
        return self.length


def get_genimg(train, batch_size=256, shuffle=True):
    """Get feature dataset loader."""
    # image pre-processing

    # dataset and data loader
    dataset = GenImgs()

    validation_split = 0.2
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    if shuffle:
        np.random.seed(None)
        np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    if train:
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    else:
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)
    return data_loader


if __name__ == "__main__":
    data_loader = get_genimg(True, 50)
    for data, label in data_loader:
        print("data.shape: ", data.shape)
        print("label.shape : ", label.shape)
        break
