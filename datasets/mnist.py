import torch
from torchvision import datasets, transforms


def get_mnist(train, batch_size=1024):
    """Get MNIST dataset loader."""
    # image pre-processing
    pre_process = transforms.Compose([
                       transforms.Resize((32, 32)),
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])

    # dataset and data loader
    mnist_dataset = datasets.MNIST(root='cache/data/',
                                 train=train,
                                 transform=pre_process,
                                 download=True)


    data_loader = torch.utils.data.DataLoader(
        dataset=mnist_dataset,
        batch_size=batch_size,  # train:256, test: 1024
        num_workers=8,
        shuffle=True,
        )

    return data_loader
