import torch
from torchvision import datasets, transforms


def get_usps(train, batch_size=1024):
    """Get USPS dataset loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.Resize(32),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,), (0.5,))])

    # dataset and data loader
    usps_dataset = datasets.USPS(root='cache/data/',
                                 train=train,
                                 transform=pre_process,
                                 download=True)

    usps_data_loader = torch.utils.data.DataLoader(
        dataset=usps_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8)

    return usps_data_loader