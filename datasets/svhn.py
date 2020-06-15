import torch
from torchvision import datasets, transforms


def get_svhn(train, batch_size):

    pre_process = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    dataset = datasets.SVHN(root='cache/data/svhn/',
                            split='train' if train else 'test',
                            transform=pre_process,
                            download=True)

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,  # train:256, test: 1024
        num_workers=1,
        shuffle=True,
    )

    return data_loader